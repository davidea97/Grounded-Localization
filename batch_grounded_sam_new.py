import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from glob import glob
import yaml
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image = transform(image_pil)
    return image_pil, image

def load_model(device, checkpoint):
    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_phrases_from_posmap_batch(logits_filt, tokenized, tokenizer, text_threshold):
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase)
    return pred_phrases

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip() + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]
    tokenized = model.tokenizer(caption)
    pred_phrases = get_phrases_from_posmap_batch(logits_filt, tokenized, model.tokenizer, text_threshold)
    return boxes_filt, pred_phrases

"""def save_mask_data(output_dir, mask_list, box_list, label_list, base_name):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0]] = value + idx + 1
    mask_img_np = mask_img.cpu().numpy().astype(np.uint8)
    plt.imsave(os.path.join(output_dir, 'masks', f'{base_name}_mask.png'), mask_img_np, cmap='gray')
    json_data = [{'value': value, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1
        if '(' in label and label.endswith(')'):
            name, logit = label.split('(')
            logit = logit[:-1]
        else:
            name, logit = label, 0.0
        json_data.append({'value': value, 'label': name, 'logit': float(logit), 'box': box.tolist()})
    with open(os.path.join(output_dir, 'json_masks/' + base_name + '_mask.json'), 'w') as f:
        json.dump(json_data, f)"""

def save_mask_and_json(mask_list, label_list, box_list, output_dir, base_name):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:], dtype=torch.uint8)
    
    # Vectorized mask processing
    mask_np = mask_list.cpu().numpy()  # Bring mask_list to CPU only once
    indices = np.arange(1, len(mask_list) + 1)
    
    for idx, mask in enumerate(mask_np):
        mask_img[mask[0]] = indices[idx]

    mask_img_np = mask_img.numpy()  # Convert to NumPy only once
    plt.imsave(os.path.join(output_dir, 'masks', f'{base_name}_mask.png'), mask_img_np, cmap='gray')
    
    # JSON data preparation
    json_data = [{'value': value, 'label': 'background'}]
    for idx, (label, box) in enumerate(zip(label_list, box_list)):
        value = idx + 1
        if '(' in label and label.endswith(')'):
            name, logit = label.split('(')
            logit = float(logit[:-1])
        else:
            name, logit = label, 0.0
        json_data.append({'value': value, 'label': name.strip(), 'logit': logit, 'box': box.tolist()})

    with open(os.path.join(output_dir, 'json_masks', f'{base_name}_mask.json'), 'w') as f:
        json.dump(json_data, f)

def get_object_identifier(filepath):
    filename = filepath.split('/')[-1]
    object_identifier = filename.split('_')[-1]
    object_identifier = object_identifier.split('.')[0]
    return object_identifier

def extract_image_number(filename):
    match = re.search(r'original_frame_(\d+)\.jpg', filename)
    if match:
        return int(match.group(1))
    return None

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_pil, image_tensor = load_image(image_path, self.transform)
        return image_tensor, image_path

def get_object_identifier(filepath):
    # Split by '/' to isolate the filename
    filename = filepath.split('/')[-1]
    # Split by '_' to get the last part of the filename
    object_identifier = filename.split('_')[-1]
    # Remove the file extension
    object_identifier = object_identifier.split('.')[0]
    return object_identifier

def process_batch(batch, model, box_threshold, text_threshold, device, predictor, output_dir):
    images, paths = batch
    images = images.to(device)

    batch_boxes = []
    for image, image_path in zip(images, paths):
        #print("Image path: ", image_path)
        text_prompt = get_object_identifier(image_path)
        #print("Text prompt: ", text_prompt)
        dino_time = time.time()
        boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device)
        print(f"Dino time: {time.time() -dino_time} seconds!")
        if boxes_filt.nelement() == 0:
            print(f"No bounding boxes detected for {image_path}.")
            continue
        batch_boxes.append((boxes_filt, image_path, pred_phrases))

    for boxes_filt, image_path, pred_phrases in batch_boxes:
        image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        set_image_time = time.time()
        predictor.set_image(image_cv)
        print(f"Time setting image: {time.time() - set_image_time} seconds")
        scale_tensor = torch.tensor([image_cv.shape[1], image_cv.shape[0], image_cv.shape[1], image_cv.shape[0]], device=device)
        boxes_filt = boxes_filt * scale_tensor
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv.shape[:2]).to(device)

        masks, scores, logits_sam = predictor.predict_torch(None, None, transformed_boxes, multimask_output=False)
        #colored_mask = np.zeros_like(image_cv)
        #colored_mask[masks[0].cpu().numpy()[0]] = (0, 255, 0)
        #combined_image = cv2.addWeighted(image_cv, 0.7, colored_mask, 0.3, 0)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        #save_mask_data(output_dir, masks, transformed_boxes, pred_phrases, base_name)
        save_mask_and_json(masks, pred_phrases, transformed_boxes, output_dir, base_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument('--config', type=str, default='config/grounded_sam_config.yaml', help='Path to configuration file')
    parser.add_argument('--input', type=str, default='image', help='Input file type: video "video" or set of images "image"')
    parser.add_argument('--video_path', type=str, default='panda_video.mp4', help='Video input file type')

    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_dir"] + "/masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/json_masks", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Fixed size for all images
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    box_threshold = config['box_threshold']
    text_threshold = config['text_threshold']
    output_dir = config['output_dir']

    model = load_model(device=device, checkpoint=config["dino_checkpoint"])
    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"]).to(device)
    predictor = SamPredictor(sam)

    image_paths = glob(os.path.join(config["input_dir"], '*' + config['image_format']))
    image_paths.sort()

    dataset = ImageDataset(image_paths, transform)
    batch_size = config['batch_size']
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=1, pin_memory=True)

    for batch in tqdm(dataloader):
        start_time = time.time()
        process_batch(batch, model, config['box_threshold'], config['text_threshold'], device, predictor, config["output_dir"])
        #print(f"Time spent for a batch of {batch_size}: {time.time()-start_time} seconds!")
        