import argparse
import os
import sys
import numpy as np
import json
sys.path.append("..")
import torch
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt
from glob import glob
import yaml
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import sam_model_registry, SamPredictor
from utils.DatasetILSVRC2017 import ImageDetDataset, SAMImageDetDataset, SAMDataLoader
import re
from utils.generic import load_config, print_config

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image, _ = transform(image_pil, None)
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
    """
    Optimized version to get phrases from posmap in batch.
    """
    pred_phrases = []
    for logit in logits_filt:
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase)
    return pred_phrases

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    # Prepare the caption
    caption = caption.lower().strip() + "."

    # Move model and image to the specified device
    model = model.to(device)
    image = image.to(device)

    # Get model outputs without calculating gradients
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    # Extract logits and boxes
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # Filter the outputs based on the threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    # Tokenize the caption
    tokenized = model.tokenizer(caption)

    # Get phrases from the logits and tokenized caption
    pred_phrases = get_phrases_from_posmap_batch(logits_filt, tokenized, model.tokenizer, text_threshold)

    return boxes_filt, pred_phrases

def save_mask_data(output_dir, mask_list, box_list, label_list, base_name):
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
            logit = logit[:-1]  # remove the closing ')'
        else:
            name, logit = label, 0.0  # Default logit value if not present
        json_data.append({'value': value, 'label': name, 'logit': float(logit), 'box': box.tolist()})
    with open(os.path.join(output_dir, 'json_masks/' + base_name + '_mask.json'), 'w') as f:
        json.dump(json_data, f)


def process_frame(frame, model, text_prompt, box_threshold, text_threshold, device, predictor, output_dir, frame_count, input_type, base_name):
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    _, image = load_image(frame, transform)

    dino = time.time()
    boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device)
    if boxes_filt.nelement() == 0:
        print("No bounding boxes detected.")
        return

    image = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) if input_type == "image" else frame
    start = time.time()
    predictor.set_image(image)
    H, W = image.shape[:2]
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    boxes_filt = boxes_filt * scale_tensor
    boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    boxes_filt[:, 2:] += boxes_filt[:, :2]
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    
    total_start_time_sam = time.time()
    masks, scores, logits_sam = predictor.predict_torch(None, None, transformed_boxes.to(device), multimask_output=False)
    
    temp_time = time.time()
    for idx, mask in enumerate(masks):
        colored_mask = np.zeros_like(image)
        colored_mask[mask.cpu().numpy()[0]] = (0, 255, 0)
        combined_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    save_mask_data(output_dir, masks, boxes_filt, pred_phrases, base_name)
    return cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

def get_object_identifier(filepath):
    # Split by '/' to isolate the filename
    filename = filepath.split('/')[-1]
    # Split by '_' to get the last part of the filename
    object_identifier = filename.split('_')[-1]
    # Remove the file extension
    object_identifier = object_identifier.split('.')[0]
    return object_identifier

def extract_image_number(filename):
  match = re.search(r'original_frame_(\d+)\.jpg', filename)
  if match:
      return int(match.group(1))
  return None  # In case there's a filename that doesn't match

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument('--config', type=str, default='config/grounded_sam_config.yaml', help='Path to configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    print_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_dir"] + "/np_masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/orig", exist_ok=True)
    os.makedirs(config["output_dir"] + "/json_masks", exist_ok=True)



    model = load_model(device=device, checkpoint=config["dino_checkpoint"])
    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"]).to(device)
    predictor = SamPredictor(sam)

    base_path = config["input_dir"]
    image_paths = glob(os.path.join(base_path, '*' + config['image_format']))
    image_paths.sort(key=extract_image_number)

    # Dataloader
    batch_size = config['batch_size']
    list_path = []
    sam_image_reader = SAMImageDetDataset(base_path, list_path, category_labels_path, sam)
    image_loader = SAMDataLoader(sam_image_reader, batch_size=batch_size, shuffle=False)


    for frame_count, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        print("Processing", image_path)
        total_time = time.time()
        text_prompt = config['text_prompt']
        #text_prompt = get_object_identifier(image_path)
        process_frame(image_path, model, text_prompt, config['box_threshold'], config['text_threshold'], device, predictor, config["output_dir"], frame_count, base_name)
        print(f"Execution time: {time.time() - total_time} seconds")
    
