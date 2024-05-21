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
from tqdm import tqdm

from segment_anything.utils.transforms import ResizeLongestSide

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

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()


# Function to read a specific row from a text file
def read_specific_row(file_path, row_number):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if row_number < 1 or row_number > len(lines):
            raise IndexError("Row number out of range")
        return lines[row_number - 1].strip()  # Adjust for 1-based index and remove newline character

# Ensure that all other tensors used in the forward pass are also on the same device
def ensure_same_device(tensor, device):
    return tensor.to(device) if tensor.device != device else tensor

def check_tensors_on_gpu(batch_input):
    all_on_gpu = True
    for item in batch_input:
        print("Image: ", item['image'].is_cuda)
        print("Box: ", item['boxes'].is_cuda)
        print("Original boxes: ", item['original_boxes'].is_cuda)
        if not (item['image'].is_cuda and item['boxes'].is_cuda and item['original_boxes'].is_cuda):
            all_on_gpu = False
            break
    return all_on_gpu

def move_tensors_to_gpu(batch_input):
    for item in batch_input:
        if isinstance(item['image'], torch.Tensor):
            item['image'] = item['image'].to('cuda')
        if isinstance(item['boxes'], torch.Tensor):
            item['boxes'] = item['boxes'].to('cuda')
        if isinstance(item['original_boxes'], torch.Tensor):
            item['original_boxes'] = item['original_boxes'].to('cuda')
    return batch_input

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument('--config', type=str, default='config/grounded_sam_config.yaml', help='Path to configuration file')

    args = parser.parse_args()
    config = load_config(args.config)
    print_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_dir"] + "/masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/json_masks", exist_ok=True)

    dino_model = load_model(device=device, checkpoint=config["dino_checkpoint"])
    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"])
    sam = sam.to(device)
    #predictor = SamPredictor(sam)

    base_path = config["input_dir"]
    image_paths = glob(os.path.join(base_path, '*' + config['image_format']))
    image_paths.sort()
    output_dir = config["output_dir"]

    # Dataloader
    batch_size = config['batch_size']  
    prepare_list = config['prepare_list']  

    # Output file path
    output_file_path = "imagenet_paths.txt"
    if (prepare_list):

        # Writing the list to the text file
        with open(output_file_path, 'w') as file:
            for image_path in image_paths:
                file.write(f"{image_path}\n")

    box_thresh = config['box_threshold']
    text_thresh = config['text_threshold']
    sam_image_reader = SAMImageDetDataset(base_path, output_file_path, sam, box_thresh, text_thresh, dino_model, device)
    image_loader = SAMDataLoader(sam_image_reader, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    total_batch = len(sam_image_reader) // batch_size
    torch.set_grad_enabled(False)

    for idx, batched_input in enumerate(tqdm(image_loader, total=total_batch)):
        batched_output = sam(batched_input, multimask_output=False)
