import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import time

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# Segment Anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

# Miscellaneous
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import yaml


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_config(config):
    print("Configurations:")
    for key, value in config.items():
        print(f"{key}: {value}")

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_video_image(frame):

    # Convert the OpenCV frame (which is in BGR format) to a PIL Image in RGB format
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")

    # Define the transformations
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),  # Convert the image to a PyTorch tensor
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the tensor
    ])

    # Apply the transformation to the PIL image
    image_tensor, _ = transform(image_pil, None)

    return image_pil, image_tensor


def load_model(device, checkpoint):
    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    model_checkpoint_path = checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]


    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]


    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list, base_name):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'masks/' + base_name + '_mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'json_masks/' + base_name + '_mask.json'), 'w') as f:
        json.dump(json_data, f)

def get_base_name(image_path):
    # Get the base name with extension
    base_name_with_extension = os.path.basename(image_path)
    # Split the base name and extension and return just the base name
    base_name, _ = os.path.splitext(base_name_with_extension)
    return base_name



def process_frame(frame, model, text_prompt, box_threshold, text_threshold, device, predictor, output_dir, frame_count, input, base_name):

    if input=="image":
        _, image = load_image(frame)
    else: 
        _, image = load_video_image(frame)

    # Run grounding dino model
    start_time_dino = time.time()
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )
    end_time_dino = time.time()
    print(f"Dino execution time: {end_time_dino - start_time_dino} seconds")

    # Exit if no bounding boxes are extracted
    if boxes_filt.nelement() == 0:
        print("No bounding boxes detected.")
        return

    # Read images
    if input=="image":
        image = cv2.imread(frame)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = cv2.imread(frame)
    else: 
        image = frame
        cv2.imwrite(os.path.join(output_dir, 'orig/original_frame_' + str(frame_count) + '.jpg'), image)
        original_image = frame
    

    total_start_time_sam = time.time()
    predictor.set_image(image)

    # Image dimensions
    H, W = image.shape[:2]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    total_end_time_sam = time.time()
    
    # Save mask output
    for idx, mask in enumerate(masks):
        mask_np = mask.cpu().numpy()
        np.save(os.path.join(output_dir, 'np_masks/frame_' + str(frame_count) + '_mask' + str(idx+1) + '.npy'), mask_np)
        
        # Reshape mask to match the image dimensions and create a colored mask
        mask_np_squeezed = np.squeeze(mask_np)  # This will remove the dimension of size 1
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask_np_squeezed > 0] = (0, 255, 0)  # Green color mask
        # Combine the mask with the image
        combined_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
        # Display the image with the mask
    # Convert BGR to RGB
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)

    total_end_time_sam = time.time()
    print(f"SAM execution time: {total_end_time_sam - total_start_time_sam} seconds")

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases, base_name)

    return combined_image_rgb

import matplotlib.cm as cm
import matplotlib.animation as animation



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)

    parser.add_argument('--config', type=str, default='config/grounded_sam_config.yaml', help='Path to configuration file')
    parser.add_argument('--input', type=str, default='image', help='Input file type: video "video" or set of images "image"')
    parser.add_argument('--video_path', type=str, default='panda_video.mp4', help='Video input file type')

    args = parser.parse_args()

    # cfg
    config = load_config(args.config)

    sam_version = config["sam_model_type"]
    image_path = config["input_dir"]
    output_dir = config["output_dir"]
    box_threshold  = config["box_threshold"]
    text_threshold = config["text_threshold"]
    sam_checkpoint = config["sam_checkpoint"]
    dino_checkpoint = config["dino_checkpoint"]
    batch_size = config['batch_size']
    text_prompt = config['text_prompt']
    input = args.input
    input_video = args.video_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+"/np_masks", exist_ok=True)
    os.makedirs(output_dir+"/masks", exist_ok=True)
    os.makedirs(output_dir+"/orig", exist_ok=True)
    os.makedirs(output_dir+"/json_masks", exist_ok=True)

    # load model
    model = load_model(device=device, checkpoint=dino_checkpoint)

    # SAM MODEL
    sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    if input=="image":
        # Load image
        image_paths = glob(os.path.join(image_path, 'demo*.jpg'))  # Adjust pattern if needed
        for frame_count, image_path in enumerate(image_paths):
            base_name = get_base_name(image_path)
            print("########## Processing " + image_path + " ##########")
            process_frame(image_path, model, text_prompt, box_threshold, text_threshold, device, predictor, output_dir, frame_count, input, base_name)
    else:
        
        fig, ax = plt.subplots()

        # Load video
        frame_count = 0
        cap = cv2.VideoCapture(os.path.join(image_path, input_video))
        base_name = get_base_name(input_video)
        frames = [] # for storing the generated images

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            base_name_mask = base_name + str(frame_count)
            print("########## Processing frame " + str(frame_count) + " ##########")
            frame_mask = process_frame(frame, model, text_prompt, box_threshold, text_threshold, device, predictor, output_dir, frame_count, input, base_name_mask)
            frame_count += 1
            #frames.append([plt.imshow(frame_mask, cmap=cm.Greys_r,animated=True)])
            #plt.imshow(frame_mask)
            #plt.show()
        #cap.release()
        
        #ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
        #                        repeat_delay=1000)
        #9ani.save('movie.mp4', writer='ffmpeg')



        