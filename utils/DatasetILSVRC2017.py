import os
import yaml
import xml.etree.ElementTree as ET
#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
import cv2
import torch
from torch.utils.data import DataLoader
from PIL import Image
import random
import time
import torchvision.transforms as transforms
from segment_anything.utils.transforms import ResizeLongestSide
import GroundingDINO.groundingdino.datasets.transforms as T
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

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

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

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


class ImageDetDataset(Dataset):
    def __init__(self, base_path, list_file):
        self.base_path = base_path
        self.list_file = list_file

        # Read image paths and annotation paths from the list file
        with open(list_file, 'r') as file:
            self.image_annotation_paths = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.image_annotation_paths)

    def __getitem__(self, idx):
        image_file = self.image_annotation_paths[idx][0]

        image_path = image_file
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, image_file

    def parse_annotation(self, xml_file):
        try:
            tree = ET.parse(xml_file)
        except:
            print(f"Error parsing {xml_file}")
            return None, None, None, None, None

        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            label_code = obj.find('name').text
            label_dict = self.category_labels.get(label_code, None)
            label = label_dict['label'] if label_dict else 'Unknown'
            id = label_dict['id'] if label_dict else -1
            boxes.append((xmin, ymin, xmax, ymax))
            labels.append((id, label))

        return filename, width, height, boxes, labels

    def visualize_sample(self, idx):
        image, boxes, labels = self.__getitem__(idx)
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        ax = plt.gca()
        for box, lab in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            id, label = lab
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f"{id}:{label}", fontsize=8, color='r')
        plt.axis('off')
        plt.show()

"""def get_object_identifier(filepath):
    # Split by '/' to isolate the filename
    filename = filepath.split('/')[-1]
    # Split by '_' to get the last part of the filename
    object_identifier = filename.split('_')[-1]
    # Remove the file extension
    object_identifier = object_identifier.split('.')[0]
    return object_identifier"""

def get_object_identifier(filepath):
    # Split by '/' to isolate the filename
    filename = filepath.split('/')[-1]
    # Split by '_' to separate the code and the object identifier
    parts = filename.split('_')
    # Remove the file extension from the last part
    parts[-1] = parts[-1].split('.')[0]
    # Join all parts after the first part (which is the code)
    object_identifier_parts = parts[1:]
    if len(object_identifier_parts) == 2:
        # Join with space if there are exactly two words
        object_identifier = ' '.join(object_identifier_parts)
    else:
        # Otherwise, join with underscore
        object_identifier = '_'.join(object_identifier_parts)
    return object_identifier

def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image_pil_size = image_pil.size
    image = transform(image_pil)
    return image_pil, image, image_pil_size


class SAMImageDetDataset(ImageDetDataset):
    def __init__(self, base_path, list_file, sam, box_thresh, text_thresh, dino_model, device):
        super().__init__(base_path, list_file)
        self.resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        self.sam = sam
        self.dino_model = dino_model
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((400, 400)),  # Fixed size for all images
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        image, image_file = super().__getitem__(idx)

        text_prompt = get_object_identifier(image_file)

        # Prepare image
        time_now = time.time()
        prepared_image = prepare_image(image, self.resize_transform, self.sam)

        _, image_dino, image_pil_size = load_image(image_file, self.transform)
        dino_time = time.time()
        boxes, _ = get_grounding_output(self.dino_model, image_dino, text_prompt, self.box_thresh, self.text_thresh, self.device)
        if boxes.nelement() == 0:
            #print("No bounding box detected!")
            width, height = image_pil_size
            boxes = torch.tensor([[0, 0, width, height]], dtype=torch.float32).to(self.device)
            



        resized_boxes = self.resize_transform.apply_boxes_torch(boxes, image.shape[:2])
        # Create batched input dictionary
        batched_input = {
            'original_image': image,  # For visualization
            'image': prepared_image,
            'boxes': resized_boxes,
            'original_boxes': boxes,
            'original_size': image.shape[:2],
            #'ids': [lab[0] for lab in labels],
            #'labels': [lab[1] for lab in labels],
            'relative_image_path': image_file
        }

        return batched_input


"""class SAMDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))


        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            mini_batch = [self.dataset[idx] for idx in batch_indices]
            # mini_batch = []
            # for idx in batch_indices:
            #     sample = self.dataset[idx]
            #     if sample is not None:
            #         mini_batch.append(sample)
            yield mini_batch"""


class SAMDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_samples = len(dataset)
        self.indices = list(range(self.num_samples))

        if self.shuffle:
            random.shuffle(self.indices)

        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def __iter__(self):
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            mini_batch = [self.dataset[idx] for idx in batch_indices]
            yield mini_batch


class ILSVRCPseudoMaskDataset(Dataset):
    def __init__(self, list_file, transform=None):
        self.list_file = list_file
        self.transform = transform
        self.data = self.read_list_file()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, mask_path = self.data[idx]
        image = cv2.imread(image_path)  # Read image using OpenCV
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask in grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def read_list_file(self):
        data = []
        with open(self.list_file, 'r') as f:
            for line in f:
                image_path, mask_path = line.strip().split()
                data.append((image_path, mask_path))
        return data
