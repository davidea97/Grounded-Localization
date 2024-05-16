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

from segment_anything.utils.transforms import ResizeLongestSide

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device)
    return image.permute(2, 0, 1).contiguous()

class ImageDetDataset(Dataset):
    def __init__(self, base_path, list_file, category_labels_file):
        self.base_path = base_path
        self.list_file = list_file

        # Load category labels from YAML file
        with open(category_labels_file, 'r') as file:
            self.category_labels = yaml.safe_load(file)

        # Read image paths and annotation paths from the list file
        with open(list_file, 'r') as file:
            self.image_annotation_paths = [line.strip().split() for line in file]

    def __len__(self):
        return len(self.image_annotation_paths)

    def __getitem__(self, idx):
        image_file, annotation_file = self.image_annotation_paths[idx]

        # Parse annotation
        filename, width, height, boxes, labels = self.parse_annotation(annotation_file)

        # Load image
        image_path = os.path.join(self.base_path, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, boxes, labels, image_file

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


class SAMImageDetDataset(ImageDetDataset):
    def __init__(self, base_path, list_file, category_labels_file, sam):
        super().__init__(base_path, list_file, category_labels_file)
        self.resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        self.sam = sam

    def __getitem__(self, idx):
        image, boxes, labels, image_file = super().__getitem__(idx)

        # Prepare image
        prepared_image = prepare_image(image, self.resize_transform, self.sam)

        # Apply resize transform to bounding boxes
        boxes = torch.tensor(boxes, device=self.sam.device)
        resized_boxes = self.resize_transform.apply_boxes_torch(boxes, image.shape[:2])
        # Create batched input dictionary
        batched_input = {
            'original_image': image,  # For visualization
            'image': prepared_image,
            'boxes': resized_boxes,
            'original_boxes': boxes,
            'original_size': image.shape[:2],
            'ids': [lab[0] for lab in labels],
            'labels': [lab[1] for lab in labels],
            'relative_image_path': image_file
        }

        return batched_input


class SAMDataLoader(DataLoader):
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


# Example usage:
if __name__ == '__main__':
    base_path = "/media/data/Datasets/ILSVRC"
    category_labels_file = 'ILSVRC2017_category_labels.yaml'
    category_labels_path = os.path.join(base_path, "lists", category_labels_file)
    train_list_file = 'train.txt'
    train_list_path = os.path.join(base_path, "lists", train_list_file)

    dataset = ImageDetDataset(base_path, train_list_path, category_labels_path)

    # Visualize a sample
    sample_idx = 0
    dataset.visualize_sample(sample_idx)
