import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

import torchvision
from PIL import Image
import colorsys


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    ax.add_patch(patches.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))
    ax.text(x0, y0, label, bbox={'facecolor': 'white', 'alpha': 0.5})

