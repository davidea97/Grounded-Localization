import torch
import numpy as np
from prettytable import PrettyTable

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.class_intersection = np.zeros(num_classes)
        self.class_union = np.zeros(num_classes)
        self.mIoU = 0.0
        self.partial_mIoU = 0.0
        self.total_samples = 0
        self.seen_class = np.zeros(num_classes).astype(bool)

    def update_metrics(self, pred_label, gt_label, ignore_index=255):

        label = gt_label
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        if not self.seen_class.any():
            current_classes = np.unique(label.cpu().numpy())
            self.seen_class[current_classes] = True

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1).cpu()
        area_pred_label = torch.histc(pred_label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1).cpu()
        area_label = torch.histc(label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect

        #class_iou = area_intersect / (area_union + 1e-10)

        for i in range(self.num_classes):
            self.class_intersection[i] += area_intersect[i]
            self.class_union[i] += area_union[i]

        self.total_samples += 1

        #self.compute_miou()

    def compute_miou(self):
        class_iou = self.class_intersection / (self.class_union + 1e-10)
        class_iou = class_iou[~np.isnan(class_iou)]
        self.mIoU = np.mean(class_iou)
        self.partial_mIoU = np.mean(class_iou[self.seen_class])

    def get_miou(self):
        self.compute_miou()
        return self.mIoU

    def get_partial_miou(self):
        self.compute_miou()
        return self.partial_mIoU

    def get_class_miou(self):
        class_iou = self.class_intersection / (self.class_union + 1e-10)
        return class_iou

    def reset(self):
        self.class_intersection = np.zeros(self.num_classes)
        self.class_union = np.zeros(self.num_classes)
        self.mIoU = 0.0
        self.total_samples = 0

    def print_table(self, class_names):
        self.compute_miou()
        header = ['Class', 'IoU']
        data = [[class_names[i], float(self.class_intersection[i] / (self.class_union[i] + 1e-10))] for i in range(self.num_classes)]
        data.append(['mIoU', float(self.mIoU)])
        table = PrettyTable()
        table.field_names = header
        for row in data:
            table.add_row(row)
        print(table)

        return dict(data)
