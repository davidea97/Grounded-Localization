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
import torchvision.transforms as transforms
import open3d as o3d
from sklearn.decomposition import PCA
import pandas as pd
import json

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image = transform(image_pil)
    return image_pil, image

def load_video_image(frame, transform):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    image_tensor = transform(image_pil)
    return image_pil, image_tensor

def load_model(device, checkpoint):
    model_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def load_camera_data_from_file(file_path):
    with open(file_path, 'r') as file:
        camera_data = json.load(file)
    return camera_data

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
    max_value = torch.max(logits.max(dim=1)[0])
    filt_mask = logits.max(dim=1)[0] >= box_threshold
    
    
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

def load_rototranslation(file_path):
    """
    Load the 4x4 rototranslation matrix from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    np.ndarray: 4x4 rototranslation matrix.
    """
    df = pd.read_csv(file_path, header=None)
    matrix_str = df.values
    matrix = np.zeros((4, 4))
    
    for i in range(4):
        row = matrix_str[i][0].split()
        matrix[i] = np.array([float(val) for val in row])
    
    return matrix

def create_pca_lines(obb, rotation_matrix_z):
    # Create lines to represent the PCA axes
    obb_center = obb.get_center()
    pca_axis_1 = obb_center + rotation_matrix_z @ np.array([0.1, 0, 0])
    pca_axis_2 = obb_center + rotation_matrix_z @ np.array([0, 0.1, 0])

    lines = [
        [obb_center, pca_axis_1],
        [obb_center, pca_axis_2]
    ]
    colors = [[1, 0, 0], [0, 1, 0]]  # Red for the first axis, Green for the second axis

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([obb_center, pca_axis_1, pca_axis_2]),
        lines=o3d.utility.Vector2iVector([[0, 1], [0, 2]])
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def bbox_generation(camera_data, combined_mask, image_rgb, depth_image, pose):

    # PointCloud extraction
    # Extract the colored point cloud
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    fx, fy, cx, cy = camera_data.get('fx'), camera_data.get('fy'), camera_data.get('cx'), camera_data.get('cy')  # Replace with actual intrinsics
    H, W = image_rgb.shape[:2]
    points = []
    colors = []
    
    if len(depth_image.shape)>2:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

    for v in range(H):
        for u in range(W):
            if combined_mask[v, u]:
                Z = depth_image[v, u] / 1000.0  # Assuming depth is in millimeters, convert to meters
                if Z > 0:  # Valid depth
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append([X, Y, Z])
                    colors.append(image_rgb[v, u] / 255.0)  # Normalize RGB values to [0, 1]

    points = np.array(points)
    colors = np.array(colors)
    
    # Create coordinate frames
    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Apply the robot transformation to the points
    ones = np.ones((points.shape[0], 1))
    points_hom = np.hstack([points, ones])  # Convert to homogeneous coordinates
    transformed_points_hom = (pose @ points_hom.T).T
    transformed_points = transformed_points_hom[:, :3] / transformed_points_hom[:, 3][:, np.newaxis]

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)

    # Compute the axis-aligned bounding box
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (0, 1, 0)  # Green color for AABB
    aabb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=aabb.get_center())

    # Convert the axis-aligned bounding box to an oriented bounding box
    obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb)
    obb.color = (1, 0, 0)  # Red color for the bounding box
    obb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=obb.get_center())

    # Project points onto the XY plane of the OBB frame
    obb_points = np.asarray(pcd.points) - obb.get_center()
    obb_points_xy = obb_points @ obb.R.T  # Transform points to OBB frame
    obb_points_xy[:, 2] = 0  # Project onto XY plane

    # Create a point cloud for the projected points
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(obb_points_xy)
    projected_pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(obb_points_xy))  # Set color to black

    # Perform PCA on the projected points
    pca = PCA(n_components=2)
    pca.fit(obb_points_xy[:, :2])
    pca_components = pca.components_

    # Ensure the rotation matrix is proper (i.e., no reflection)
    if np.linalg.det(pca_components) < 0:
        pca_components[1, :] = -pca_components[1, :]

    # Determine the principal direction (the one with the largest eigenvalue)
    principal_direction = pca_components[0]  # Assuming the first component is the largest

    # Compute the rotation around the Z-axis to align with the PCA components
    theta = np.arctan2(principal_direction[1], principal_direction[0])
    rotation_matrix_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Additional rotation of 90 degrees around Z-axis
    rotation_90_z = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # Apply the rotation to the OBB
    obb.R = rotation_matrix_z @ obb.R

    # Apply the same rotation to the robot frame
    obb_frame.rotate(rotation_matrix_z, center=obb.get_center())

    # Ensure the longest side aligns with the x-axis of the OBB frame
    obb_extents = obb.extent
    if obb_extents[0] < obb_extents[1]:
        rotation_90_z = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        obb.R = rotation_90_z @ obb.R

    previous_center = obb.get_center()

    # Recalculate the bounding box dimensions to fit the point cloud
    obb_points_transformed = np.asarray(pcd.points) - obb.get_center()
    obb_points_transformed = obb_points_transformed @ obb.R  # Align points with the OBB

    min_bound = np.min(obb_points_transformed, axis=0)
    max_bound = np.max(obb_points_transformed, axis=0)
    obb_extent = max_bound - min_bound

    # Update the OBB with new extents and center
    obb.center = (min_bound + max_bound) / 2
    obb.center = obb.center @ obb.R + obb.get_center()
    obb.extent = obb_extent
    #obb_to_robot_translation = previous_center

    obb.center = previous_center
    line_set = create_pca_lines(obb, rotation_matrix_z)
    
    # Visualize the point cloud with the updated OBB and projected points
    o3d.visualization.draw_geometries([pcd, obb, robot_frame, obb_frame, line_set], window_name='Colored Point Cloud')

    # 6D pose of OBB with respect to the robot frame
    obb_position = obb.get_center()
    obb_orientation = obb.R

    return obb, obb_position, obb_orientation

def reproject_obb_to_image(obb, obb_position, obb_orientation, camera_data, image_rgb, pose):
    
    # Get OBB corners
    obb_corners = np.asarray(obb.get_box_points())
    

    # Sort corners based on Z value
    obb_corners = sorted(obb_corners, key=lambda x: x[2])
    obb_corners = np.asarray(obb_corners)

    # Transform OBB corners to camera frame
    pose_inv = np.linalg.inv(pose)
    obb_corners_hom = np.hstack([obb_corners, np.ones((obb_corners.shape[0], 1))])
    obb_corners_cam_hom = (pose_inv @ obb_corners_hom.T).T
    obb_corners_cam = obb_corners_cam_hom[:, :3] / obb_corners_cam_hom[:, 3][:, np.newaxis]

    # Camera intrinsics
    fx, fy, cx, cy = camera_data['fx'], camera_data['fy'], camera_data['cx'], camera_data['cy']

    # Project OBB corners onto the image plane
    obb_corners_2d = np.zeros((obb_corners_cam.shape[0], 2))
    for i, point in enumerate(obb_corners_cam):
        X, Y, Z = point
        u = int((X * fx) / Z + cx)
        v = int((Y * fy) / Z + cy)
        obb_corners_2d[i] = [u, v]
    
    # Convert to integer tuples
    obb_corners_2d = obb_corners_2d.astype(int)

    # Draw the bounding box on the image
    image_with_bbox = image_rgb.copy()
    # Base rectangle (bottom four corners)
    for i in range(4):
        start_point = tuple(obb_corners_2d[i])
        end_point = tuple(obb_corners_2d[(i+1)%4])
        cv2.line(image_with_bbox, start_point, end_point, (0, 255, 0), 2)
        #cv2.circle(image_with_bbox, start_point, 1, (0,255,0), 3)
    
    # Top rectangle (top four corners)
    """for i in range(4, 8):
        start_point = tuple(obb_corners_2d[i])
        end_point = tuple(obb_corners_2d[(i+1)%4 + 4])
        cv2.line(image_with_bbox, start_point, end_point, (0, 255, 0), 2)"""
        
    # Vertical lines
    """for i in range(4):
        start_point = tuple(obb_corners_2d[i])
        end_point = tuple(obb_corners_2d[i+4])
        cv2.line(image_with_bbox, start_point, end_point, (0, 255, 0), 2)"""

    return image_with_bbox

def process_frame(frame, depth_frame, pose_path, camera_data, model, text_prompt, box_threshold, text_threshold, device, predictor, output_dir, frame_count, input_type, base_name, transform, f):
    if input_type == "image":
        _, image = load_image(frame, transform)
    else: 
        _, image = load_video_image(frame, transform)

    dino = time.time()
    boxes_filt, pred_phrases = get_grounding_output(model, image, text_prompt, box_threshold, text_threshold, device)
    if boxes_filt.nelement() == 0:
        print("No bounding boxes detected.")
        return

    print("Boxes filt: ", boxes_filt)
    print("Pred phrases: ", pred_phrases)
    pose = load_rototranslation(pose_path)

    #image = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) if input_type == "image" else frame
    image = cv2.imread(frame)
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image.copy()
    depth_image = cv2.imread(depth_frame, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        print("Failed to read depth image")
    else:
        print("Depth image read successfully")
        print("Depth image shape:", depth_image.shape)
        print("Depth image data type:", depth_image.dtype)

    start = time.time()
    predictor.set_image(image)
    H, W = image.shape[:2]
    scale_tensor = torch.tensor([W, H, W, H], device=device)
    boxes_filt = boxes_filt * scale_tensor
    boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
    boxes_filt[:, 2:] += boxes_filt[:, :2]
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    
    for box in boxes_filt:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 1)
        f.write(f"{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}\n")
    cv2.imwrite(os.path.join(output_dir, 'bbox2d/' + base_name + '_image.jpg'), image)

    total_start_time_sam = time.time()
    masks, scores, logits_sam = predictor.predict_torch(None, None, transformed_boxes.to(device), multimask_output=False)

    # Select the mask with the highest
    highest_score_index = torch.argmax(scores)
    masks = masks[highest_score_index]
    masks = torch.unsqueeze(masks, dim=0)
    temp_time = time.time()
    combined_mask = np.zeros((H, W), dtype=bool)
    for idx, mask in enumerate(masks):
        combined_mask |= mask.cpu().numpy()[0]
        np.save(os.path.join(output_dir, 'np_masks/mask_' + str(frame_count) + '.npy'), mask.cpu().numpy())
        colored_mask = np.zeros_like(image)
        colored_mask[mask.cpu().numpy()[0]] = (0, 255, 0)
        combined_image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    save_mask_data(output_dir, masks, boxes_filt, pred_phrases, base_name)

    obb, obb_position, obb_orientation = bbox_generation(camera_data, combined_mask, image_rgb, depth_image, pose)
    image_with_bbox = reproject_obb_to_image(obb, obb_position, obb_orientation, camera_data, image_rgb, pose)

    # Save the image
    cv2.imwrite(os.path.join(output_dir, 'bbox3d/bbox_' + base_name + '.jpg'), image_with_bbox)

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
    parser.add_argument('--input', type=str, default='image', help='Input file type: video "video" or set of images "image"')
    parser.add_argument('--video_path', type=str, default='panda_video.mp4', help='Video input file type')

    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["output_dir"] + "/masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/json_masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/np_masks", exist_ok=True)
    os.makedirs(config["output_dir"] + "/bbox3d", exist_ok=True)
    os.makedirs(config["output_dir"] + "/bbox2d", exist_ok=True)

    transform = transforms.Compose([
            transforms.Resize((800, 800)),  # Fixed size for all images
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    model = load_model(device=device, checkpoint=config["dino_checkpoint"])
    sam = sam_model_registry[config["sam_model_type"]](checkpoint=config["sam_checkpoint"]).to(device)
    predictor = SamPredictor(sam)

    if args.input == "image":
        image_paths = glob(os.path.join(config["input_dir"], 'rgb/*' + config['image_format']))
        depth_paths = glob(os.path.join(config["input_dir"], 'depth/*' + config['image_format']))
        pose_paths = glob(os.path.join(config["input_dir"], 'pose/*csv'))
        camera_intrinisc_path = os.path.join(config["input_dir"], 'camera.json')
        camera_data = load_camera_data_from_file(camera_intrinisc_path)
        #image_paths.sort(key=extract_image_number)
        image_paths.sort()
        depth_paths.sort()
        pose_paths.sort()
        with open(os.path.join(config["output_dir"], 'rectangles.txt'), 'w') as f:
            for frame_count, image_path in enumerate(image_paths):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                print("Processing", image_path)
                total_time = time.time()
                text_prompt = config['text_prompt']
                depth_path = depth_paths[frame_count]
                pose_path = pose_paths[frame_count]
                #text_prompt = get_object_identifier(image_path)
                process_frame(image_path, depth_path, pose_path, camera_data, model, text_prompt, config['box_threshold'], config['text_threshold'], device, predictor, config["output_dir"], frame_count, args.input, base_name, transform, f)
                #process_frame_wo_pose(image_path, depth_path, model, text_prompt, config['box_threshold'], config['text_threshold'], device, predictor, config["output_dir"], frame_count, args.input, base_name, transform, f)
                print(f"Execution time: {time.time() - total_time} seconds")
    else:
        fig, ax = plt.subplots()
        cap = cv2.VideoCapture(os.path.join(config["input_dir"], args.video_path))
        frame_count = 0
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            base_name_mask = base_name + str(frame_count)
            print("Processing frame", frame_count)

            total_time = time.time()
            f#rame_mask = process_frame(frame, model, config['text_prompt'], config['box_threshold'], config['text_threshold'], device, predictor, config["output_dir"], frame_count, args.input, base_name_mask, transform)
            print(f"Execution time: {time.time() - total_time} seconds")

            frame_count += 1

        cap.release()
