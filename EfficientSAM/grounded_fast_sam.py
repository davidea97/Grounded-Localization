import argparse
import cv2
from ultralytics import YOLO
from FastSAM.tools import *
from groundingdino.util.inference import load_model, load_image, predict, predict_batch, annotate, Model
from torchvision.ops import box_convert
import ast
import time
import torchvision.transforms as transforms
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./FastSAM/FastSAM-s.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="../assets/demo1.jpg", help="path to image file"
    )
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument(
        "--text", type=str, default="bear", help="text prompt for GroundingDINO"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[0,0,0,0]", help="[x,y,w,h]")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def get_object_identifier(filepath):
    object_identifier = filepath.split('_')[-1]
    # Remove the file extension
    object_identifier = object_identifier.split('.')[0]
    return object_identifier

def load_image(image_path, transform):
    image_pil = Image.open(image_path).convert("RGB")
    image_pil_size = image_pil.size
    print(f"Image size: {image_pil_size}")
    
    image = transform(image_pil)
    return image_pil, image, image_pil_size

def save_mask_and_json(mask_list, output_dir, base_name):
    
    # Vectorized mask processing
    mask_img = torch.zeros(mask_list.shape[-2:], dtype=torch.uint8)
    mask_np = mask_list # Bring mask_list to CPU only once
    indices = np.arange(1, len(mask_list) + 1)
    
    for idx, mask in enumerate(mask_np):
        mask_img[mask[0]] = indices[idx]

    mask_img_np = mask_img.numpy()  # Convert to NumPy only once
    plt.imsave(os.path.join(output_dir, f'{base_name}_mask.png'), mask_img_np, cmap='gray')
   
def draw_box_on_image(image, box, color=(0, 255, 0), thickness=2):
    start_point = (int(box[0]), int(box[1]))
    end_point = (int(box[2]), int(box[3]))
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

def process_images_batch(batch, img_folder, save_path, args, model, groundingdino_model, transform):
    for img_name in batch:
        start_time = time.time()
        if img_name == ".git":
            continue

        img_path = os.path.join(img_folder, img_name)
        basename = os.path.basename(img_path).split(".")[0]
        text = get_object_identifier(basename)
        
        fast_sam_time = time.time()

        # Load image and perform Fast-SAM inference
        results = model(
            img_path,
            imgsz=args.imgsz,
            device=args.device,
            retina_masks=args.retina,
            iou=args.iou,
            conf=args.conf,
            max_det=100,
        )
        print(f"Fast-SAM inference time: {time.time() - fast_sam_time}")

        # Load image for GroundingDINO
        image_source, image, image_pil_size = load_image(img_path, transform)

        # Perform GroundingDINO prediction
        dino_time = time.time()
        boxes, logits, phrases = predict(
            model=groundingdino_model,
            image=image,
            caption=text,
            box_threshold=0.3,
            text_threshold=0.25,
            device=args.device,
        )

        print(f"GroundingDINO inference time: {time.time() - dino_time}")

        # Grounded-Fast-SAM processing
        ori_img = cv2.imread(img_path)
        ori_h, ori_w = ori_img.shape[:2]

        if len(boxes) == 0 or results[0].masks is None:
            continue

        # Select the box with the highest score
        max_idx = np.argmax(logits)
        max_box = boxes[max_idx].unsqueeze(0)
        max_phrase = phrases[max_idx]

        # Convert box format and scale to original image dimensions
        max_box = max_box * torch.Tensor([ori_w, ori_h, ori_w, ori_h])
        max_box = box_convert(max_box, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy().tolist()[0]

        mask, _ = box_prompt(
            results[0].masks.data,
            max_box,
            ori_h,
            ori_w,
        )

        annotations = np.array([mask])

        # Save black and white mask image
        bw_image = (annotations[0] * 255).astype(np.uint8)
        bw_output_filename = os.path.join(save_path, f"{basename}_mask.jpg")
        cv2.imwrite(bw_output_filename, bw_image)
        print(f"Time taken for {img_name}: {time.time() - start_time} seconds")



def main(args):
    img_folder = args.img_folder
    save_path = args.output
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = YOLO(args.model_path)
    groundingdino_config = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    groundingdino_ckpt_path = "../dino_models/groundingdino_swint_ogc.pth"
    groundingdino_model = load_model(groundingdino_config, groundingdino_ckpt_path)

    transform = transforms.Compose([
        transforms.Resize((250,250)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #image_paths = os.listdir(img_folder)
    image_paths = [img for img in os.listdir(img_folder) if img != ".git"]
    image_paths.sort()
    
    batch_size = 16  # Adjust based on your GPU memory capacity
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    for batch in batches:
        start_batch_time = time.time()
        process_images_batch(batch, img_folder, save_path, args, model, groundingdino_model, transform)
        print(f"Time taken for batch: {time.time() - start_batch_time} seconds")

if __name__ == "__main__":
    args = parse_args()
    main(args)