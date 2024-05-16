from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import cv2

import torch

if not torch.cuda.is_available():
    print("CUDA is not available. Exiting...")
    exit()
else:
    print("CUDA is available.")


CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
DEVICE = "cuda"
IMAGE_PATH = "assets/demo7.jpg"
TEXT_PROMPT = "Dark Brown Horse. White horse. Light Brown Horse. Tail. Brown Horse head"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
FP16_INFERENCE = True

image_source, image = load_image(IMAGE_PATH)
model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

#if FP16_INFERENCE:
#    image = image.half()
#    model = model.half()

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=DEVICE,
)



annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("GroundingDINO/annotated_image_davide.jpg", annotated_frame)
