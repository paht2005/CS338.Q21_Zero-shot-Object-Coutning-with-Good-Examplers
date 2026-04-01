"""
Official Inference Module for VA-Count
Supports both YOLO and Grounding DINO detection
With Gemini prompt enhancement for Grounding DINO
"""

# ===== WINDOWS COMPATIBILITY FIX - MUST BE FIRST =====
import pathlib
import sys
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath
# ======================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import time
import inflect
import google.generativeai as genai

# CLIP and detection models
import clip
from ultralytics import YOLOWorld

# Grounding DINO imports
import sys

sys.path.append("./GroundingDINO")
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)

# Project modules
import models_mae_cross

# ===== CONSTANTS =====
MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]
CONF_THRESHOLD = 0.05
GEMINI_API_KEY = "AIzaSyD4JRCmtzblaw33zzHvKq01Xbg_kshlM5c"

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
p = inflect.engine()


# ===== BINARY CLASSIFIER =====
class ClipClassifier(nn.Module):
    """Binary classifier to filter exemplar patches"""

    def __init__(self, clip_model, embed_dim=512):
        super(ClipClassifier, self).__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float()
        x = self.fc(image_features)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits


# ===== MODEL LOADING =====
def load_counting_model(
    checkpoint_path, device="cuda", model_name="mae_vit_base_patch16"
):
    """Load VA-Count model from checkpoint"""
    model = models_mae_cross.__dict__[model_name](norm_pix_loss=False)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def load_yolo_model(model_path="./yolov8x-worldv2.pt", device="cuda"):
    """Load YOLOWorld for object detection"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLOWorld model not found: {model_path}")

    model = YOLOWorld(model_path)
    if device == "cuda" and torch.cuda.is_available():
        model.to(device)
    return model


def load_grounding_dino_model(
    config_path="./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    checkpoint_path="./GroundingDINO/weights/groundingdino_swint_ogc.pth",
    device="cuda",
):
    """Load Grounding DINO model"""
    args = SLConfig.fromfile(config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    return model


def load_clip_model(device="cuda"):
    """Load CLIP for exemplar classification"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess


def load_binary_classifier(
    clip_model, weights_path="./data/out/classify/best_model.pth", device="cuda"
):
    """Load binary classifier for filtering exemplars"""
    classifier = ClipClassifier(clip_model).to(device)
    if Path(weights_path).exists():
        classifier.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    classifier.eval()
    return classifier


# ===== GEMINI PROMPT ENHANCEMENT =====
def enhance_prompt_with_gemini(image, class_name, use_contrastive=True):
    """
    Enhance text prompt using Gemini 2.5 Flash
    Logic from enhance_prompt_v3.ipynb

    Strategy: Contrastive Prompting for Grounding DINO
    - Generates both "group" and "single item" descriptions
    - Helps model differentiate between collective and individual objects
    - Essential for detecting small, dense objects (e.g., grapes in a bunch)

    Args:
        image: PIL Image
        class_name: Original class name (e.g., "cow", "grape")
        use_contrastive: If True, prepends group term for better detection

    Returns:
        enhanced_prompt: Detailed description with contrastive format
        time_taken: Time in seconds
    """
    start_time = time.time()

    try:
        # Convert to singular
        singular_name = p.singular_noun(class_name)
        if not singular_name:
            singular_name = class_name

        # Detect plural form for contrastive prompting
        plural_name = class_name if class_name.endswith("s") else class_name + "s"

        # Prompt template from notebook
        prompt = f"""
Look at the image and provide the **visual definition** of a single '{singular_name}'.

Task: Describe the intrinsic physical appearance of just **ONE** instance, as if it were cropped out and isolated.

**STRICT RULES:**
1. **Subject**: Describe ONE item only. Ignore the background or other identical items in the group.
2. **Format**: Start with 'single {singular_name}'. Use dot-separated phrases.
3. **Content**: Focus on Shape, Color, Material.

**Example for 'keyboard key':**
BAD (Describing the group): keyboard key . rows of buttons . full keyboard layout . many keys .
GOOD (Describing the instance): single keyboard key . square shape . black plastic material . white printed letter . smooth surface .

**Your output for '{singular_name}':**
"""

        response = gemini_model.generate_content([prompt, image])
        text = response.text.strip().replace("\n", " ").replace("..", ".")

        if not text.endswith("."):
            text += " ."

        # Apply contrastive prompting strategy for Grounding DINO
        if use_contrastive:
            # Add group/collection term to help model differentiate
            group_terms = {
                "grape": "grape bunch",
                "berry": "berry cluster",
                "apple": "apple pile",
                "person": "crowd",
                "car": "parking lot",
                "bird": "flock",
                "flower": "bouquet",
                "cookie": "cookie jar",
                "candy": "candy pile",
            }

            group_term = group_terms.get(singular_name.lower(), f"{plural_name} group")

            # Format: "group_term . single_item_description"
            # This helps Grounding DINO distinguish between collective and individual
            text = f"{group_term} . {text}"

        time_taken = time.time() - start_time
        return text, time_taken

    except Exception as e:
        print(f"⚠️ Gemini API error: {e}")
        time_taken = time.time() - start_time
        fallback = f"single {class_name} ."
        if use_contrastive:
            plural_name = class_name if class_name.endswith("s") else class_name + "s"
            fallback = f"{plural_name} group . {fallback}"
        return fallback, time_taken


# ===== IMAGE PREPROCESSING =====
def preprocess_image_for_model(image):
    """
    Preprocess image to 384x384 for VA-Count model

    Args:
        image: PIL Image (any size)

    Returns:
        resized_image: PIL Image (384x384)
        scale_factors: (scale_w, scale_h) tuple
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    W, H = image.size
    resized_image = image.resize((MAX_HW, MAX_HW), Image.BILINEAR)

    scale_factor_w = MAX_HW / W
    scale_factor_h = MAX_HW / H

    return resized_image, (scale_factor_w, scale_factor_h)


# ===== YOLO DETECTION =====
def detect_objects_yolo(yolo_model, image, text_prompt, conf_threshold=CONF_THRESHOLD):
    """
    Detect objects using YOLOWorld with text prompt

    Args:
        yolo_model: YOLOWorld model
        image: PIL Image (384x384)
        text_prompt: Text description (e.g., "cow")
        conf_threshold: Confidence threshold

    Returns:
        boxes: List of [x1, y1, x2, y2] in image coordinates
        time_taken: Detection time in seconds
    """
    start_time = time.time()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Set detection classes - Use simple prompt like Roboflow
    # Roboflow uses single class name (e.g., "tomato"), not complex prompts
    yolo_model.set_classes([text_prompt])

    # Convert to numpy
    img_np = np.array(image)

    # Detect - with optimized parameters for better detection
    # Use lower conf threshold and enable agnostic NMS for better recall
    results = yolo_model.predict(
        img_np, 
        conf=conf_threshold,
        iou=0.45,  # NMS IoU threshold (default 0.45)
        max_det=300,  # Max detections per image (default 300)
        verbose=False,
        imgsz=640,  # Standard YOLO input size
        agnostic_nms=True  # Class-agnostic NMS
    )

    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_score = box.conf[0].cpu().numpy()
            # Keep boxes that meet threshold
            if conf_score >= conf_threshold:
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

    time_taken = time.time() - start_time
    return boxes, time_taken


# ===== GROUNDING DINO DETECTION =====
def detect_objects_grounding_dino(
    model,
    image,
    text_prompt,
    box_threshold=0.20,
    text_threshold=0.25,
    device="cuda",
    filter_small_objects=True,
):
    """
    Detect objects using Grounding DINO with optimizations for small, dense objects

    Strategies Applied:
    1. Contrastive Prompting: Uses "group . single item" format to help model differentiate
    2. Low Threshold: box_threshold=0.20 (down from 0.30) to catch partially occluded objects
    3. Smart Filtering: Filters out large "group" bboxes, keeps only "single" item detections

    This is optimized for dense object scenarios like grape bunches, where we want
    to detect individual grapes instead of the whole bunch as one bbox.

    Args:
        model: Grounding DINO model
        image: PIL Image (384x384)
        text_prompt: Enhanced contrastive prompt (e.g., "grape bunch . single grape . round . ...")
        box_threshold: Box confidence threshold (0.20 for better recall on small objects)
        text_threshold: Text confidence threshold
        device: Device
        filter_small_objects: If True, filters out large "group" bboxes

    Returns:
        boxes: List of [x1, y1, x2, y2] in pixel coordinates (filtered for single items)
        time_taken: Detection time in seconds
    """
    start_time = time.time()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare caption
    caption = text_prompt.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."

    # Transform image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_transformed, _ = transform(image, None)
    image_transformed = image_transformed.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_transformed[None], captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4) - normalized [cx, cy, w, h]

    # Filter by threshold (lower threshold for better recall)
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    # Get phrase labels for each box (to filter "single" vs "group")
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    # Extract phrases for filtering
    from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap

    phrases = []
    for logit in logits_filt:
        phrase = (
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            .replace(".", "")
            .strip()
        )
        phrases.append(phrase)

    # Filter boxes based on contrastive prompting
    # Keep only boxes that match "single" keyword if present in prompt
    filtered_boxes = []
    W, H = image.size

    for i, (box, phrase) in enumerate(zip(boxes_filt, phrases)):
        # If using contrastive prompting, filter out "group/bunch/cluster" detections
        if filter_small_objects and "single" in caption:
            # Check if phrase contains "single" or individual descriptors
            # Skip if phrase contains group words
            group_words = [
                "bunch",
                "group",
                "cluster",
                "pile",
                "crowd",
                "flock",
                "bouquet",
                "parking",
                "jar",
            ]
            if any(word in phrase.lower() for word in group_words):
                continue  # Skip large groupings

            # Prefer phrases with "single" or small size descriptors
            if "single" not in phrase.lower():
                # Still accept if bbox is small enough (not a large grouping)
                cx, cy, w, h = box
                # Normalized width/height > 0.3 means it's likely a large group
                # For dense objects like grapes, we want small individual bboxes
                if w > 0.3 or h > 0.3:
                    continue

        # Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
        cx, cy, w, h = box
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        filtered_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    time_taken = time.time() - start_time
    return filtered_boxes, time_taken


# ===== EXEMPLAR CLASSIFICATION =====
def classify_exemplars(
    clip_model,
    clip_preprocess,
    binary_classifier,
    image,
    boxes,
    text_prompt,
    device="cuda",
    top_k=3,
):
    """
    Classify detected boxes into positive/negative exemplars
    Uses CLIP similarity + binary classifier

    Args:
        clip_model: CLIP model
        clip_preprocess: CLIP preprocessing
        binary_classifier: Binary classifier model
        image: PIL Image (384x384)
        boxes: List of [x1, y1, x2, y2]
        text_prompt: Target object name
        device: Device
        top_k: Number of positive exemplars

    Returns:
        positive_boxes: Top K positive exemplar boxes
        negative_boxes: Remaining boxes (for background)
        positive_crops: PIL Images of positive exemplars
        negative_crops: PIL Images of negative exemplars (sampled from background)
    """
    if len(boxes) == 0:
        return [], [], [], []

    # Text prompt for CLIP
    text_pos = f"a photo of a single {text_prompt}"
    text_inputs = clip.tokenize([text_pos]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Score all boxes
    scored_boxes = []
    W, H = image.size

    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        crop = image.crop((x1, y1, x2, y2))

        # Skip tiny/invalid crops
        if crop.width < 5 or crop.height < 5:
            continue

        # Skip too large crops (likely full image)
        if crop.width > W / 2 or crop.height > H / 2:
            continue

        # Binary classifier filtering
        crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = binary_classifier(crop_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prob_label_1 = probabilities[0, 1].item()

        # Skip if binary classifier rejects (threshold from grounding_pos_yolo.py)
        if prob_label_1 < 0.8:
            continue

        # CLIP similarity
        with torch.no_grad():
            image_features = clip_model.encode_image(crop_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0).item()
        scored_boxes.append((box, similarity, crop))

    if len(scored_boxes) == 0:
        return [], [], [], []

    # Sort by CLIP similarity
    scored_boxes.sort(key=lambda x: x[1], reverse=True)

    # Select top K as positive
    positive_data = scored_boxes[: min(top_k, len(scored_boxes))]
    positive_boxes = [box for box, _, _ in positive_data]
    positive_crops = [crop for _, _, crop in positive_data]

    negative_boxes = [box for box, _, _ in scored_boxes[top_k:]]

    # Sample negative patches from background
    negative_crops = sample_negative_patches(
        image, positive_boxes + negative_boxes, num_samples=5
    )

    return positive_boxes, negative_boxes, positive_crops, negative_crops


def sample_negative_patches(image, exclude_boxes, num_samples=5, patch_size=64):
    """
    Sample random patches from background (areas without detected objects)

    Args:
        image: PIL Image
        exclude_boxes: List of boxes to avoid
        num_samples: Number of negative patches
        patch_size: Size of patches

    Returns:
        patches: List of PIL Images
    """
    W, H = image.size
    patches = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(patches) < num_samples and attempts < max_attempts:
        attempts += 1

        # Random position
        x = np.random.randint(0, max(1, W - patch_size))
        y = np.random.randint(0, max(1, H - patch_size))

        # Check if overlaps with any box
        patch_box = [x, y, x + patch_size, y + patch_size]
        overlap = False

        for box in exclude_boxes:
            if boxes_iou(patch_box, box) > 0.1:  # Small overlap threshold
                overlap = True
                break

        if not overlap:
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def boxes_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# ===== COUNTING MODEL INFERENCE =====
def extract_box_crops_tensor(image, boxes, crop_size=128):
    """
    Extract crops from boxes and convert to tensor for model input

    Args:
        image: PIL Image (384x384)
        boxes: List of [x1, y1, x2, y2]
        crop_size: Target crop size (128 for VA-Count)

    Returns:
        crops_tensor: Tensor [num_boxes, 3, 128, 128]
    """
    if len(boxes) == 0:
        return torch.zeros(1, 3, crop_size, crop_size)

    transform = transforms.Compose(
        [
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD),
        ]
    )

    crops = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        if x2 > x1 and y2 > y1:
            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = transform(crop)
            crops.append(crop_tensor)

    if len(crops) == 0:
        return torch.zeros(1, 3, crop_size, crop_size)

    crops_tensor = torch.stack(crops)
    return crops_tensor


def predict_count(model, image, positive_boxes, device="cuda"):
    """
    Run VA-Count model to predict density map and count

    Args:
        model: VA-Count model
        image: PIL Image (384x384)
        positive_boxes: List of positive exemplar boxes
        device: Device

    Returns:
        count: Predicted count
        density_map: Numpy array of density map
        time_taken: Inference time
    """
    start_time = time.time()

    if len(positive_boxes) == 0:
        return 0, None, 0.0

    # Extract exemplar crops
    pos_crops = extract_box_crops_tensor(image, positive_boxes, crop_size=128)
    pos_crops = pos_crops.unsqueeze(0).to(device)  # [1, num_boxes, 3, 128, 128]

    # Prepare image tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Run model
    shot_num = len(positive_boxes)
    with torch.no_grad():
        density_map = model(image_tensor, pos_crops, shot_num)

    # Calculate count (divided by 60 as per FSC_test.py)
    count = torch.abs(density_map.sum()).item() / 60.0
    count = max(0, count)

    # Convert density map to numpy
    density_map_np = density_map.squeeze().cpu().numpy()

    time_taken = time.time() - start_time
    return count, density_map_np, time_taken


# ===== VISUALIZATION =====
def visualize_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw boxes on image

    Args:
        image: PIL Image
        boxes: List of [x1, y1, x2, y2]
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        result_image: PIL Image with boxes
    """
    img_np = np.array(image.copy())

    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)

    result_image = Image.fromarray(img_np)
    return result_image


def visualize_density_map(density_map):
    """
    Create heatmap from density map

    Args:
        density_map: Numpy array

    Returns:
        heatmap_image: PIL Image of heatmap
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    # Normalize
    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(density_map, cmap="jet", interpolation="bilinear")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # Convert to image
    fig.canvas.draw()
    heatmap_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    heatmap_np = heatmap_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    heatmap_image = Image.fromarray(heatmap_np)

    plt.close(fig)

    return heatmap_image


def create_overlay(image, density_map, alpha=0.5):
    """
    Overlay density map on original image

    Args:
        image: PIL Image (384x384)
        density_map: Numpy array
        alpha: Transparency

    Returns:
        overlay_image: PIL Image
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")

    # Normalize density map
    if density_map.max() > 0:
        density_map_norm = density_map / density_map.max()
    else:
        density_map_norm = density_map

    # Apply colormap
    cmap = plt.cm.jet
    heatmap = cmap(density_map_norm)[:, :, :3]  # RGB only

    # Convert image to numpy
    img_np = np.array(image.resize((MAX_HW, MAX_HW))) / 255.0

    # Blend
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = (overlay * 255).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)
    return overlay_image
