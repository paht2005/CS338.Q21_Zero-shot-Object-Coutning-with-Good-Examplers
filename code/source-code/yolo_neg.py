import torch
import os
import clip
import argparse
from PIL import Image
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLOWorld

# Global variables
device = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.02

# ClipClassifier class
class ClipClassifier(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super(ClipClassifier, self).__init__()
        self.clip_model = clip_model.to(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(clip_model.visual.output_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float().to(device)
        x = self.fc(image_features)
        x = F.relu(x)
        logits = self.classifier(x)
        return logits

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device)
clip_model.eval()

# Load binary classifier
binary_classifier = ClipClassifier(clip_model).to(device)
model_weights_path = './data/out/classify/best_model.pth'
binary_classifier.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=False))
binary_classifier.eval()

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in xyxy format"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def is_valid_patch(patch, binary_classifier, preprocess, device):
    if patch.size[0] <= 0 or patch.size[1] <= 0:
        return False
    patch_tensor = preprocess(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = binary_classifier(patch_tensor)
        probabilities = torch.softmax(logits, dim=1)
        prob_label_1 = probabilities[0, 1]
    return prob_label_1.item() > 0.8  # Lower threshold for negatives

def process_images(text_file_path, dataset_path, yolo_model, preprocess, output_folder, device='cpu'):
    boxes_dict = {}
    
    with open(text_file_path, 'r') as f:
        for line in f:
            image_name, class_name = line.strip().split('\t')
            print(f"Processing image: {image_name}")
            
            image_path = os.path.join(dataset_path, image_name)
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            
            # Detect target class objects
            class_prompts = [
                class_name,
                f"a {class_name}",
                f"a photo of a {class_name}",
                class_name + 's',
                f"multiple {class_name}"
            ]
            yolo_model.set_classes(class_prompts)
            results_class = yolo_model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)
            
            # Detect all objects
            object_prompts = ["object", "thing", "item","anything","entity","body","substance","stuff"]
            yolo_model.set_classes(object_prompts)
            results_object = yolo_model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)
            
            if len(results_object[0].boxes) == 0:
                print(f"No boxes detected for {image_name}")
                boxes_dict[image_name] = [[0, 0, 20, 20]] * 3
                continue
            
            # Get boxes and confidences
            boxes_class = results_class[0].boxes.xyxy.cpu().numpy() if len(results_class[0].boxes) > 0 else np.array([])
            boxes_object = results_object[0].boxes.xyxy.cpu().numpy()
            confs_object = results_object[0].boxes.conf.cpu().numpy()
            
            # Filter class boxes
            valid_class_boxes = []
            for box in boxes_class:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                
                if x2 - x1 > w / 2 or y2 - y1 > h / 2:
                    continue
                    
                patch = img.crop((x1, y1, x2, y2))
                if is_valid_patch(patch, binary_classifier, preprocess, device):
                    valid_class_boxes.append(box)
            
            # Find negative samples (low IoU with class boxes)
            negative_patches = []
            for i, (box_obj, conf) in enumerate(zip(boxes_object, confs_object)):
                # Calculate max IoU with all class boxes
                max_iou = 0
                if len(valid_class_boxes) > 0:
                    for box_class in valid_class_boxes:
                        iou = calculate_iou(box_obj, box_class)
                        max_iou = max(max_iou, iou)
                
                # If IoU < 0.5, it's a negative sample
                if max_iou < 0.5:
                    x1, y1, x2, y2 = box_obj.astype(int)
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                    
                    # Skip invalid patches
                    if (x2 - x1 > w / 2 or y2 - y1 > h / 2 or 
                        y2 - y1 < 5 or x2 - x1 < 5):
                        continue
                    
                    patch = img.crop((x1, y1, x2, y2))
                    if patch.size == (0, 0) or not is_valid_patch(patch, binary_classifier, preprocess, device):
                        continue
                    
                    negative_patches.append((i, float(conf), box_obj.copy()))
            
            # Sort by confidence and get top 3
            negative_patches.sort(key=lambda x: x[1], reverse=True)
            top_3_boxes = [patch[2] for patch in negative_patches[:3]]
            
            # Ensure 3 boxes
            while len(top_3_boxes) < 3:
                if len(top_3_boxes) > 0:
                    top_3_boxes.append(top_3_boxes[-1].copy())
                else:
                    top_3_boxes.append(np.array([0, 0, 20, 20]))
            
            boxes_dict[image_name] = [box.tolist() for box in top_3_boxes]
    
    return boxes_dict

def main(args):
    output_folder = os.path.join(args.root_path, "annotated_images_neg")
    text_file_path = os.path.join(args.root_path, "ImageClasses_FSC147.txt")
    dataset_path = os.path.join(args.root_path, "images_384_VarV2")
    input_json_path = os.path.join(args.root_path, "annotation_FSC147_384.json")
    output_json_path = os.path.join(args.root_path, "annotation_FSC147_neg_yolo_local.json")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLO World X
    yolo_model = YOLOWorld("yolov8x-worldv2.pt")
    yolo_model.to(device)
    
    boxes_dict = process_images(text_file_path, dataset_path, yolo_model, preprocess, output_folder, device=device)
    
    # Update JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    for image_name, boxes in boxes_dict.items():
        if image_name in data:
            new_boxes = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in boxes]
            data[image_name]["box_examples_coordinates"] = new_boxes
    
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO World Negative Sampling")
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the dataset")
    args = parser.parse_args()
    main(args)