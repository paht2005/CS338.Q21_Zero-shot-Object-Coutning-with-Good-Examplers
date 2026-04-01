import torch
import os
import clip
import argparse
from torchvision.ops import box_convert
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image
import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import inflect  # <--- [MỚI] Import thư viện xử lý ngữ pháp

# --- CẤU HÌNH ---
device = "cuda" if torch.cuda.is_available() else "cpu"
BOX_THRESHOLD = 0.05
TEXT_THRESHOLD = 0.05
# Alpha * groundingdino_score ( logits) + Beta * clip_score
ALPHA = 0 # Trọng số GroundingDINO
BETA = 1   # Trọng số CLIP

# --- [MỚI] KHỞI TẠO INFLECT ENGINE ---
p = inflect.engine()

# --- KHỞI TẠO MODEL PHỤ ---
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

# Load Models
print("Loading CLIP B/32 for Binary Classifier...")
clip_model_b32, preprocess_b32 = clip.load("ViT-B/32", device)
clip_model_b32.eval()

print("Loading CLIP L/14 for Scoring...")
clip_model_l14, preprocess_l14 = clip.load("ViT-L/14", device)
clip_model_l14.eval()

print("Loading Binary Classifier...")
binary_classifier = ClipClassifier(clip_model_b32).to(device)
model_weights_path = './data/out/classify/best_model.pth'
if os.path.exists(model_weights_path):
    binary_classifier.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=False))
else:
    print("WARNING: Không tìm thấy weight classifier, model sẽ chạy kém chính xác.")
binary_classifier.eval()

# Hàm kiểm tra hợp lệ
def is_valid_patch(patch, binary_classifier, preprocess_b32, device):
    if patch.size[0] <= 0 or patch.size[1] <= 0: return False
    patch_tensor = preprocess_b32(patch).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = binary_classifier(patch_tensor)
        prob = torch.softmax(logits, dim=1)[0, 1]
    return prob.item() > 0.8 # Ngưỡng lọc Single Object

# --- HÀM ĐỌC FILE TXT PROMPT ---
def load_image_specific_prompts(txt_path):
    prompts_dict = {}
    if not os.path.exists(txt_path):
        print(f"WARNING: Không tìm thấy file prompt {txt_path}")
        return prompts_dict
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            print("line:", line)
            if not line: continue
            parts = line.split(None,1)
            print("parts:", len(parts))
            if len(parts) >= 2:
                img_name = parts[0].strip()
                prompt = parts[1].strip()
                print("img_name, prompt:", img_name, prompt)
                if not prompt.endswith('.'):
                    prompt += ' .'
                prompts_dict[img_name] = prompt
    return prompts_dict

# --- HÀM XỬ LÝ CHÍNH ---
def process_images(class_file_path, dataset_path, model, preprocess_b32, preprocess_l14, clip_model_l14, image_prompts_dict, device='gpu'):
    boxes_dict = {}
    
    with open(class_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(None,1)
            if len(parts) < 2: continue
            image_name, class_name = parts[0], parts[1]
            
            # --- [MỚI] XỬ LÝ SỐ ÍT (SINGULARIZATION) ---
            # Chuyển "keys" -> "key", "cars" -> "car"
            singular_name = p.singular_noun(class_name)
            if not singular_name: # Nếu trả về False nghĩa là nó đã là số ít rồi
                singular_name = class_name
            print('Singular :', singular_name)
            # -------------------------------------------

            image_path = os.path.join(dataset_path, image_name)
            if not os.path.exists(image_path): continue

            # Load ảnh
            image_source, image = load_image(image_path)
            pil_img = Image.open(image_path).convert("RGB")
            h, w, _ = image_source.shape
            
            # === BƯỚC 1: Dùng class name gốc với Grounding DINO ===
            print(f"Processing: {image_name} | Step 1: Using base class name: {singular_name}")
            base_prompt = f"single {singular_name} ."
            try:
                boxes, logits, _ = predict(model, image, base_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
            except IndexError:
                boxes, logits = torch.tensor([]), torch.tensor([])
            # === BƯỚC 1: Dùng class name gốc với Grounding DINO ===
            print(f"Processing: {image_name} | Step 1: Using base class name: {singular_name}")
            base_prompt = f"single {singular_name} ."
            try:
                boxes, logits, _ = predict(model, image, base_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
            except IndexError:
                boxes, logits = torch.tensor([]), torch.tensor([])
                
            patches = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy") if len(boxes) > 0 else torch.tensor([])

            # Filter và tính score cho các boxes từ base prompt
            candidate_patches_base = []
            
            for i, (box, gd_logit) in enumerate(zip(patches, logits)):
                box_np = box.cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)
                x1, y1, x2, y2 = box_np.astype(int)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                
                if x2 - x1 < 5 or y2 - y1 < 5: continue
                patch = pil_img.crop((x1, y1, x2, y2))
                if patch.size == (0,0): continue

                # Binary Filter (Single Object Check)
                if not is_valid_patch(patch, binary_classifier, preprocess_b32, device):
                    continue
                
                # Lưu với score là logit từ Grounding DINO
                candidate_patches_base.append((box_np, gd_logit.item()))
            
            # Sort theo logits và lấy top 5
            candidate_patches_base.sort(key=lambda x: x[1], reverse=True)
            top_boxes = candidate_patches_base[:5]
            
            print(f"  -> Found {len(top_boxes)} valid boxes from base prompt")
            
            # === BƯỚC 2: Nếu không đủ 5, dùng Rich Prompt + CLIP ===
            if len(top_boxes) < 5:
                print(f"  -> Step 2: Not enough boxes, using rich prompt + CLIP")
                
                # Lấy rich prompt nếu có
                if image_name in image_prompts_dict:
                    text_prompt = class_name + ' . ' + image_prompts_dict[image_name]
                    print(f"     Rich Prompt: {text_prompt[:100]}...")
                else:
                    text_prompt = base_prompt
                
                # Predict lại với rich prompt
                try:
                    boxes_rich, logits_rich, _ = predict(model, image, text_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
                except IndexError:
                    boxes_rich, logits_rich = torch.tensor([]), torch.tensor([])
                
                patches_rich = box_convert(boxes_rich, in_fmt="cxcywh", out_fmt="xyxy") if len(boxes_rich) > 0 else torch.tensor([])
                
                # Chuẩn bị Text Feature cho CLIP L/14 Scoring
                text_sim_input = clip.tokenize([f"a photo of a single {singular_name}"]).to(device)
                with torch.no_grad():
                    text_sim_emb = clip_model_l14.encode_text(text_sim_input).float()
                    text_sim_emb /= text_sim_emb.norm(dim=-1, keepdim=True)

                candidate_patches_rich = []
                
                # Lấy các boxes đã có để tránh trùng
                existing_boxes_set = set([tuple(box.tolist()) for box, _ in top_boxes])
                
                for i, (box, gd_logit) in enumerate(zip(patches_rich, logits_rich)):
                    box_np = box.cpu().numpy() * np.array([w, h, w, h], dtype=np.float32)
                    
                    # Kiểm tra trùng
                    if tuple(box_np.tolist()) in existing_boxes_set:
                        continue
                    
                    x1, y1, x2, y2 = box_np.astype(int)
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
                    
                    if x2 - x1 < 5 or y2 - y1 < 5: continue
                    patch = pil_img.crop((x1, y1, x2, y2))
                    if patch.size == (0,0): continue

                    # Binary Filter
                    if not is_valid_patch(patch, binary_classifier, preprocess_b32, device):
                        continue

                    # Tính CLIP Score với L/14
                    patch_input = preprocess_l14(patch).unsqueeze(0).to(device)
                    with torch.no_grad():
                        patch_emb = clip_model_l14.encode_image(patch_input).float()
                        patch_emb /= patch_emb.norm(dim=-1, keepdim=True)
                        clip_score = (patch_emb @ text_sim_emb.T).item()
                    
                    # Final Score = Alpha * groundingdino_logit + Beta * clip_score
                    final_score = (ALPHA * gd_logit.item()) + (BETA * clip_score)
                    candidate_patches_rich.append((box_np, final_score))
                
                # Sort và lấy thêm để đủ 5
                candidate_patches_rich.sort(key=lambda x: x[1], reverse=True)
                need_more = 5 - len(top_boxes)
                top_boxes.extend(candidate_patches_rich[:need_more])
                
                print(f"  -> Added {min(need_more, len(candidate_patches_rich))} boxes from rich prompt")

            # Chuyển sang format cuối cùng
            final_boxes = [box for box, score in top_boxes]
            
            # Fill dummy nếu vẫn thiếu
            while len(final_boxes) < 5:
                if len(final_boxes) > 0:
                    final_boxes.append(final_boxes[-1])
                else:
                    # Tạo dummy box nhỏ ở góc
                    final_boxes.append(np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32))
                print(f"  -> Filled with dummy box")

            boxes_dict[image_name] = [box.tolist() for box in final_boxes[:5]]

    return boxes_dict

def main(args):
    # Đường dẫn
    model_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    output_folder = os.path.join(args.root_path, "annotated_images")
    os.makedirs(output_folder, exist_ok=True)

    # File input gốc (Class mapping)
    class_file_path = os.path.join(args.root_path, "ImageClasses_FSC147.txt")
    
    # File Prompt
    rich_prompts_path = os.path.join(args.root_path, "ImageClasses_FSC147_detailed_v6_pos.txt") 

    dataset_path = os.path.join(args.root_path, "images_384_VarV2")
    input_json = os.path.join(args.root_path, "annotation_FSC147_384.json")
    output_json = os.path.join(args.root_path, "annotation_FSC147_pos_final.json")

    # Load dữ liệu
    print(f"Loading Rich Prompts from {rich_prompts_path}...")
    image_prompts_dict = load_image_specific_prompts(rich_prompts_path)
    
    print("Loading Grounding DINO...")
    model = load_model(model_config, model_weights, device=device)

    # Xử lý
    boxes_dict = process_images(class_file_path, dataset_path, model, preprocess_b32, preprocess_l14, clip_model_l14, image_prompts_dict, device=device)

    # Lưu JSON
    print("Saving JSON...")
    if os.path.exists(input_json):
        with open(input_json, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    for img, boxes in boxes_dict.items():
        new_boxes = [[[x1, y1], [x1, y2], [x2, y2], [x2, y1]] for x1, y1, x2, y2 in boxes]
        if img in data:
            data[img]["box_examples_coordinates"] = new_boxes
    
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    args = parser.parse_args()
    main(args)