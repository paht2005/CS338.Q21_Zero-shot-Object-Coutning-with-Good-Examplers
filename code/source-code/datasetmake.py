from PIL import Image
import os
import random
print(os.getcwd())
def is_image_file(filename):
    """判断文件是否是图像文件"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # 支持的图像文件扩展名列表
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def random_crop(img, size=(256, 256)):
    """从给定的图片中随机裁剪出指定大小的区域"""
    width, height = img.size
    crop_width, crop_height = size

    if width < crop_width or height < crop_height:
        return None  # 如果图片尺寸小于裁剪尺寸，则返回None

    x_left = random.randint(0, width - crop_width)
    y_upper = random.randint(0, height - crop_height)

    return img.crop((x_left, y_upper, x_left + crop_width, y_upper + crop_height))

import json 
def create_box_image_dataset():
    # load các tên ảnh tập train
    img_dir = './data/FSC147/images_384_VarV2'
    anno_file = './data/FSC147/annotation_FSC147_384.json'
    train_txt_name_path = './data/FSC147/train.txt'
    with open(train_txt_name_path, 'r') as f:
        train_img_names = f.read().splitlines()
    train_img_set = set(train_img_names)
    # load annotation
    with open(anno_file, 'r') as f:
        annotations = json.load(f)
    # lọc annotation chỉ giữ ảnh trong tập train
    train_annotations = {k: v for k, v in annotations.items() if k in train_img_set}
    # tạo folder lưu ảnh examplar
    exemplar_folder = './data/FSC147/box'
    if not os.path.exists(exemplar_folder):
        os.makedirs(exemplar_folder)
    # duyệt qua từng ảnh trong tập train
    for img_name, anno in train_annotations.items():
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        boxes = anno['box_examples_coordinates']
        # lưu từng box thành ảnh riêng
        for i, box in enumerate(boxes):
            x1, y1 = box[0]
            x2, y2 = box[2]
            box_img = img.crop((x1, y1, x2, y2))
            box_img_save_path = os.path.join(exemplar_folder, f"{img_name[:-4]}_box_{i}.jpg")
            box_img.save(box_img_save_path)


create_box_image_dataset() # tạo box
    

# 文件夹路径设置（根据实际情况修改）
single_object_folder = './data/FSC147/box'
multiple_objects_folder = './data/FSC147/images_384_VarV2'
output_folder = './data/FSC147/one'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

output_txt_path = os.path.join(output_folder, 'labels.txt')
with open(output_txt_path, 'w') as f:
    for folder, label in [(single_object_folder, 'one'), (multiple_objects_folder, 'more')]:
        for filename in os.listdir(folder):
            if is_image_file(filename):  # 只处理图像文件
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)

                # 保存原图并记录到txt文件
                original_img_output_path = os.path.join(output_folder, filename)
                img.save(original_img_output_path)
                f.write(f"{filename},{label}\n")

                # 从原图中随机裁剪并保存裁剪图像
                for size in [(256, 384), (256, 256), (384, 384),(128,256),(256,128)]:
                    img_cropped = random_crop(img, size=size)
                    if img_cropped:
                        cropped_img_output_path = os.path.join(output_folder, f"{filename[:-4]}_random_{size[0]}x{size[1]}.jpg")
                        img_cropped.save(cropped_img_output_path)
                        f.write(f"{filename[:-4]}_random_{size[0]}x{size[1]}.jpg,{label}\n")

print("数据集准备完成。")
