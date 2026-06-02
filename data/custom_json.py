import os
import json
import glob
import numpy as np
from PIL import Image as pil_image
import torch.utils.data as data
import torchvision.transforms as transforms

class CustomJSONSplitDataset(data.Dataset):
    def __init__(self, root, json_path):
        """
        root: Thư mục chứa các folder con của từng class (ví dụ: "C:/Users/HP/OneDrive/Desktop/your_dataset_images")
        json_path: Đường dẫn tới file test_split.json
        """
        self.root = root
        with open(json_path, 'r') as f:
            self.split_classes = json.load(f)
            
    def load_dataset(self, partition, size=(84, 84)):
        # partition ở đây nhận giá trị 'train', 'val', hoặc 'test'
        classes_in_partition = self.split_classes[partition]
        
        # Gom toàn bộ các lớp để mã hóa đồng nhất tên lớp thành chỉ số số nguyên (class ID)
        all_classes = []
        for p in ['train', 'val', 'test']:
            all_classes.extend(self.split_classes[p])
        
        all_classes = sorted(list(set(all_classes)))
        label_encoder = {class_name: idx for idx, class_name in enumerate(all_classes)}
        
        # Cấu hình transform chuẩn theo phong cách AGNN
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        box_size = int(size[0] * 1.15)
        
        if partition == 'train':
            # Tập train: lưu PIL đã resize, phần augmentation ngẫu nhiên sẽ áp dụng khi sinh batch
            self.transform = transforms.Resize((box_size, box_size))
            self.aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(size[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                normalize
            ])
        else:
            # Tập val/test: biến đổi cố định hoàn toàn
            self.transform = transforms.Compose([
                transforms.Resize((box_size, box_size)),
                transforms.CenterCrop(size[0]),
                transforms.ToTensor(),
                normalize
            ])
            self.aug_transform = None
            
        data_dict = {}
        for class_name in classes_in_partition:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                print(f"Cảnh báo: Không tìm thấy thư mục của lớp {class_dir}")
                continue
                
            # Tìm tất cả các file ảnh trong thư mục lớp này
            image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
            images_path = []
            for ext in image_extensions:
                images_path.extend(glob.glob(os.path.join(class_dir, ext)))
                
            class_idx = label_encoder[class_name]
            data_dict[class_idx] = []
            
            for path in images_path:
                try:
                    img = pil_image.open(path).convert('RGB')
                    if partition == 'train':
                        # Giữ nguyên ảnh dạng PIL (đã resize sơ bộ) để augment ngẫu nhiên mỗi lần lấy
                        img_processed = self.transform(img)
                    else:
                        # Thực hiện đầy đủ các bước tiền xử lý cố định rồi chuyển thành numpy
                        tensor_img = self.transform(img)
                        img_processed = tensor_img.numpy()
                        
                    data_dict[class_idx].append(img_processed)
                except Exception as e:
                    print(f"Lỗi khi đọc file ảnh {path}: {e}")
                    
        print(f"Loaded {partition} split (AGNN-style): {len(data_dict)} classes, {sum(len(v) for v in data_dict.values())} images.")
        return data_dict, label_encoder
