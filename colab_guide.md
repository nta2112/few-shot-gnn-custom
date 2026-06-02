# Hướng dẫn chạy trên Google Colab với Custom Dataset

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy mô hình `few-shot-gnn` trên Google Colab.

---

## 1. Chuẩn bị trên Google Drive

1. **Thư mục ảnh:** Tải thư mục chứa ảnh (ở Bước 1) lên Google Drive.
   * Ví dụ: `/content/drive/MyDrive/tlu_dataset/`
2. **File JSON:** Tải file `test_split.json` lên Google Drive.
   * Ví dụ: `/content/drive/MyDrive/test_split.json`

---

## 2. Các bước trên Google Colab

### Bước 1: Kết nối Google Drive
Mở một notebook Colab mới và chạy đoạn mã sau để kết nối với Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 2: Clone repository về Colab
Chạy lệnh sau để tải mã nguồn từ GitHub của bạn và di chuyển vào thư mục dự án:
```bash
!git clone https://github.com/nta2112/few-shot-gnn-custom.git
%cd few-shot-gnn-custom
```

### Bước 3: Chạy mô hình
Chạy lệnh huấn luyện bằng cách chỉ định các đường dẫn trỏ đến Google Drive của bạn:

```bash
!python main.py \
    --dataset custom \
    --dataset_root "/content/drive/MyDrive/tlu_dataset" \
    --json_path "/content/drive/MyDrive/test_split.json" \
    --exp_name colab_custom_run \
    --train_N_way 5 \
    --train_N_shots 1 \
    --test_N_way 5 \
    --test_N_shots 1
```

---

## 3. Lưu ý & Tham số quan trọng:
* `--dataset_root`: Đường dẫn tới thư mục chứa các thư mục con tương ứng với từng class trên Google Drive.
* `--json_path`: Đường dẫn tới file JSON định nghĩa splits trên Google Drive.
* `--exp_name`: Tên lượt chạy (kết quả và checkpoint mô hình sẽ được lưu trong thư mục `checkpoints/colab_custom_run` trên bộ nhớ tạm Colab).
* **Lưu giữ kết quả:** Vì bộ nhớ Colab sẽ bị xóa sạch khi hết phiên làm việc, nếu muốn lưu trữ checkpoints vĩnh viễn trên Google Drive, bạn có thể sao chép thư mục checkpoints sang Drive bằng lệnh:
  ```bash
  !cp -r checkpoints "/content/drive/MyDrive/GPN_checkpoints"
  ```
