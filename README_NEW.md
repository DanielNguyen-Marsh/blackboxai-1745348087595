# Hệ thống phát hiện vi khuẩn Vibrio

Hệ thống phát hiện vi khuẩn Vibrio trong hình ảnh sử dụng YOLOv8.

## Cài đặt

1. **Tạo môi trường ảo**:
   ```
   python -m venv .venv
   ```

2. **Kích hoạt môi trường ảo**:
   - Windows:
     ```
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```
     source .venv/bin/activate
     ```

3. **Cài đặt các gói phụ thuộc**:
   ```
   pip install -r requirements.txt
   ```

## Sử dụng

### Tạo dữ liệu mẫu

Để tạo dữ liệu mẫu cho việc kiểm thử:

```
python -m vibrio_detection.main --action create-samples --num-samples 10
```

Lệnh này sẽ tạo một tập dữ liệu với cấu trúc sau:
```
dataset/
├── images/
│   ├── train/
│   │   └── ... (hình ảnh huấn luyện)
│   └── val/
│       └── ... (hình ảnh kiểm thử)
└── labels/
    ├── train/
    │   └── ... (nhãn huấn luyện ở định dạng YOLO)
    └── val/
        └── ... (nhãn kiểm thử ở định dạng YOLO)
```

### Huấn luyện mô hình

Để huấn luyện mô hình:

```
python -m vibrio_detection.main --action train --epochs 100 --batch-size 16 --image-size 640
```

Lệnh này sẽ huấn luyện mô hình YOLOv8 trên tập dữ liệu của bạn và lưu mô hình tốt nhất.

### Chạy suy luận

Để chạy suy luận trên một hình ảnh:

```
python -m vibrio_detection.main --action inference --image path/to/image.jpg --conf 0.25
```

Nếu không chỉ định hình ảnh, hệ thống sẽ tìm kiếm một hình ảnh mẫu trong tập dữ liệu.

### Đánh giá mô hình

Để đánh giá mô hình:

```
python -m vibrio_detection.main --action evaluate
```

## Xử lý sự cố

- Nếu bạn gặp lỗi về các gói bị thiếu, hãy đảm bảo bạn đã cài đặt tất cả các gói phụ thuộc:
  ```
  pip install -r requirements.txt
  ```

- Nếu bạn thấy lỗi về file mô hình bị thiếu, hãy đảm bảo bạn đã huấn luyện mô hình trước:
  ```
  python -m vibrio_detection.main --action train
  ```

- Nếu bạn thấy lỗi về file dữ liệu bị thiếu, hãy đảm bảo tập dữ liệu của bạn được tổ chức đúng cách hoặc tạo dữ liệu mẫu:
  ```
  python -m vibrio_detection.main --action create-samples
  ```

## Cấu trúc dự án

```
vibrio_detection/
├── configs/            # File cấu hình
├── data/               # Xử lý dữ liệu
├── models/             # Định nghĩa mô hình
├── scripts/            # Script cho huấn luyện, suy luận, v.v.
├── utils/              # Các hàm tiện ích
└── main.py             # Script chính
```

## Các script cũ

Các script cũ sau đây vẫn có sẵn nhưng đã bị lỗi thời:

- `train_yolov8.py`: Script cũ để huấn luyện mô hình YOLOv8
- `inference_and_evaluation.py`: Script cũ để chạy suy luận và đánh giá mô hình
- `vibrio_dataset.yaml`: File cấu hình cũ cho tập dữ liệu

## Tác giả

Vibrio Detection Team
