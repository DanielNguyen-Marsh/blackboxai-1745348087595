# Hướng Dẫn Toàn Bộ Dự Án VibrioKit

## 1. Cài Đặt Môi Trường

- Tạo và kích hoạt môi trường ảo Python (khuyến nghị):

```bash
python3 -m venv venv
source venv/bin/activate
```

- Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

- Cài đặt thêm công cụ LabelImg và YOLOv8:

```bash
pip install labelImg ultralytics
```

## 2. Chuẩn Bị Dữ Liệu

### 2.1. Thu Thập Ảnh

- Chụp nhiều ảnh đĩa cấy VibrioKit trong các điều kiện khác nhau:
  - Ánh sáng tự nhiên và đèn trong buồng chụp.
  - Mật độ khuẩn lạc khác nhau (ít, vừa, dày đặc, chồng lấp).
  - Chất lượng đĩa cấy khác nhau (mới, cũ, có vết xước nhẹ).
  - Các góc chụp khác nhau (nếu có thể, chuẩn hóa bằng buồng chụp).
  - Ảnh từ các điện thoại khác nhau.
- Cần có kết quả đếm "chuẩn" từ chuyên gia hoặc phương pháp truyền thống cho một phần dữ liệu.

### 2.2. Gán Nhãn Ảnh với LabelImg

- Khởi chạy LabelImg:

```bash
labelImg
```

- Mở thư mục chứa ảnh qua "Open Dir".
- Chọn định dạng lưu nhãn là YOLO (bên trên thanh công cụ).
- Tạo file `classes.txt` chứa tên các lớp:

```
v_para
v_algi
```

- Trong LabelImg, chọn "Change Save Dir" để chọn thư mục lưu file nhãn.
- Vẽ bounding box quanh từng khuẩn lạc và gán nhãn đúng.
- Lưu file nhãn sẽ được tạo dưới dạng `.txt` theo định dạng YOLO.

### 2.3. Cấu Trúc Thư Mục Dữ Liệu

```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

- Ảnh và file nhãn `.txt` tương ứng đặt trong các thư mục train và val.

### 2.4. Tạo File Cấu Hình Dataset cho YOLOv8

- Tạo file `vibrio_dataset.yaml` với nội dung:

```yaml
path: ./dataset
train: images/train
val: images/val

nc: 2
names: ['v_para', 'v_algi']
```

## 3. Huấn Luyện Mô Hình YOLOv8

- Tạo file `train_yolov8.py` với nội dung:

```python
from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')  # Sử dụng model YOLOv8 nano pretrained
    model.train(
        data='vibrio_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='vibrio_yolov8_model'
    )

if __name__ == '__main__':
    train_model()
```

- Chạy huấn luyện:

```bash
python train_yolov8.py
```

## 4. Chạy Suy Luận và Đánh Giá Mô Hình

- Tạo file `inference_and_evaluation.py` với nội dung:

```python
from ultralytics import YOLO

def run_inference(image_path):
    model = YOLO('runs/train/vibrio_yolov8_model/weights/best.pt')
    results = model.predict(source=image_path, conf=0.25, save=True)
    print(results)

def evaluate_model():
    model = YOLO('runs/train/vibrio_yolov8_model/weights/best.pt')
    metrics = model.val(data='vibrio_dataset.yaml')
    print(metrics)

if __name__ == '__main__':
    run_inference('dataset/images/val/sample.jpg')
    evaluate_model()
```

- Chạy suy luận và đánh giá:

```bash
python inference_and_evaluation.py
```

## 5. Lưu Ý

- Điều chỉnh các tham số huấn luyện như epochs, batch size, kích thước ảnh tùy theo tài nguyên và yêu cầu.
- Đảm bảo dữ liệu được gán nhãn chính xác để mô hình học tốt.
- Sử dụng các kỹ thuật tăng cường dữ liệu để cải thiện hiệu quả mô hình.

---

Nếu cần hỗ trợ thêm về cấu hình hoặc triển khai, vui lòng liên hệ.
