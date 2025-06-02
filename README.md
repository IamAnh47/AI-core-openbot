# Robot AI Core

## Mục tiêu
- Detect “person” và “obstacle” trong khung hình real-time bằng YOLOv10-N.
- Track ID duy nhất cho mỗi người (DeepSORT), chỉ follow 1 người.
- Tính toán lệnh điều khiển (linear_vel, angular_vel) để robot bám theo và né chướng ngại.
- Ghi nhớ đường đi - waypoints để tìm lại khi mất người.
- Hỗ trợ fine-tune model với dataset riêng, test inference trên video.
- Còn chức năng vẽ bản đồ và đường đi thì tui chưa thết lập

## Cấu trúc dự án
- Mn xem cấu trúc dự án có thể thêm những folder bị ẩn do qua nặng :v
```
robot_ai_core/               
├── cfg/                      
│   ├── coco.names            # Danh sách 80 class COCO, dòng đầu là “person”
│   └── yolo_data.yaml        # Config data cho Ultralytics YOLOv10 fine-tune
│
├── data/       
│   ├── raw/                  # Thư mục chứa ảnh và annotation “thô”
│   │   ├── images/           # Ảnh chụp raw (chưa chia)
│   │   │   └── ...           
│   │   └── annotations/      # File annotation tương ứng (YOLO txt)
│   │       └── ...                        
│   ├── images/               # Ảnh đã chia train/val/test
│   │   ├── train/            
│   │   │   ├── img00001.jpg    
│   │   │   ├── img00002.jpg    
│   │   │   └── …             
│   │   ├── val/              
│   │   │   ├── img10001.jpg   
│   │   │   └── …             
│   │   └── test/             
│   │       ├── img20001.jpg   
│   │       └── …             
│   │
│   └── labels/               # File annotation định dạng YOLO (.txt)
│       ├── train/            
│       │   ├── img001.txt    
│       │   ├── img002.txt    
│       │   └── ...             
│       ├── val/              
│       │   ├── img1001.txt   
│       │   └── ...           
│       └── test/             
│           ├── img2001.txt   
│           └── ...            
│
├── weights/                  
│   ├── pretrain/             # Model YOLOv10-N gốc (trên COCO)
│   │   └── yolov10n.pt       
│   ├── custom/               # Chứa kết quả fine-tune của Ultralytics
│   │   ├── exp1/             
│   │   │   ├── weights/      
│   │   │   │   ├── best.pt   
│   │   │   │   └── last.pt   
│   │   │   └── results.png   
│   │   └── exp2/             
│   │       ├── weights/      
│   │       └── results.png   
│   └── yolov10n.tflite       # Model TFLite đã export (inference)
│
├── scripts/                  
│   ├── train_custom.sh       # Script để fine-tune YOLOv10-N với dataset riêng
│   ├── export_to_tflite.py   # Script eksport model .pt -> .tflite
│   ├── infer_video.py        # Script standalone để test inference trên video
│   └── convert_annotations.py# (tùy chọn) hỗ trợ convert annotation nếu cần
│   └── data_process.py       # Mới: chia data/raw -> data/images/{train,val}/ labels/{train,val}/
│
├── src/                      
│   ├── config.py             # Cấu hình chung gồm: paths, thresholds, parameters
│   ├── detector.py           # Detector TFLite (và fallback Ultralytics) 
│   ├── tracker.py            # DeepSORT tracker để gán ID và track
│   ├── controller.py         # Tính toán linear_vel, angular_vel
│   ├── path_memory.py        # Lưu waypoint dựa trên odometry/SLAM (giả lập) - chưa dùng
│   ├── robot_interface.py    # Giao tiếp robot - chưa dùng
│   └── main.py               # Entrypoint: capture -> detect -> track -> cmd -> gửi
│
├── utils/                    
│   ├── preprocessor.py       # Hàm tiền xử lý ảnh (resize, normalize)  
│   ├── viz.py                # Hàm vẽ bounding box, ID, lệnh, đường đi  
│   └── metrics.py            # (tùy chọn) tính IoU, mAP nếu cần  
│
├── notebooks/                
│   ├── train_explore.ipynb   # (tùy chọn) Notebook phân tích dataset, loss curves  - chưa dùng
│   └── infer_demo.ipynb      # (tùy chọn) Notebook demo inference video  - chưa dùng
│
├── videos/                   
│   ├── sample_test.mp4       # Video mẫu để test inference  
│   └── output_test.mp4       # -> file này sẽ được tạo sau khi mình test với video
│
├── requirements.txt          
├── README.md                 
└── LICENSE                   
```

## Hướng dẫn cài đặt & chạy

1. **Clone repo**  
   ```bash
   git clone <repo_url>
   cd robot_ai_core
   ```
   
2. Tạo và kích hoạt virtualenv (Python 3.8+)

   ```bash
   python3 -m venv venv
   source venv/bin/activate     # Linux/macOS
   venv\Scripts\activate        # Windows
   ```

3. Cài dependencies

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Chuẩn bị model YOLOv10-N**

- Model gốc yolov10n.pt đã nằm trong weights/pretrain/.

- Nếu chưa có, Ultralytics sẽ tự động tải gọi detect/fine-tune.

5. **Test inference trên video mẫu**
   ```bash
   python scripts/infer_video.py --video videos/sample_test.mp4 --output videos/output_test.mp4
   ```
6. **Chạy pipeline end-to-end (AI Core)**
   ```bash
   # Không thu thập, chỉ mở cam, mn thay 0 bằng vides/tên video.mp4
   python src/main.py --source 0
   # Thu thập qua camera
   python src/main.py --collect --collect_duration 30 --collect_freq 1.0 --source 0
   # Thu thập qua video
   python src/main.py --collect --collect_duration 30 --collect_freq 1.0 --source videos/sample_test.mp4
   ```
-> Chương trình bật camera, mỗi giây chụp 1 ảnh, lưu ảnh vào data/raw/images/img00001.jpg, img00002.jpg,…; file annotation (dựa YOLO detection) vào data/raw/annotations/img00001.txt,… Sau 30 giây, tự động dừng.
- Mở webcam (hoặc sửa trong main.py thành video file).

- Hiển thị bounding box, ID, FPS và in ra lệnh điều khiển (console hoặc gửi serial).

- Nhấn q để thoát.
7. **Fine-tune YOLOv10-N với dataset riêng**

- Chuẩn bị dataset theo hướng dẫn (ảnh trong data/images/train, label trong data/labels/train, v.v.).

- Sửa cfg/yolo_data.yaml cho đúng đường dẫn.
   ```bash
   bash scripts/train_custom.sh 50 8
   ```
-> Kết quả lưu trong ```weights/custom/exp_custom/weights/best.pt.```
8. Export model fine-tune sang TFLite - đang lỗi ...............
   ```bash
   python scripts/export_to_tflite.py --weights weights/custom/exp_custom/weights/best.pt --imgsz 320 --output weights/yolov10n_custom.tflite
   ```
   ```bash
   pip install --upgrade ultralytics onnx onnx-tf tensorflow
   python scripts/export_to_tflite.py --weights weights/pretrain/yolov10n.pt --imgsz 320 --output weights/yolov10n.tflite
   
   python scripts/export_to_tflite.py
   ```
9. Đánh giá mAP trên tập validation - chưa chỉnh sửa nên không dùng
   ```bash
   yolo task=detect mode=val \
     model=weights/custom/exp_custom/weights/best.pt \
     data=cfg/yolo_data.yaml \
     imgsz=320
   ```
## Một số chế độ khác
- Chia data và biểu đồ phân tích
```bash
# Chia raw -> train/val 80/20 - mặc định, không có test
python scripts/data_process.py

# Nếu muốn 70% train, 20% val, 10% test, shuffle trước
python scripts/data_process.py --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 --shuffle --report_dir eval_plots
```
- Định dạng data
```bash
<class_id> <x_center> <y_center> <width> <height>
```
```bash
#Ví dụ: 0 0.542188 0.573958 0.846875 0.843750
#0: class_id 
#
#0.542188: x_center – tọa độ trung tâm bbox theo chiều ngang, đã được chuẩn hóa (normalized).
#
#0.573958: y_center – tọa độ trung tâm bbox theo chiều dọc, đã được chuẩn hóa.
#
#0.846875: width – độ rộng của bbox, đã được chuẩn hóa (= bbox_width / image_width).
#
#0.843750: height – độ cao của bbox, đã được chuẩn hóa (= bbox_height / image_height).
```

## Liên hệ 
Email: anh.phanitskyye@hcmut.edu.vn 