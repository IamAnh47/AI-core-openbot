import os
import glob
import shutil
import random
import argparse
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def compute_md5(file_path, block_size=65536):
    """
    Tính MD5 hash cho file tại file_path.
    """
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def ensure_dirs_split():
    """
    Tạo thư mục train/val/test :v
    """
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join("data/images", split), exist_ok=True)
        os.makedirs(os.path.join("data/labels", split), exist_ok=True)

def clear_split_dirs():
    """
    Xóa toàn bộ nội dung trong data/images/{train,val,test} và data/labels/{train,val,test}
    để tránh trùng lặp cũ. - không xử lý đoạn này mô hình toang quá :))
    """
    for split in ["train", "val", "test"]:
        img_dir   = os.path.join("data/images", split)
        label_dir = os.path.join("data/labels", split)
        if os.path.isdir(img_dir):
            for f in glob.glob(os.path.join(img_dir, "*")):
                os.remove(f)
        if os.path.isdir(label_dir):
            for f in glob.glob(os.path.join(label_dir, "*")):
                os.remove(f)

def ensure_report_dir(report_dir):
    os.makedirs(report_dir, exist_ok=True)

def split_data(args):
    """
    - Xóa sạch folder split
    - Đọc raw_images & raw_labels, loại bỏ ảnh duplicate dùng MD5
    - Chia ngẫu nhiên theo train/val/test tỉ lệ args.train_ratio, args.val_ratio, args.test_ratio
    - Copy ảnh và label tương ứng
    - Vẽ và lưu các biểu đồ đánh giá vào thư mục args.report_dir
    """
    raw_img_dir = args.raw_images
    raw_lbl_dir = args.raw_labels
    report_dir  = args.report_dir

    #Xóa sạch folder split
    clear_split_dirs()
    ensure_dirs_split()

    #Đảm bảo report_dir tồn tại
    ensure_report_dir(report_dir)

    #Lấy danh sách ảnh raw
    all_files = sorted([
        f for f in os.listdir(raw_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    #Loại bỏ ảnh trùng lặp
    unique_files = []
    seen_hashes = set()
    for img_name in all_files:
        img_path = os.path.join(raw_img_dir, img_name)
        file_hash = compute_md5(img_path)
        if file_hash in seen_hashes:
            print(f"[DUPLICATE] Bỏ qua {img_name} vì đã có ảnh giống (hash={file_hash})")
            continue
        seen_hashes.add(file_hash)
        unique_files.append(img_name)

    # Shuffle nếu cần
    if args.shuffle:
        random.shuffle(unique_files)

    # Tính số lượng cho mỗi split
    num_total = len(unique_files)
    num_train = int(num_total * args.train_ratio)
    num_val   = int(num_total * args.val_ratio)
    num_test  = num_total - num_train - num_val

    print(f"[INFO] Tổng ảnh duy nhất sau lọc trùng: {num_total}")
    print(f"[INFO] Train: {num_train}, Val: {num_val}, Test: {num_test}")

    train_files = unique_files[:num_train]
    val_files   = unique_files[num_train:num_train + num_val]
    test_files  = unique_files[num_train + num_val:] if num_test > 0 else []

    def copy_subset(file_list, split):
        for img_name in file_list:
            base, ext = os.path.splitext(img_name)
            src_img = os.path.join(raw_img_dir, img_name)
            src_lbl = os.path.join(raw_lbl_dir, base + ".txt")
            dst_img = os.path.join("data/images", split, img_name)
            dst_lbl = os.path.join("data/labels", split, base + ".txt")

            # Copy ảnh nếu chưa tồn tại
            if not os.path.isfile(dst_img):
                shutil.copy2(src_img, dst_img)
            else:
                print(f"[SKIP] Ảnh {img_name} đã tồn tại ở data/images/{split}/")

            # Copy label nếu tồn tại, nếu không thì tạo file rỗng
            if os.path.isfile(src_lbl):
                if not os.path.isfile(dst_lbl):
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    print(f"[SKIP] Label {base}.txt đã tồn tại ở data/labels/{split}/")
            else:
                if not os.path.isfile(dst_lbl):
                    open(dst_lbl, "w").close()
                    print(f"[INFO] Tạo label rỗng cho {base}.txt tại split={split}")

    copy_subset(train_files, "train")
    copy_subset(val_files, "val")
    if num_test > 0:
        copy_subset(test_files, "test")

    print("[INFO] Chia dữ liệu hoàn tất.\n")

# ----------- Phần vẽ và lưu biểu đồ đánh giá - này để mình bỏ vào document -------------
    # Thu thập thông tin về số ảnh và bbox từ mỗi split
    split_stats = {'train': {'num_images': len(train_files), 'bboxes': []},
                   'val':   {'num_images': len(val_files),   'bboxes': []},
                   'test':  {'num_images': len(test_files),  'bboxes': []}}

    # Đọc annotation để lấy bbox
    for split in ['train', 'val', 'test']:
        label_dir = os.path.join("data/labels", split)
        if not os.path.isdir(label_dir):
            continue
        files = glob.glob(os.path.join(label_dir, '*.txt'))
        for file in files:
            with open(file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                x_center, y_center, w_bbox, h_bbox = map(float, parts[1:])
                split_stats[split]['bboxes'].append((x_center, y_center, w_bbox, h_bbox))

    # Hàm phụ để tạo tên file với timestamp
    def timestamped_name(prefix):
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{t}.png"

    #Biểu đồ số lượng ảnh mỗi split
    labels_plot = list(split_stats.keys())
    counts_plot = [split_stats[s]['num_images'] for s in labels_plot]

    plt.figure(figsize=(6, 4))
    plt.bar(labels_plot, counts_plot, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Số lượng ảnh theo split')
    plt.ylabel('Số ảnh')
    plt.xlabel('Split')
    filename1 = os.path.join(report_dir, timestamped_name("num_images_per_split"))
    plt.savefig(filename1)
    plt.close()
    print(f"[SAVED CHART] {filename1}")

    #Phân bố độ rộng & độ cao bbox (normalized) -- gộp cả 3 split
    all_widths = []
    all_heights = []
    all_aspect = []
    for s in labels_plot:
        for bbox in split_stats[s]['bboxes']:
            _, _, w_bbox, h_bbox = bbox
            all_widths.append(w_bbox)
            all_heights.append(h_bbox)
            if h_bbox > 0:
                all_aspect.append(w_bbox / h_bbox)

    plt.figure(figsize=(6, 4))
    plt.hist(all_widths, bins=20, alpha=0.7, label='Width')
    plt.hist(all_heights, bins=20, alpha=0.7, label='Height')
    plt.title('Phân bố độ rộng & độ cao bbox (normalized)')
    plt.xlabel('Giá trị chuẩn hóa')
    plt.ylabel('Số lượng')
    plt.legend()
    filename2 = os.path.join(report_dir, timestamped_name("bbox_wh_distribution"))
    plt.savefig(filename2)
    plt.close()
    print(f"[SAVED CHART] {filename2}")

    #Phân bố tỷ lệ khung hình (aspect ratio)
    plt.figure(figsize=(6, 4))
    plt.hist(all_aspect, bins=20, color='violet')
    plt.title('Phân bố tỷ lệ khung hình bbox (width/height)')
    plt.xlabel('Aspect ratio')
    plt.ylabel('Số lượng')
    filename3 = os.path.join(report_dir, timestamped_name("bbox_aspect_ratio"))
    plt.savefig(filename3)
    plt.close()
    print(f"[SAVED CHART] {filename3}")

    #Scatter plot tọa độ trung tâm bbox
    centers_x = [bbox[0] for s in labels_plot for bbox in split_stats[s]['bboxes']]
    centers_y = [bbox[1] for s in labels_plot for bbox in split_stats[s]['bboxes']]

    plt.figure(figsize=(6, 6))
    plt.scatter(centers_x, centers_y, s=5, alpha=0.5)
    plt.title('Tọa độ trung tâm bbox (normalized)')
    plt.xlabel('X_center')
    plt.ylabel('Y_center')
    plt.gca().invert_yaxis()  # Đảo tung y-axis cho đúng tọa độ ảnh
    filename4 = os.path.join(report_dir, timestamped_name("bbox_center_scatter"))
    plt.savefig(filename4)
    plt.close()
    print(f"[SAVED CHART] {filename4}")

    #Lưu bảng tóm tắt ra CSV (tuỳ chọn)
    df_summary = pd.DataFrame({
        'Split': labels_plot,
        'NumImages': counts_plot,
        'NumBBoxes': [len(split_stats[s]['bboxes']) for s in labels_plot]
    })
    csv_path = os.path.join(report_dir, timestamped_name("summary")[:-4] + ".csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"[SAVED SUMMARY CSV] {csv_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_images", type=str, default="data/raw/images",
        help="Thư mục chứa ảnh raw"
    )
    parser.add_argument(
        "--raw_labels", type=str, default="data/raw/annotations",
        help="Thư mục chứa annotation raw"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Tỉ lệ dành cho training (mặc định 0.8)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="Tỉ lệ dành cho validation (mặc định 0.2)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.0,
        help="Tỉ lệ dành cho test (mặc định 0.0)"
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Có shuffle file trước khi chia"
    )
    parser.add_argument(
        "--report_dir", type=str, default="reports",
        help="Đường dẫn thư mục để lưu biểu đồ và báo cáo (mặc định 'reports/')"
    )
    args = parser.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        print("[ERROR] Tổng train_ratio + val_ratio + test_ratio phải = 1.0")
        return

    split_data(args)

if __name__ == "__main__":
    main()
