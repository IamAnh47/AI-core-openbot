import os
import xml.etree.ElementTree as ET
import argparse

def convert_xml_to_yolo(xml_file, img_width, img_height, classes_mapping):
    """
    xml_file: đường dẫn tới file Pascal VOC .xml
    classes_mapping: dict mapping class name -> class_id
    Trả về list các line YOLO: ["<class_id> x_center y_center w h", ba chấm]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in classes_mapping:
            continue
        class_id = classes_mapping[cls_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return yolo_lines

def main(xml_dir, img_dir, label_out_dir):
    """
    xml_dir: thư mục chứa file .xml - Pascal VOC
    img_dir: thư mục chứa ảnh tương ứng - ở đây hiện chỉ dùng JPG
    label_out_dir: thư mục save .txt
    """
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)

    # Chỉ dùng class "person" với class_id=0
    classes_mapping = {"person": 0}

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        # Tìm tên ảnh tương ứng
        base = os.path.splitext(xml_file)[0]
        jpg_path = os.path.join(img_dir, base + '.jpg')
        png_path = os.path.join(img_dir, base + '.png')
        if os.path.exists(jpg_path):
            img_path = jpg_path
        elif os.path.exists(png_path):
            img_path = png_path
        else:
            print(f"[WARN] Không tìm thấy ảnh cho {xml_file}")
            continue

        # Lấy kích thước ảnh
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Convert annotation
        yolo_lines = convert_xml_to_yolo(xml_path, w, h, classes_mapping)
        out_txt = os.path.join(label_out_dir, base + '.txt')
        with open(out_txt, 'w') as f:
            f.write("\n".join(yolo_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_dir",      type=str, required=True, help="Folder chứa .xml PascalVOC")
    parser.add_argument("--img_dir",      type=str, required=True, help="Folder chứa ảnh (.jpg/.png)")
    parser.add_argument("--label_out_dir", type=str, required=True, help="Folder lưu YOLO .txt")
    args = parser.parse_args()
    main(args.xml_dir, args.img_dir, args.label_out_dir)
