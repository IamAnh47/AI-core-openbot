#!/bin/bash
# scripts/train_custom.sh
# Usage: bash scripts/train_custom.sh <epochs> <batch_size>
EPOCHS=${1:-50}
BATCH=${2:-8}

cd "$(dirname "$0")"/..

echo "=== Bắt đầu fine-tune YOLOv10-N ($EPOCHS epochs, batch=$BATCH) ==="
yolo task=detect mode=train \
     model=weights/pretrain/yolov10n.pt \
     data=cfg/yolo_data.yaml \
     epochs=$EPOCHS \
     imgsz=320 \
     batch=$BATCH \
     project=weights/custom \
     name=exp_custom
echo "=== Fine-tune hoàn tất. Kết quả nằm ở weights/custom/exp_custom/ ==="

# bash scripts/train_custom.sh 50 8