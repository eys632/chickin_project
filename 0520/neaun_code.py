import os
import cv2
import csv
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# === 설정 ===
video_path = "/home/nas/data/YeonSeung/chicken_30sec - Trim.mp4"
model_path = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt"
base_save_dir = "/home/a202192020/객체탐지 프로젝트(송교수님)/0520"

# 실행 시각 기반 폴더 생성
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join(base_save_dir, timestamp)
os.makedirs(save_dir, exist_ok=True)

tracking_csv_path = os.path.join(save_dir, f"{timestamp}_tracking.csv")
mapping_csv_path = os.path.join(save_dir, f"{timestamp}_id_mapping.csv")
output_video_path = os.path.join(save_dir, f"{timestamp}_output.mp4")

# YOLO 분석 설정
grid_size = 3
selected_cell = (2, 0)
conf_threshold = 0.5

# IOU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (area1 + area2 - inter_area)

# 모델 및 비디오 로드
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_idx = 0
next_id = 1
last_detections = []  # [((x1, y1, x2, y2), obj_id)]
id_mappings = {}      # {new_id: prev_id}

# 셀 크기 계산
row, col = selected_cell
cell_h = height // grid_size
cell_w = width // grid_size

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (cell_w, cell_h))

# CSV 초기화
tracking_file = open(tracking_csv_path, mode='w', newline='')
tracking_writer = csv.writer(tracking_file)
tracking_writer.writerow(['frame', 'id', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'prev_id'])

mapping_file = open(mapping_csv_path, mode='w', newline='')
mapping_writer = csv.writer(mapping_file)
mapping_writer.writerow(['frame', 'new_id', 'prev_id', 'iou'])

# === 분석 루프 시작 ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # 선택 셀 영역만 잘라내기
    x, y = col * cell_w, row * cell_h
    cell_frame = frame[y:y+cell_h, x:x+cell_w]

    # YOLO 탐지
    results = model.track(cell_frame, conf=conf_threshold, persist=True, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [-1] * len(boxes)

    new_detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        obj_id = ids[i]
        prev_id = ""

        # IOU 기반 ID 매칭
        if obj_id == -1:
            matched = False
            for last_box, last_id in last_detections:
                iou = calculate_iou(box, last_box)
                if iou > 0.5:
                    obj_id = next_id
                    prev_id = last_id
                    id_mappings[obj_id] = prev_id
                    mapping_writer.writerow([frame_idx, obj_id, prev_id, round(iou, 3)])
                    next_id += 1
                    matched = True
                    break
            if not matched:
                obj_id = next_id
                next_id += 1
        else:
            if obj_id in id_mappings:
                prev_id = id_mappings[obj_id]

        new_detections.append(((x1, y1, x2, y2), obj_id, prev_id))
        tracking_writer.writerow([frame_idx, obj_id, x1, y1, x2, y2, cx, cy, prev_id])

    # 시각화 영상 프레임 생성
    plotted = cell_frame.copy()
    for (x1, y1, x2, y2), obj_id, prev_id in new_detections:
        label = f"ID {obj_id}"
        if prev_id != "":
            label += f" (prev: {prev_id})"
        cv2.rectangle(plotted, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(plotted, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    video_writer.write(plotted)

    # 진행률 출력
    percent = (frame_idx / total_frames) * 100
    print(f"\r진행 중... {percent:.2f}%", end="")

# 종료 처리
cap.release()
video_writer.release()
tracking_file.close()
mapping_file.close()
print(f"\n✅ 완료! 결과 저장됨: {save_dir}")
