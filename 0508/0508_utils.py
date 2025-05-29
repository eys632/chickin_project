# main_detector.py - 비디오 세그먼트 처리 모듈
import os
import cv2
import numpy as np
import torch
import time
from datetime import datetime
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def process_video_segment(video_path, model_path, save_dir, start_frame=0, end_frame=None, device=0):
    """
    비디오의 특정 프레임 범위를 처리하는 함수
    
    Args:
        video_path (str): 비디오 파일 경로
        model_path (str): YOLO 모델 경로
        save_dir (str): 결과 저장 디렉토리
        start_frame (int): 시작 프레임 (0부터 시작)
        end_frame (int): 끝 프레임 (None이면 비디오 끝까지)
        device (int): 사용할 GPU 장치 번호
    """
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 로그 파일 설정
    log_file = os.path.join(save_dir, "detection_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'chicken_id', 'x1', 'y1', 'x2', 'y2', 'static_ratio', 'iou', 'is_static'])
    
    # 비디오 캡쳐 객체 초기화
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {video_path}")
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 프레임 범위 설정
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    
    # 결과 비디오 설정
    output_video = os.path.join(save_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # YOLO 모델 로드
    print(f"[GPU {device}] YOLO 모델 로딩 중: {model_path}")
    model = YOLO(model_path)
    
    # 배경 서브트랙터 초기화 (MOG2)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,          # 배경 모델 히스토리 길이
        varThreshold=16,      # 전경/배경 구분 임계값
        detectShadows=False   # 그림자 감지 비활성화 (성능 향상)
    )
    
    # 파라미터 설정
    # 1. 배경 모델 학습률
    learning_rate = 0.01
    
    # 2. 정적 픽셀 관련 파라미터
    static_threshold = 8      # 픽셀 변화 임계값 (Δ8)
    min_static_area = 200     # 최소 정적 영역 크기 (픽셀)
    
    # 3. YOLO 파라미터
    yolo_conf = 0.3           # 객체 탐지 신뢰도 임계값
    yolo_nms = 0.5            # NMS 임계값
    
    # 4. IoU 정적 임계값
    iou_threshold = 0.9       # IoU 정적 판정 임계값
    
    # 5. 마스크 비율 임계값
    mask_ratio_threshold = 0.8  # 바운딩 박스 내 정적 픽셀 비율 임계값
    
    # 트래킹을 위한 딕셔너리 초기화
    prev_boxes = {}           # 이전 프레임의 바운딩 박스
    chicken_tracks = {}       # 닭 트래킹 정보
    static_counter = defaultdict(int)  # 정적 상태 카운터
    static_status = {}        # 닭의 정적 상태
    
    # 트래킹 ID 관리 (시작 프레임별로 다르게 설정)
    next_id = 1000 * (device + 1)  # 장치 번호별로 다른 ID 범위 사용
    
    # IOU 계산 함수
    def calculate_iou(box1, box2):
        """두 바운딩 박스의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교차 영역 계산
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # 교차 영역 없음
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 각 박스 영역 계산
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU 계산
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return iou
    
    # 객체 ID 할당 함수
    def assign_ids(current_boxes):
        """현재 바운딩 박스에 ID 할당"""
        nonlocal next_id, prev_boxes
        assigned_boxes = {}
        
        # 이전 박스가 없으면 모든 박스에 새 ID 할당
        if not prev_boxes:
            for box in current_boxes:
                assigned_boxes[next_id] = box
                next_id += 1
            return assigned_boxes
        
        # 현재 박스와 이전 박스 간의 IoU 계산
        iou_matrix = {}
        for prev_id, prev_box in prev_boxes.items():
            for curr_box in current_boxes:
                if curr_box not in iou_matrix:
                    iou_matrix[curr_box] = {}
                iou_matrix[curr_box][prev_id] = calculate_iou(prev_box, curr_box)
        
        # 이미 할당된 박스 추적
        assigned_curr_boxes = set()
        assigned_prev_ids = set()
        
        # IoU가 가장 높은 상자부터 ID 할당
        while len(assigned_curr_boxes) < len(current_boxes):
            max_iou = 0
            best_match = None
            
            for curr_box in current_boxes:
                if curr_box in assigned_curr_boxes:
                    continue
                    
                for prev_id, iou in iou_matrix.get(curr_box, {}).items():
                    if prev_id in assigned_prev_ids:
                        continue
                        
                    if iou > max_iou:
                        max_iou = iou
                        best_match = (curr_box, prev_id)
            
            # 더 이상 매칭할 박스가 없거나 IoU가 임계값보다 낮으면 새 ID 할당
            if max_iou < 0.3 or best_match is None:  # 낮은 IOU 임계값 사용
                for curr_box in current_boxes:
                    if curr_box not in assigned_curr_boxes:
                        assigned_boxes[next_id] = curr_box
                        assigned_curr_boxes.add(curr_box)
                        next_id += 1
                break
            else:
                curr_box, prev_id = best_match
                assigned_boxes[prev_id] = curr_box
                assigned_curr_boxes.add(curr_box)
                assigned_prev_ids.add(prev_id)
        
        return assigned_boxes
    
    # 첫 번째 프레임으로 이동
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 메인 처리 루프
    frame_num = start_frame
    prev_gray = None
    start_time = time.time()
    segment_frames = end_frame - start_frame
    
    print(f"[GPU {device}] 비디오 세그먼트 처리 시작: {start_frame} ~ {end_frame-1} ({segment_frames} 프레임)")
    
    try:
        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_num % 100 == 0:
                elapsed = time.time() - start_time
                fps_processed = (frame_num - start_frame + 1) / elapsed if elapsed > 0 else 0
                eta = (end_frame - frame_num) / fps_processed if fps_processed > 0 else 0
                print(f"[GPU {device}] 처리 중: {frame_num - start_frame + 1}/{segment_frames} 프레임 ({fps_processed:.2f} FPS, ETA: {eta/60:.1f}분)")
            
            # 작업용 복사본 생성
            display_frame = frame.copy()
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1단계: 배경 모델 업데이트 및 전경 마스크 추출
            fg_mask = bg_subtractor.apply(frame, learningRate=learning_rate)
            
            # 2단계: 정적 픽셀 마스크 추출
            if prev_gray is not None:
                # 프레임 간 절대 차이 계산
                frame_diff = cv2.absdiff(gray, prev_gray)
                
                # 임계값 적용하여 정적 마스크 생성 (차이가 작은 픽셀 = 정적)
                static_mask = cv2.threshold(frame_diff, static_threshold, 255, cv2.THRESH_BINARY_INV)[1]
                
                # 노이즈 제거를 위한 모폴로지 연산
                kernel = np.ones((5, 5), np.uint8)
                static_mask = cv2.erode(static_mask, kernel, iterations=1)
                static_mask = cv2.dilate(static_mask, kernel, iterations=2)
                
                # 배경 마스크 (fg_mask의 반전)와 교차하여 실제 정적 영역 얻기
                bg_mask = cv2.bitwise_not(fg_mask)
                static_mask = cv2.bitwise_and(static_mask, bg_mask)
                
                # 작은 정적 영역 제거
                contours, _ = cv2.findContours(static_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                static_mask_filtered = np.zeros_like(static_mask)
                for contour in contours:
                    if cv2.contourArea(contour) > min_static_area:
                        cv2.drawContours(static_mask_filtered, [contour], -1, 255, -1)
                
                static_mask = static_mask_filtered
                
                # 3단계: YOLO로 닭 탐지
                results = model(frame, conf=yolo_conf, iou=yolo_nms, device=device)
                
                # 현재 바운딩 박스 추출
                current_boxes = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        current_boxes.append((x1, y1, x2, y2))
                
                # 박스에 ID 할당
                id_boxes = assign_ids(current_boxes)
                
                # 4단계 & 5단계: IoU 정적 여부 판정 및 정적 픽셀 비율 계산
                for chicken_id, box in id_boxes.items():
                    x1, y1, x2, y2 = box
                    
                    # 바운딩 박스 내부의 정적 마스크 추출
                    box_mask = np.zeros_like(static_mask)
                    box_mask[y1:y2, x1:x2] = 255
                    box_static_mask = cv2.bitwise_and(static_mask, box_mask)
                    
                    # 박스 내 정적 픽셀 비율 계산
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area == 0:  # 0으로 나누기 방지
                        static_ratio = 0
                    else:
                        static_pixels = cv2.countNonZero(box_static_mask)
                        static_ratio = static_pixels / box_area
                    
                    # 이전 위치와의 IoU 계산
                    iou = 0
                    if chicken_id in prev_boxes:
                        iou = calculate_iou(prev_boxes[chicken_id], box)
                    
                    # 정적 여부 판정 (IoU와 정적 픽셀 비율 모두 고려)
                    is_static = False
                    if iou > iou_threshold and static_ratio > mask_ratio_threshold:
                        static_counter[chicken_id] += 1
                        # 10프레임(약 0.3초) 이상 정적이면 정적으로 판정
                        if static_counter[chicken_id] >= 10:
                            is_static = True
                    else:
                        # 카운터 감소 (최소 0)
                        static_counter[chicken_id] = max(0, static_counter[chicken_id] - 1)
                    
                    static_status[chicken_id] = is_static
                    
                    # 결과 기록
                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([frame_num, chicken_id, x1, y1, x2, y2, 
                                         static_ratio, iou, int(is_static)])
                    
                    # 바운딩 박스 그리기
                    color = (0, 0, 255) if is_static else (0, 255, 0)  # 정적=빨강, 움직임=초록
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # ID와 정적 비율 표시
                    text = f"ID:{chicken_id} S:{static_ratio:.2f}"
                    cv2.putText(display_frame, text, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 정적 마스크 시각화 (반투명 효과)
                static_mask_color = cv2.cvtColor(static_mask, cv2.COLOR_GRAY2BGR)
                static_mask_color[np.where((static_mask_color == [255, 255, 255]).all(axis=2))] = [0, 0, 180]  # 약간 빨간색
                alpha = 0.3
                display_frame = cv2.addWeighted(static_mask_color, alpha, display_frame, 1 - alpha, 0)
                
                # 상태 정보 표시
                status_text = f"프레임: {frame_num} | GPU: {device} | 정적 닭: {sum(static_status.values())}/{len(static_status)}"
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 결과 저장
                out.write(display_frame)
                
                # 30프레임마다 이미지 저장
                if frame_num % 30 == 0:
                    img_path = os.path.join(save_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(img_path, display_frame)
                
                # 현재 박스를 이전 박스로 업데이트
                prev_boxes = id_boxes.copy()
            
            # 현재 프레임을 이전 프레임으로 저장
            prev_gray = gray.copy()
            
            # 프레임 카운터 증가
            frame_num += 1
    
    except KeyboardInterrupt:
        print(f"[GPU {device}] 사용자에 의해 처리가 중단되었습니다.")
    finally:
        # 자원 해제
        cap.release()
        out.release()
        
        # 처리 시간 계산
        total_time = time.time() - start_time
        processed_frames = frame_num - start_frame
        print(f"[GPU {device}] 처리 완료: {processed_frames} 프레임, 총 소요 시간: {total_time:.2f}초 ({processed_frames/total_time:.2f} FPS)")
        
        # 결과 분석 및 보고서 생성
        report_path = os.path.join(save_dir, "report.txt")
        with open(report_path, 'w') as f:
            f.write(f"닭 움직임 분석 보고서 (GPU {device})\n")
            f.write(f"=======================\n")
            f.write(f"비디오: {video_path}\n")
            f.write(f"세그먼트: {start_frame} ~ {frame_num-1}\n")
            f.write(f"처리 시간: {total_time:.2f}초 ({processed_frames/total_time:.2f} FPS)\n")
            f.write(f"총 감지된 닭 수: {len(static_status)}\n")
            f.write(f"정적 상태로 판정된 닭 수: {sum(static_status.values())}\n\n")
            
            f.write("정적 닭 ID 목록:\n")
            for chicken_id, is_static in static_status.items():
                if is_static:
                    f.write(f"- 닭 ID: {chicken_id}\n")

if __name__ == "__main__":
    # 단독 실행시 테스트 코드
    import argparse
    
    parser = argparse.ArgumentParser(description='닭 움직임 감지 비디오 세그먼트 처리')
    parser.add_argument('--video', required=True, help='비디오 파일 경로')
    parser.add_argument('--model', required=True, help='YOLO 모델 경로')
    parser.add_argument('--output', required=True, help='결과 저장 경로')
    parser.add_argument('--start', type=int, default=0, help='시작 프레임')
    parser.add_argument('--end', type=int, default=None, help='끝 프레임')
    parser.add_argument('--device', type=int, default=0, help='GPU 장치 번호')
    
    args = parser.parse_args()
    
    process_video_segment(
        args.video, 
        args.model, 
        args.output, 
        args.start, 
        args.end, 
        args.device
    )