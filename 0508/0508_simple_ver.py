import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

# =========================
# 설정값
# =========================
VIDEO_PATH = "/home/nas/data/YeonSeung/chicken_7min.mp4"
MODEL_PATH = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt"
BASE_DIR = "/home/a202192020/객체탐지 프로젝트(송교수님)/chicken_proj"
FRAME_INTERVAL_SEC = 60
FRAME_BATCH = 100
TOLERANCE_RADIUS = 3  # 중앙점 기준 반지름 (지름 = 6픽셀 허용)

# =========================
# 중앙점 거리 비교 함수
# =========================
def is_within_radius(p1, p2, r=TOLERANCE_RADIUS):
    """두 점이 지정된 반경 내에 있는지 확인"""
    return abs(p1[0] - p2[0]) <= r and abs(p1[1] - p2[1]) <= r

# =========================
# 메인 함수
# =========================
def detect_static_chickens():
    """닭 정적 감지 메인 함수"""
    
    # GPU 설정 - 0번 GPU만 사용
    device = 0
    torch.cuda.set_device(device)
    print(f"GPU {device} 사용")
    
    # 결과 저장 폴더
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_DIR = os.path.join(BASE_DIR, f"dead_check_{run_id}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 비디오 캡쳐 객체 초기화
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {VIDEO_PATH}")
    
    # 비디오 속성 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    
    # YOLO 모델 로드
    print(f"YOLO 모델 로딩 중: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # 4분할
    regions = {
        "topleft": (0, width // 2, 0, height // 2),
        "topright": (width // 2, width, 0, height // 2),
        "bottomleft": (0, width // 2, height // 2, height),
        "bottomright": (width // 2, width, height // 2, height)
    }
    
    # 전체 결과를 저장할 DataFrame
    all_tracking_logs = []
    
    # 각 지역별 처리
    for region_name, (x1, x2, y1, y2) in regions.items():
        print(f"\n--- [{region_name.upper()}] 분석 시작 ---")
        region_dir = os.path.join(SAVE_DIR, region_name)
        os.makedirs(region_dir, exist_ok=True)
        
        # 작업용 변수 초기화
        tracking_log = []
        dead_candidate_dict = defaultdict(list)
        global_id_counter = 1000  # 독립적인 ID 범위
        
        # 분석 시작
        for t in range(0, duration - 1, FRAME_INTERVAL_SEC):
            frame_centers = []
            
            # 일정 시간 간격의 연속 프레임에서 중앙점 추출
            for i in range(FRAME_BATCH):
                frame_pos = int((t + i / FRAME_BATCH) * fps)
                if frame_pos >= total_frames:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # 지역 추출
                sub = frame[y1:y2, x1:x2]
                
                # YOLO 감지
                results = model.predict(sub, verbose=False, device=device)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                
                # 중앙점 계산
                frame_center_set = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = box[:4]
                    cx = int((x_min + x_max) / 2)
                    cy = int((y_min + y_max) / 2)
                    frame_center_set.append((cx, cy))
                
                frame_centers.append(frame_center_set)
            
            # 프레임 중심점이 없으면 다음 간격으로 이동
            if not frame_centers or not frame_centers[0]:
                continue
                
            # "거의 안 움직이는" 중심점 찾기
            stable_centers = []
            base_frame = frame_centers[0]
            for base_c in base_frame:
                # 모든 프레임에서 이 중심점 근처에 객체가 있는지 확인
                is_stable = True
                for centers in frame_centers[1:]:
                    if not centers:  # 빈 프레임은 건너뜀
                        continue
                    if not any(is_within_radius(base_c, other_c) for other_c in centers):
                        is_stable = False
                        break
                
                if is_stable:
                    stable_centers.append(base_c)
            
            # 기존 ID와 매칭 or 신규 ID 부여
            matched_ids = set()
            for center in stable_centers:
                matched = False
                for id_, history in dead_candidate_dict.items():
                    if not history:  # 빈 히스토리는 건너뜀
                        continue
                    last_frame, last_center = history[-1]
                    if abs(t * fps - last_frame) >= FRAME_INTERVAL_SEC * fps:
                        if is_within_radius(center, last_center):
                            for offset in range(FRAME_BATCH):
                                frame_index = int(round((t + offset / FRAME_BATCH) * fps))
                                if frame_index < total_frames:
                                    dead_candidate_dict[id_].append((frame_index, center))
                            matched = True
                            matched_ids.add(id_)
                            break
                
                if not matched:
                    for offset in range(FRAME_BATCH):
                        frame_index = int(round((t + offset / FRAME_BATCH) * fps))
                        if frame_index < total_frames:
                            dead_candidate_dict[global_id_counter].append((frame_index, center))
                    matched_ids.add(global_id_counter)
                    global_id_counter += 1
            
            # 움직인 ID는 제거
            for id_ in list(dead_candidate_dict.keys()):
                if id_ not in matched_ids:
                    del dead_candidate_dict[id_]
            
            print(f"[{region_name}] {t//60}분 분석 완료 | 현재 죽은 후보 ID: {list(dead_candidate_dict.keys())}")
        
        # 로그 저장
        for id_, hist in dead_candidate_dict.items():
            for fidx, pt in hist:
                tracking_data = {
                    "id": id_,
                    "frame": fidx,
                    "x": pt[0] + x1,  # 전체 프레임 좌표로 변환
                    "y": pt[1] + y1,  # 전체 프레임 좌표로 변환
                    "region": region_name
                }
                tracking_log.append(tracking_data)
                all_tracking_logs.append(tracking_data)
        
        # 지역별 결과 저장
        if tracking_log:
            df = pd.DataFrame(tracking_log)
            df.to_csv(os.path.join(region_dir, f"{region_name}_dead_candidates.csv"), index=False)
            print(f"[{region_name}] 분석 완료: {len(df['id'].unique())}마리 죽은 닭 후보 발견")
        else:
            # 빈 데이터프레임 저장
            pd.DataFrame(columns=["id", "frame", "x", "y", "region"]).to_csv(
                os.path.join(region_dir, f"{region_name}_dead_candidates.csv"), index=False)
            print(f"[{region_name}] 분석 완료: 죽은 닭 후보가 없습니다.")
        
        # 시각화 (마지막 프레임에 죽은 닭 표시)
        if dead_candidate_dict:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()
            if ret:
                # 지역 강조
                cv2.rectangle(last_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 죽은 닭 마킹
                for id_, hist in dead_candidate_dict.items():
                    if not hist:  # 빈 히스토리는 건너뜀
                        continue
                    _, (cx, cy) = hist[-1]
                    # 좌표를 전체 프레임 기준으로 변환
                    cx += x1
                    cy += y1
                    cv2.circle(last_frame, (cx, cy), 10, (0, 0, 255), -1)  # 빨간 원
                    cv2.putText(last_frame, f"ID: {id_}", (cx - 20, cy - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # 이미지 저장
                cv2.imwrite(os.path.join(region_dir, f"{region_name}_final_result.jpg"), last_frame)
    
    # 모든 지역 결과 병합
    if all_tracking_logs:
        merged_df = pd.DataFrame(all_tracking_logs)
        merged_df.to_csv(os.path.join(SAVE_DIR, "all_dead_candidates.csv"), index=False)
        
        # 통합 시각화
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if ret:
            # 지역 구분선 그리기
            cv2.line(last_frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
            cv2.line(last_frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
            
            # 각 지역의 죽은 닭 표시
            latest_positions = {}
            for row in all_tracking_logs:
                if row['id'] not in latest_positions or latest_positions[row['id']]['frame'] < row['frame']:
                    latest_positions[row['id']] = {
                        'frame': row['frame'],
                        'x': row['x'],
                        'y': row['y'],
                        'region': row['region']
                    }
            
            # 죽은 닭 표시
            for id_, info in latest_positions.items():
                cx, cy = int(info['x']), int(info['y'])
                cv2.circle(last_frame, (cx, cy), 15, (0, 0, 255), -1)  # 빨간 원
                cv2.putText(last_frame, f"ID: {id_}", (cx - 20, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 요약 정보 표시
            dead_count = len(latest_positions)
            region_counts = defaultdict(int)
            for info in latest_positions.values():
                region_counts[info['region']] += 1
            
            info_text = f"총 죽은 닭 수: {dead_count}"
            cv2.putText(last_frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            y_pos = 70
            for region, count in region_counts.items():
                region_text = f"{region}: {count}마리"
                cv2.putText(last_frame, region_text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_pos += 30
            
            # 이미지 저장
            cv2.imwrite(os.path.join(SAVE_DIR, "final_combined_result.jpg"), last_frame)
            print(f"통합 결과 이미지 저장: {os.path.join(SAVE_DIR, 'final_combined_result.jpg')}")
    
    cap.release()
    print(f"\n분석 완료, 결과는 {SAVE_DIR} 에 저장됨.")
    return SAVE_DIR

# =========================
# 프로그램 실행
# =========================
if __name__ == "__main__":
    result_dir = detect_static_chickens()
    print(f"최종 결과 디렉터리: {result_dir}")