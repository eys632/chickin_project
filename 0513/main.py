import os
import time
from datetime import datetime
from video_splitter import split_video_3x3
from yolo_model_loader import run_yolo_inference_all_parts
from motion_tracker import MotionTracker
from video_merger import merge_and_draw
from csv_logger import CSVLogger
import cv2
import sys

def main():
    # 시작 메시지 출력
    print("\n" + "="*80)
    print("움직이지 않는 닭 감지 및 추적 시스템 시작")
    print("="*80 + "\n")
    
    # 기본 설정
    BASE_DIR = "/home/a202192020/객체탐지 프로젝트(송교수님)/0513"
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_DIR = os.path.join(BASE_DIR, f"dead_check_{run_id}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 원래 파일 경로로 복원 (777.mp4)
    video_path = "/home/nas/data/YeonSeung/777.mp4"
    model_path = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt"

    print(f"결과 저장 폴더: {SAVE_DIR}")
    print(f"입력 영상: {video_path}")
    print(f"YOLO 모델: {model_path}")
    
    # 전체 진행률 계산을 위한 변수
    # 각 단계별 가중치 (합이 100이 되도록)
    split_weight = 10      # 영상 분할: 10%
    inference_weight = 40  # YOLO 추론: 40%
    tracking_weight = 50   # 움직임 추적 및 영상 병합: 50%
    
    # 현재까지의 진행률 계산
    current_progress = 0
    
    # 진행 막대 출력 함수
    def print_progress(percent, prefix='', suffix='', length=50):
        filled_length = int(length * percent / 100)
        bar = '=' * filled_length + ' ' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} [{bar}] {percent:.1f}% {suffix}')
        sys.stdout.flush()
    
    # 영상 정보 가져오기
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # ========================================================================
    # 1단계: 영상 9분할 (전체 진행률의 10%)
    # ========================================================================
    print("\n" + "-"*80)
    print("1. 영상 9분할 시작 (전체 진행률의 10%)")
    print("-"*80)
    
    split_start = time.time()
    
    # 분할 진행 상황을 추적하기 위한 콜백 함수
    def split_progress_callback(current_frame):
        # 분할 단계 진행률 (0-100%)
        split_progress = min(100, current_frame / total_frames * 100)
        
        # 전체 진행률에 반영 (0-10%)
        current_step_contribution = split_progress * split_weight / 100
        global_progress = current_progress + current_step_contribution
        
        # 진행 막대 출력
        print_progress(
            global_progress, 
            prefix=f'전체 진행률', 
            suffix=f'[영상 분할 {split_progress:.1f}%]'
        )
    
    # 영상 분할 함수 호출
    part_paths = split_video_3x3(video_path, SAVE_DIR, split_progress_callback)
    
    # 분할 완료 후 전체 진행률 업데이트
    current_progress += split_weight
    print()  # 줄바꿈
    
    split_end = time.time()
    print(f"분할 완료 - {len(part_paths)}개 생성됨")
    print(f"소요 시간: {split_end - split_start:.2f}초")
    
    # ========================================================================
    # 2단계: YOLO 추론 - 모든 닭 감지 (전체 진행률의 40%)
    # ========================================================================
    print("\n" + "-"*80)
    print("2. YOLO 추론 시작 - 모든 닭 감지 (전체 진행률의 40%)")
    print("-"*80)
    
    inference_start = time.time()
    
    # 분할 영상 총 개수
    total_parts = len(part_paths)
    
    # 모든 분할 영상의 총 프레임 수 계산
    inference_total_frames = 0
    for path in part_paths:
        cap = cv2.VideoCapture(path)
        inference_total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    
    inference_processed_frames = 0
    
    # YOLO 추론 진행 상황을 추적하기 위한 콜백 함수
    def yolo_inference_callback(part_idx, part_name, frame_idx, part_total_frames):
        nonlocal inference_processed_frames
        
        # 지금까지 처리한 모든 프레임 수 계산
        processed_so_far = 0
        for i in range(part_idx):
            cap = cv2.VideoCapture(part_paths[i])
            processed_so_far += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        inference_processed_frames = processed_so_far + frame_idx
        
        # 추론 단계 진행률 (0-100%)
        inference_progress = min(100, inference_processed_frames / inference_total_frames * 100)
        
        # 전체 진행률에 반영
        current_step_contribution = inference_progress * inference_weight / 100
        global_progress = current_progress + current_step_contribution
        
        # 진행 막대 출력 (100프레임마다 업데이트하여 출력 줄이기)
        if frame_idx % 100 == 0 or frame_idx == 0 or frame_idx == part_total_frames - 1:
            print_progress(
                global_progress, 
                prefix=f'전체 진행률', 
                suffix=f'[YOLO 추론 {inference_progress:.1f}%]'
            )
            
            # 현재 처리 중인 영상 및 프레임 정보 출력
            print(f"\n처리 중: [{part_idx+1}/{total_parts}] {part_name} - 프레임 {frame_idx+1}/{part_total_frames}" +
                 f" ({(frame_idx+1)/part_total_frames*100:.1f}%)")
    
    # YOLO 추론 함수 호출
    detections_dict, frames_dict = run_yolo_inference_all_parts(part_paths, model_path, yolo_inference_callback)
    
    # 추론 완료 후 전체 진행률 업데이트
    current_progress += inference_weight
    print()  # 줄바꿈
    
    inference_end = time.time()
    print(f"\nYOLO 추론 완료")
    print(f"소요 시간: {inference_end - inference_start:.2f}초")
    
    # ========================================================================
    # 3단계: 움직이지 않는 닭만 추적 및 영상 처리 (전체 진행률의 50%)
    # ========================================================================
    print("\n" + "-"*80)
    print("3. 움직이지 않는 닭만 추적 및 영상 처리 시작 (전체 진행률의 50%)")
    print("-"*80)
    
    # 수정된 MotionTracker 사용 - ID 매핑 문제 해결
    tracker = MotionTracker()
    csv_logger = CSVLogger(SAVE_DIR)
    
    # 처리된 분할 영상 저장을 위한 VideoWriter 객체 생성
    processed_writers = {}
    
    # 샘플 프레임으로 크기 확인
    first_part = next(iter(frames_dict.values()))[0]
    part_h, part_w, _ = first_part.shape
    
    for i in range(3):
        for j in range(3):
            part_name = f"part_{i}_{j}"
            processed_path = os.path.join(SAVE_DIR, f"{part_name}_processed.mp4")
            processed_writer = cv2.VideoWriter(
                processed_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                30, 
                (part_w, part_h)
            )
            processed_writers[part_name] = processed_writer
    
    # 병합 영상을 위한 VideoWriter
    merged_out_path = os.path.join(SAVE_DIR, "merged_dead_chickens.mp4")
    merged_writer = cv2.VideoWriter(
        merged_out_path, 
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30, 
        (part_w*3, part_h*3)
    )

    # 모든 분할 영상의 프레임 수
    total_frames = len(next(iter(frames_dict.values())))
    print(f"총 프레임 수: {total_frames}")
    
    # 추적 통계
    total_static_chickens = 0
    
    # 프레임별 처리
    processing_start = time.time()
    
    # 모션 추적 및 정지 닭 감지를 위한 데이터 수집 단계
    print("\n움직이지 않는 닭 감지를 위한 모션 분석 중...")
    
    # 모든 프레임을 모든 닭에 대해 1차 분석 - 목적: ID 매핑 정보 수집 및 움직이지 않는 닭 식별
    motion_frames_count = total_frames
    for frame_idx in range(motion_frames_count):
        # 모션 분석 진행률 계산 및 표시 (100%까지 표시되도록 수정)
        motion_progress = min(100, (frame_idx + 1) / motion_frames_count * 100)
        
        if frame_idx % 10 == 0 or frame_idx == 0 or frame_idx == motion_frames_count - 1:
            print_progress(
                motion_progress,  # 100%까지 표시되도록 수정
                prefix=f'모션 분석', 
                suffix=f'[{frame_idx+1}/{motion_frames_count} 프레임]'
            )
        
        # 현재 프레임의 모든 분할 영상 처리
        for part_name, det_list in detections_dict.items():
            if frame_idx >= len(det_list):
                continue
                
            boxes = det_list[frame_idx]
            cxcy_list = [((x1+x2)/2, (y1+y2)/2) for x1,y1,x2,y2 in boxes]
            
            # 객체 위치 업데이트 - 분할 영역 정보 포함
            tracker.update_positions(frame_idx, cxcy_list, part_name)
    
    # 모션 분석이 끝나면 줄바꿈 및 상태 출력
    print()  # 줄바꿈
    
    # ID 매핑 정보 출력
    if tracker.id_mapping:
        print(f"ID 매핑 정보 생성됨: {len(tracker.id_mapping)}개")
        print(f"예시 매핑: {list(tracker.id_mapping.items())[:5]}" + 
              ("..." if len(tracker.id_mapping) > 5 else ""))
    else:
        print("경고: ID 매핑 정보가 생성되지 않았습니다.")
    
    print("모션 분석 완료, 움직이지 않는 닭 식별됨")
    
    # 모션 분석 이후 정지한 닭만 처리
    # 전체 비디오에 대한 움직이지 않는 닭 목록 생성
    all_static_chickens = set()
    for frame_idx in range(total_frames):
        dead_chickens = tracker.get_dead_chickens(frame_idx)
        for id_ in dead_chickens.keys():
            all_static_chickens.add(id_)
    
    print(f"총 {len(all_static_chickens)}마리의 움직이지 않는 닭이 감지됨: {list(all_static_chickens)[:10]}" + 
          ("..." if len(all_static_chickens) > 10 else ""))
    
    # 움직이지 않는 닭만 처리하는 단계
    print("\n움직이지 않는 닭만 처리 중...")
    
    # 전체 처리 프레임 수
    processing_frames_count = total_frames
    
    for frame_idx in range(processing_frames_count):
        # 추적 진행률 계산 (0-100%)
        tracking_progress = min(100, (frame_idx + 1) / processing_frames_count * 100)
        
        # 전체 진행률에 반영 (추적은 전체의 50%)
        # 추적 단계의 진행률을 전체 진행률의 50%에 반영
        global_progress = current_progress + (tracking_progress * tracking_weight / 100)
        
        # 주기적으로 진행률 표시
        if frame_idx % 10 == 0 or frame_idx == 0 or frame_idx == processing_frames_count - 1:
            print_progress(
                global_progress, 
                prefix=f'전체 진행률', 
                suffix=f'[정지 닭 처리 {tracking_progress:.1f}%]'
            )
            
        # 세부 정보는 더 긴 간격으로 표시
        if frame_idx % 100 == 0:
            elapsed = time.time() - processing_start
            remaining = (elapsed / (frame_idx + 1)) * (processing_frames_count - frame_idx - 1) if frame_idx > 0 else 0
            
            print(f"\n프레임 {frame_idx}/{processing_frames_count} ({frame_idx/processing_frames_count*100:.1f}%) | " +
                 f"경과: {elapsed:.1f}초 | 남은 시간: {remaining:.1f}초")
        
        # 움직이지 않는 닭 감지
        dead_chickens = tracker.get_dead_chickens(frame_idx)
        
        # CSV 로깅 - 움직이지 않는 닭만
        csv_logger.log_dead(frame_idx, dead_chickens)
        
        # 통계 업데이트
        current_static_chickens = len(dead_chickens)
        total_static_chickens = max(total_static_chickens, current_static_chickens)
        
        # 각 분할 영상에 박스 그리기 및 저장
        current_frames = {}
        
        for part_name, frame_list in frames_dict.items():
            if frame_idx >= len(frame_list):
                continue
                
            # 원본 프레임 복사
            frame = frame_list[frame_idx].copy()
            current_frames[part_name] = frame
            
            # 이 분할 영역에 있는 정지한 닭만 필터링
            part_dead_chickens = {}
            for id_, chicken_info in dead_chickens.items():
                cx, cy, chicken_part_name = chicken_info
                if chicken_part_name == part_name:
                    part_dead_chickens[id_] = chicken_info
            
            # 이 분할 영역에 있는 정지한 닭에 대해 박스 및 ID 표시
            for id_, (cx, cy, _) in part_dead_chickens.items():
                abs_cx, abs_cy = int(cx), int(cy)
                x1, y1 = max(0, abs_cx - 20), max(0, abs_cy - 20)
                x2, y2 = min(part_w - 1, abs_cx + 20), min(part_h - 1, abs_cy + 20)
                
                # 빨간색 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # ID 표시
                label = f"ID {id_}"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - h_text - 5), (x1 + w_text + 5, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 분할 영역 이름 표시
            cv2.putText(frame, part_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 처리된 분할 영상 저장
            processed_writers[part_name].write(frame)
        
        # 병합된 영상 생성 및 저장 - 움직이지 않는 닭만 표시
        merge_and_draw(current_frames, dead_chickens, frame_idx, merged_writer)
    
    print()  # 줄바꿈
    
    # 모든 VideoWriter 닫기
    for writer in processed_writers.values():
        writer.release()
    merged_writer.release()
    
    # ID 매핑 정보 저장
    csv_logger.log_id_mapping(tracker.id_mapping)
    
    processing_end = time.time()
    
    # ========================================================================
    # 4단계: 결과 요약 및 완료
    # ========================================================================
    print("\n" + "-"*80)
    print("4. 처리 완료 - 결과 요약")
    print("-"*80)
    
    total_time = processing_end - split_start
    print(f"총 처리 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
    print(f"총 처리 프레임: {total_frames}프레임")
    print(f"프레임당 평균 처리 시간: {total_time/total_frames:.3f}초")
    print(f"감지된 정지 닭 최대 개수: {total_static_chickens}마리")
    print(f"총 정지 닭 ID 수: {len(all_static_chickens)}개")
    print(f"ID 매핑 수: {len(tracker.id_mapping)}개")
    
    # 결과 파일 확인
    print("\n결과 파일 크기 확인:")
    
    # 병합 영상 확인
    if os.path.exists(merged_out_path):
        size_mb = os.path.getsize(merged_out_path) / (1024 * 1024)
        print(f"병합 영상: {merged_out_path} - {size_mb:.2f} MB")
    else:
        print(f"병합 영상: {merged_out_path} - 파일이 존재하지 않음")
    
    # 처리된 분할 영상 확인
    for part_name in processed_writers.keys():
        processed_path = os.path.join(SAVE_DIR, f"{part_name}_processed.mp4")
        if os.path.exists(processed_path):
            size_mb = os.path.getsize(processed_path) / (1024 * 1024)
            print(f"처리된 분할 영상: {processed_path} - {size_mb:.2f} MB")
        else:
            print(f"처리된 분할 영상: {processed_path} - 파일이 존재하지 않음")
    
    # ID 매핑 파일 확인
    id_mapping_path = os.path.join(SAVE_DIR, "id_mapping.csv")
    if os.path.exists(id_mapping_path):
        size_kb = os.path.getsize(id_mapping_path) / 1024
        print(f"ID 매핑 CSV: {id_mapping_path} - {size_kb:.2f} KB")
        
        # 내용 확인
        import csv
        mapping_count = 0
        with open(id_mapping_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            mapping_count = sum(1 for _ in reader)
        print(f"  - ID 매핑 개수: {mapping_count}개")
    else:
        print(f"ID 매핑 CSV: {id_mapping_path} - 파일이 존재하지 않음")
    
    print("\n" + "="*80)
    print("결과 파일:")
    print(f"   병합 영상: {merged_out_path}")
    print("   처리된 분할 영상:")
    for part_name in processed_writers.keys():
        print(f"      - {part_name}_processed.mp4")
    print(f"   닭 위치 CSV: {os.path.join(SAVE_DIR, 'dead_chickens.csv')}")
    print(f"   ID 매핑 CSV: {os.path.join(SAVE_DIR, 'id_mapping.csv')}")
    print("="*80 + "\n")
    
    print("모든 과정 완료! 폴더에서 결과를 확인하세요.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"오류 발생: {e}")
        print(traceback.format_exc())