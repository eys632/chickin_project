# 프레임 이미지도 함께 반환하도록 수정
from ultralytics import YOLO
import cv2
import time

def run_yolo_inference_all_parts(video_paths, model_path, progress_callback=None):
    """
    분할된 영상에 YOLO 모델을 적용하여 객체 감지 수행
    
    Args:
        video_paths: 분할된 영상 경로 리스트
        model_path: YOLO 모델 경로
        progress_callback: 진행 상황을 외부로 전달할 콜백 함수
                          함수 형태: progress_callback(part_idx, part_name, frame_idx, total_frames)
    """
    model = YOLO(model_path)
    detections = {}
    frames = {}
    
    # 전체 진행 상황 계산을 위해 각 영상의 프레임 수 미리 확인
    video_frames = {}
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_frames[path] = frame_count

    for part_idx, path in enumerate(video_paths):
        part_name = path.split("/")[-1].replace(".mp4", "")
        
        # 영상 정보 출력
        print(f"\n[{part_idx+1}/{len(video_paths)}] 영상 추론 시작: {part_name}")
        print(f"  파일: {path}")
        print(f"  프레임 수: {video_frames[path]}")
        
        # 추론 시작 시간
        start_time = time.time()
        
        cap = cv2.VideoCapture(path)
        
        # YOLO 모델로 예측 - 스트리밍 모드와 GPU 사용
        results = model.predict(path, stream=True, device=0)

        det_list = []
        frame_list = []
        
        # 현재 프레임 인덱스 추적
        frame_idx = 0
        
        for result in results:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 감지된 객체 정보 수집
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append((x1, y1, x2, y2))
                    
            det_list.append(boxes)
            frame_list.append(frame)
            
            # 콜백 함수가 제공된 경우 진행 상황 전달
            if progress_callback:
                progress_callback(part_idx, part_name, frame_idx, video_frames[path])
            
            # 진행 상황 출력 (100프레임마다 또는 첫/마지막 프레임)
            if frame_idx % 100 == 0 or frame_idx == 0 or frame_idx == video_frames[path] - 1:
                elapsed_time = time.time() - start_time
                fps = (frame_idx + 1) / elapsed_time if elapsed_time > 0 else 0
                
                # 감지된 객체 수와 처리 속도 출력 
                print(f"  프레임 {frame_idx+1}/{video_frames[path]} " +
                      f"({(frame_idx+1)/video_frames[path]*100:.1f}%) | " +
                      f"박스 수: {len(boxes)} | " +
                      f"처리 속도: {fps:.2f} fps")
            
            frame_idx += 1
            
        cap.release()
        detections[part_name] = det_list
        frames[part_name] = frame_list
        
        # 영상 처리 완료 시간 및 통계
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_frame = total_time / frame_idx if frame_idx > 0 else 0
        
        print(f"  {part_name} 추론 완료: {frame_idx}프레임, " +
              f"소요시간: {total_time:.2f}초, " +
              f"프레임당: {avg_time_per_frame*1000:.2f}ms")

    return detections, frames