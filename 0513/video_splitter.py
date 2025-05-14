import cv2
import os
from tqdm import tqdm

def split_video_3x3(video_path, save_dir, progress_callback=None):
    """
    영상을 3x3 그리드로 분할하고 진행 상황을 실시간으로 표시
    
    Args:
        video_path: 분할할 영상 경로
        save_dir: 분할된 영상 저장 디렉토리
        progress_callback: 진행 상황을 외부로 전달할 콜백 함수
                          함수 형태: progress_callback(current_frame)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"입력 영상 정보: {width}x{height}, {fps}fps, 총 {total_frames}프레임")
    
    part_w = width // 3
    part_h = height // 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writers = []
    paths = []
    for i in range(3):
        for j in range(3):
            part_path = os.path.join(save_dir, f"part_{i}_{j}.mp4")
            writer = cv2.VideoWriter(part_path, fourcc, fps, (part_w, part_h))
            writers.append(writer)
            paths.append(part_path)
    
    # 분할 진행 상황을 표시할 tqdm 진행 막대
    progress_bar = tqdm(total=total_frames, desc="영상 분할", unit="프레임", 
                        ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        for i in range(3):
            for j in range(3):
                crop = frame[i*part_h:(i+1)*part_h, j*part_w:(j+1)*part_w]
                writers[i*3 + j].write(crop)
        
        frame_count += 1
        progress_bar.update(1)
        
        # 콜백 함수가 제공된 경우 진행 상황 전달
        if progress_callback:
            progress_callback(frame_count)
        
        # 매 30프레임(1초)마다 진행 상황 추가 정보 표시
        if frame_count % 30 == 0:
            split_progress = frame_count / total_frames * 100
            progress_bar.set_postfix({
                '완료': f'{frame_count}/{total_frames}',
                '진행률': f'{split_progress:.1f}%'
            })
    
    progress_bar.close()
    cap.release()
    for w in writers:
        w.release()

    print(f"영상 분할 완료: 총 {frame_count}프레임 처리됨")
    return paths