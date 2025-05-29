# 닭 객체 추적 및 ID 매핑 시스템

## 📋 프로젝트 개요

이 프로젝트는 **YOLO 모델을 활용한 닭 객체 추적 시스템**으로, 비디오에서 닭을 탐지하고 프레임 간 연속성을 유지하여 각 개체에 고유한 ID를 부여하는 시스템입니다. 특정 영역에 집중하여 분석함으로써 계산 효율성을 높이고, IoU 기반 매칭을 통해 안정적인 객체 추적을 구현합니다.

## 🎯 주요 기능

- **YOLO 기반 객체 탐지**: 사전 훈련된 닭 탐지 모델 활용
- **영역별 분석**: 3×3 그리드 분할로 관심 영역 집중 분석
- **IoU 기반 추적**: 프레임 간 객체 연결성 유지
- **ID 매핑 시스템**: 이전 ID와의 연관관계 추적
- **실시간 시각화**: 바운딩 박스와 ID 정보 표시
- **데이터 기록**: CSV 형태로 추적 정보 저장

## 📁 파일 구조

```
0520/
├── neaun_code.py          # 기본 추적 시스템
├── naeun_code2.py         # 간소화된 추적 시스템  
├── naeun_code3.py         # 하이브리드 추적 시스템
├── naeun_code4.py         # 디버깅 강화 버전
├── test.py               # 최종 테스트 버전
├── 20250521_021030/      # 실행 결과 폴더들
├── 20250521_022349/
├── 20250521_023659/
├── 20250521_025037/
└── README.md
```

## 🔍 파일별 상세 분석

### 1. `neaun_code.py` - 기본 추적 시스템

**핵심 특징:**
- YOLO 내장 추적 기능과 사용자 정의 IoU 매칭 결합
- ID 시작값: 1부터 순차 증가
- IoU 임계값: 0.5

**주요 로직:**
```python
# YOLO 추적 실패시 IoU 기반 매칭
if obj_id == -1:
    for last_box, last_id in last_detections:
        iou = calculate_iou(box, last_box)
        if iou > 0.5:
            obj_id = next_id
            prev_id = last_id
```

**장점:** 안정적인 기본 구현
**단점:** 복잡한 ID 관리 로직

### 2. `naeun_code2.py` - 간소화된 추적 시스템

**개선사항:**
- 모든 객체에 새 ID 부여 후 IoU로 연결
- 단순화된 ID 관리 방식

**핵심 차이점:**
```python
# 단순한 ID 할당 방식
obj_id = next_id
prev_id = ""

# 이전 프레임과 IoU 비교
for last_box, last_id in last_detections:
    iou = calculate_iou(box, last_box)
    if iou >= iou_threshold:
        prev_id = last_id
```

**장점:** 구현 단순성
**단점:** ID 중복 가능성

### 3. `naeun_code3.py` - 하이브리드 추적 시스템

**핵심 개선:**
- YOLO ID 우선 사용 정책
- 100000부터 시작하는 사용자 정의 ID로 충돌 방지

**ID 할당 전략:**
```python
if yolo_id != -1:
    obj_id = yolo_id  # YOLO ID 신뢰
else:
    obj_id = next_custom_id  # 사용자 정의 ID
    next_custom_id += 1
    # IoU 매칭으로 연결 시도
```

**장점:** ID 충돌 방지
**단점:** 메모리 사용량 증가

### 4. `naeun_code4.py` - 디버깅 강화 버전

**특별 기능:**
- 상세한 디버깅 출력
- 매 프레임 ID 할당 과정 시각화
- 추적 성공/실패 상태 명시

**디버깅 출력 예시:**
```python
print(f"[프레임 {frame_idx}] Box {i}: YOLO ID = {yolo_id} → 새 ID {obj_id} 부여")
print(f"   ↪️ 비교 대상: ID {last_id}, IOU = {round(iou, 3)}")
print(f"   ✅ 매칭됨 → prev_id = {prev_id}")
```

**장점:** 문제 진단 용이
**단점:** 성능 저하

### 5. `test.py` - 최종 테스트 버전

**최종 개선사항:**
- YOLO ID 우선 사용
- 정교한 로깅 시스템
- 추적 실패 케이스 상세 분석

**로직 플로우:**
1. YOLO ID 존재시 그대로 사용
2. 없으면 새 ID 부여 후 IoU 매칭
3. 매칭 성공시 연결 관계 기록
4. 전 과정 상세 로깅

## ⚙️ 공통 기술 구성요소

### 1. IoU (Intersection over Union) 계산
```python
def calculate_iou(box1, box2):
    # 교집합 영역 계산
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
```

### 2. 영상 영역 분할
- **그리드 시스템**: 3×3 분할로 9개 영역 생성
- **선택 영역**: (2,0) = 좌하단 영역 집중 분석
- **효과**: 계산 부하 감소 및 관심 영역 집중

### 3. 데이터 저장 형식

**tracking.csv** - 프레임별 객체 정보:
```csv
frame,id,x1,y1,x2,y2,center_x,center_y,prev_id
1,100000,150,200,250,300,200,250,
2,100001,155,205,255,305,205,255,100000
```

**mapping.csv** - ID 연결 관계:
```csv
frame,new_id,prev_id,iou
2,100001,100000,0.857
3,100002,100001,0.923
```

## 🚀 실행 방법

### 환경 설정
```bash
pip install ultralytics opencv-python numpy pandas
```

### 기본 실행
```bash
python neaun_code.py
```

### 디버깅 모드 실행
```bash
python naeun_code4.py
```

### 최종 버전 실행
```bash
python test.py
```

## 📊 시스템 설정

### 주요 파라미터
```python
# 비디오 및 모델 경로
video_path = "/home/nas/data/YeonSeung/chicken_30sec - Trim.mp4"
model_path = "/home/a202192020/객체탐지 프로젝트(송교수님)/양륜비박사님모델/weights/best.pt"

# 분석 설정
grid_size = 3           # 3x3 그리드 분할
selected_cell = (2, 0)  # 좌하단 영역 선택
conf_threshold = 0.5    # YOLO 신뢰도 임계값
iou_threshold = 0.5     # IoU 매칭 임계값
```

### 결과 출력
- **비디오**: `{timestamp}_output.mp4` - 추적 결과 시각화
- **로그**: `{timestamp}_tracking.csv` - 상세 추적 데이터
- **매핑**: `{timestamp}_id_mapping.csv` - ID 연결 관계

## 📈 버전별 발전 과정

| 버전 | 주요 특징 | ID 관리 | 장점 | 단점 |
|------|-----------|---------|------|------|
| neaun_code.py | 기본 구현 | YOLO + 사용자 정의 | 안정적 | 복잡함 |
| naeun_code2.py | 간소화 | 모두 새 ID | 단순함 | ID 중복 위험 |
| naeun_code3.py | 하이브리드 | YOLO 우선 + 큰수 시작 | 충돌 방지 | 메모리 증가 |
| naeun_code4.py | 디버깅 강화 | 상세 로깅 | 진단 용이 | 성능 저하 |
| test.py | 최종 버전 | 정교한 로깅 | 균형잡힌 성능 | - |

## 💡 활용 분야

### 주요 용도
1. **축산업 자동화**: 개체별 행동 모니터링
2. **동물 행동 연구**: 시간별 위치 변화 추적
3. **폐사체 감지**: 움직이지 않는 개체 식별 전처리
4. **밀도 분석**: 공간별 개체 수 계산

### 확장 가능성
- **다중 영역 동시 처리**: 전체 화면 커버
- **실시간 알림 시스템**: 이상 상황 감지
- **웹 인터페이스**: 실시간 모니터링 대시보드
- **데이터 분석**: 행동 패턴 분석 도구

## 🔧 기술적 특징

### 강점
- ✅ 경량화된 알고리즘으로 실시간 처리 가능
- ✅ YOLO 내장 추적과 사용자 정의 추적의 효과적 결합
- ✅ 상세한 로깅으로 디버깅 및 성능 분석 용이
- ✅ 모듈화된 구조로 유지보수성 높음
- ✅ 단계별 개선 과정으로 학습 자료로도 활용 가능

### 개선 가능 영역
- 🔄 다중 영역 동시 처리 기능
- 🔄 더 정교한 ID 재할당 로직
- 🔄 메모리 효율성 최적화
- 🔄 실시간 시각화 성능 향상

## 📞 개발 정보

**개발자**: a202192020  
**지도교수**: 송교수님  
**개발 기간**: 2025년 5월  
**주요 모델**: 양륜비박사님 제공 YOLO 모델

---

이 시스템은 닭 폐사체 감지 프로젝트의 핵심 구성 요소로, 개체별 추적을 통해 움직이지 않는 개체를 식별하는 데 필요한 기반 기술을 제공합니다.
