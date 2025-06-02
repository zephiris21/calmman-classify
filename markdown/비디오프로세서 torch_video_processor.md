# 침착맨 킹받는 순간 탐지 시스템 - 설명서

## 🎯 프로젝트 개요

본 프로젝트는 침착맨의 "킹받는 순간"을 자동으로 탐지하는 시스템입니다. 


## 🚀 실행 방법

### 기본 실행
```bash
# 기본 설정으로 실행 (config 파일에 지정된 영상)
python src/torch_video_processor.py

# 특정 영상 파일 지정
python src/torch_video_processor.py "침착맨_영상.mp4"

# 커스텀 디렉토리와 설정 파일
python src/torch_video_processor.py 영상파일.mp4 --dir /path/to/videos --config config/custom_config.yaml
```

### 설정 파일 수정
`config/config_torch.yaml`에서 다음 항목들을 조정할 수 있습니다:
- 비디오 경로 및 파일명
- 얼굴 인식 활성화/비활성화
- 배치 크기 및 성능 설정
- 출력 디렉토리 설정

## 🧠 시스템 로직 설명

### 1. 파이프라인 구조

본 시스템은 **"Classifying Emotions and Engagement in Online Learning Based on a Single Facial Expression Recognition Neural Network"** 논문의 아키텍처를 기반으로 합니다.

```
논문의 원래 파이프라인:
얼굴 탐지 → 감정 특징 추출 → 통계 집계 → 참여도 예측

우리의 변형된 파이프라인:
얼굴 탐지 → 감정 특징 추출 → 배치 분류 → 킹받는 순간 탐지
```

### 2. 핵심 컴포넌트

#### A. 얼굴 탐지 (MTCNN)
- **역할**: 비디오 프레임에서 얼굴 영역을 탐지하고 224x224 크기로 정규화
- **배치 처리**: 여러 프레임을 동시에 처리하여 효율성 향상
- **논문 연관성**: 논문의 "Face detection" 단계와 동일

#### B. 얼굴 인식 (FaceNet) - 선택적
- **역할**: 탐지된 얼굴이 침착맨인지 확인하여 필터링
- **설정**: `config.yaml`의 `face_recognition.enabled`로 활성화/비활성화 가능
- **장점**: 다른 사람의 얼굴을 제외하여 정확도 향상

#### C. 감정 특징 추출 (EfficientNet-B0)
- **핵심 아이디어**: 논문의 감정 특징 추출 네트워크를 재활용
- **구조**: 
  - EfficientNet-B0 백본 (논문에서 사용한 경량 모델)
  - 사전 훈련: 얼굴 인식 → AffectNet 감정 데이터셋으로 미세 조정
  - 출력: Softmax 이전 단계의 특징 벡터 (논문과 동일)

#### D. 배치 분류
- **변형 포인트**: 논문은 통계 집계 후 회귀 모델을 사용했지만, 우리는 직접 이진 분류
- **모델**: EfficientNet-B0 + 커스텀 분류 헤드
- **출력**: 킹받는 정도의 확률값 (0~1)

### 3. 멀티스레딩 아키텍처

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  프레임 읽기  │ →  │  얼굴 탐지   │ →  │  배치 분류   │
│   스레드     │    │   스레드     │    │   스레드     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  Frame Queue        Face Queue          Result Queue
```

**장점**:
- 각 단계가 독립적으로 실행되어 전체 처리 속도 향상
- GPU와 CPU 자원을 효율적으로 활용
- 대용량 비디오도 메모리 부족 없이 처리 가능

### 4. e-learning 참여도 측정 논문 내용과과의 차이점

| 구분 | 논문 (학생 참여도) | 우리 시스템 (킹받는 순간) |
|------|------------------|-------------------------|
| **목표** | 참여도 수준 예측 (4단계) | 화난 표정 이진 분류 |
| **입력** | 5-10초 비디오 클립 | 개별 프레임 |
| **특징 집계** | 표준편차 등 통계 함수 | 직접 분류 (집계 없음) |
| **모델** | 회귀 모델 (Ridge) | 이진 분류 (Neural Net) |
| **데이터셋** | EngageWild | 커스텀 침착맨 데이터 |

### 5. 데이터 흐름

```python
# 1. 프레임 추출ㄴㄴ
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 2. 얼굴 탐지 (MTCNN 배치)
face_images = face_detector.process_image_batch(frames)

# 3. 얼굴 인식 필터링 (선택적)
if face_recognition_enabled:
    embeddings = facenet_model(face_images)
    similarities = cosine_similarity(embeddings, target_embedding)
    filtered_faces = faces[similarities > threshold]

# 4. 감정 분류 (EfficientNet 배치)
features = efficientnet_backbone(face_images)  # 논문의 감정 특징
predictions = classifier_head(features)        # 우리의 킹받는 분류
angry_probability = softmax(predictions)[:, 1]
```

## 📊 출력 결과

- **하이라이트 이미지**: `results/날짜_영상명/highlights/` - 킹받는 순간의 얼굴 이미지
- **타임스탬프 JSON**: `angry_moments.json` - 정확한 시간과 신뢰도 정보
- **처리 로그**: 상세한 처리 과정과 성능 통계
- **필터링된 얼굴**: 인식되지 않은 얼굴들 (디버깅용, 옵션)

## ⚙️ 주요 설정 옵션

```yaml
face_recognition:
  enabled: true              # 얼굴 인식 활성화
  test_mode: false          # 테스트 모드 (필터링 건너뜀)
  similarity_threshold: 0.6  # 유사도 임계값

classifier:
  batch_size: 8             # 분류 배치 크기
  confidence_threshold: 0.7  # 킹받는 판정 임계값

performance:
  max_queue_size: 100       # 큐 최대 크기 (메모리 관리)
```

## 🔧 문제 해결

- **메모리 부족**: `batch_size`와 `max_queue_size` 줄이기
- **느린 처리**: `frame_skip` 늘리거나 얼굴 인식 비활성화
- **정확도 낮음**: `confidence_threshold` 조정 또는 얼굴 인식 활성화
- **CUDA 오류**: `device: cpu`로 설정하여 CPU 모드로 실행
