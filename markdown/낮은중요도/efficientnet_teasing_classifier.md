# EfficientNet 얼굴표정 분류기

> **킹받는사진 vs 평범한사진** 감정 상태를 자동 분류하는 딥러닝 모델


## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 특징](#-주요-특징)
- [설치 및 환경설정](#-설치-및-환경설정)
- [데이터 구조](#-데이터-구조)
- [모델 아키텍처](#-모델-아키텍처)
- [사용법](#-사용법)
- [성능 및 결과](#-성능-및-결과)
- [기술적 세부사항](#-기술적-세부사항)
- [문제해결](#-문제해결)

## 🎯 프로젝트 개요

이 프로젝트는 **EfficientNet-B0** 기반 전이학습을 활용하여 얼굴 표정에서 **약올리기(teasing)**와 **비약올리기(non-teasing)** 감정 상태를 자동으로 분류하는 이진 분류기입니다.

### 핵심 목표
- 얼굴 이미지에서 약올리기 의도 감지
- 실시간 감정 상태 분류
- 높은 정확도와 효율적인 추론 속도 달성

## ✨ 주요 특징

### 🔬 **전이학습 기반**
- **ImageNet** 사전학습된 EfficientNet-B0 활용
- **2단계 학습**: 백본 동결 → 미세조정
- 소규모 데이터셋에서도 높은 성능

### 🛡️ **데이터 누수 방지**
- 훈련/검증 데이터 사전 분할
- 검증 데이터는 원본 이미지만 사용
- 증강은 훈련 데이터에만 적용

### ⚖️ **클래스 불균형 해결**
- 데이터 증강으로 클래스 균형 조정
- 클래스 가중치 적용 옵션
- F1 Score 기반 성능 평가

### 🎨 **시각화 및 분석**
- 학습 과정 실시간 모니터링
- 예측 결과 시각적 분석
- 혼동행렬 및 확률 분포 시각화

## 🛠️ 설치 및 환경설정

### 필수 요구사항
- Python 3.9+
- CUDA 지원 GPU (권장)
- 최소 8GB RAM

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/calmman-facial-classification.git
cd calmman-facial-classification
```

### 2. 가상환경 생성
```bash
# Windows
python -m venv calm-env
calm-env\Scripts\activate

# Linux/Mac
python -m venv calm-env
source calm-env/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

#### 주요 패키지
```txt
tensorflow>=2.15.0
opencv-python>=4.8.0
albumentations>=1.3.1
scikit-learn>=1.3.0
matplotlib>=3.7.2
seaborn>=0.12.2
tqdm>=4.66.1
numpy>=1.24.3
pillow>=10.0.0
```

## 📁 데이터 구조

```
data/
├── processed/
│   ├── teasing/          # 약올리기 이미지
│   ├── non_teasing/      # 비약올리기 이미지
│   └── test_image/       # 테스트용 이미지
└── raw/                  # 원본 데이터 (선택사항)
```

### 데이터 요구사항
- **이미지 형식**: JPG, PNG, BMP, TIFF, WebP
- **권장 크기**: 최소 224x224 픽셀
- **훈련 데이터**: 클래스당 최소 50개 이상 권장

## 🏗️ 모델 아키텍처

### EfficientNet-B0 백본
```
입력 (224, 224, 3)
    ↓
EfficientNet 전처리
    ↓
EfficientNet-B0 백본 (ImageNet 사전학습)
    ↓
Global Average Pooling 2D
    ↓
Dropout (0.3)
    ↓
Dense(1, activation='sigmoid')
    ↓
출력 (이진 분류 확률)
```

### 훈련 전략
1. **1단계**: 백본 완전 동결 + 분류 헤드 학습
2. **2단계**: 백본 마지막 3개 레이어 해제 + 미세조정

## 🚀 사용법

### Jupyter Notebook 실행
```bash
jupyter notebook notebooks/efficientnet_teasing-v1.ipynb
```

### 셀 실행 순서
1. **셀 1-2**: 라이브러리 import 및 함수 정의
2. **셀 3-7**: 데이터 로딩 및 전처리
3. **셀 8-10**: 모델 구성 및 설정
4. **셀 11-14**: 2단계 훈련 실행
5. **셀 15-17**: 성능 평가 및 시각화
6. **셀 18-21**: 테스트 및 모델 저장

### 주요 설정 변경
```python
# 데이터 경로 (셀 3)
base_path = r'D:\your_project_path\data\processed'

# 클래스 가중치 (셀 10)
class_weight = {0: 1.0, 1: 2.0}  # teasing 가중치 증가

# 데이터 증강량 (셀 5)
target_per_class = 250  # 클래스당 목표 데이터 수
```

### 저장된 모델 사용
```python
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('models/efficientnet_teasing_classifier.h5')

# 단일 이미지 예측
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (224, 224))
img = img.astype('float32')
prediction = model.predict(np.expand_dims(img, axis=0))

# 결과 해석
probability = prediction[0][0]
result = "약올리기" if probability > 0.5 else "비약올리기"
confidence = probability if probability > 0.5 else 1 - probability

print(f"예측: {result} (확신도: {confidence:.2f})")
```

## 📊 성능 및 결과

### 최종 성능 지표
- **정확도**: 82.5%
- **F1 Score**: 0.74
- **향상도**: 기본 CNN 대비 +5.0%p (+6.5%)

### 클래스별 성능
| 클래스 | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 비약올리기 | 0.79 | 0.96 | 0.87 | 24 |
| 약올리기 | 0.91 | 0.62 | 0.74 | 16 |
| **평균** | **0.84** | **0.82** | **0.82** | **40** |

### 혼동 행렬
```
              예측
실제    비약올리기  약올리기
비약올리기    23       1
약올리기       6      10
```

### 학습 특성
- **총 에포크**: 57회 (1단계 + 2단계)
- **조기 종료**: 검증 손실 기반
- **학습률 감소**: 적응적 조정
- **정규화**: Dropout 0.3

## 🔧 기술적 세부사항

### 데이터 증강 기법
```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
])
```

### 전이학습 전략
1. **사전학습**: ImageNet에서 일반적 시각적 특징 학습
2. **백본 동결**: 저수준 특징 보존
3. **분류 헤드 학습**: 얼굴 표정 특화 특징 학습
4. **미세조정**: 고수준 특징 적응

### 클래스 가중치 효과
- **기본**: {0: 1.0, 1: 1.0} → 80.0% 정확도
- **조정**: {0: 1.0, 1: 2.0} → 82.5% 정확도
- **개선**: teasing 재현율 56% → 62.5%

### 메모리 최적화
- **배치 크기**: 32 (GPU 메모리 고려)
- **이미지 크기**: 224x224 (EfficientNet 최적 크기)
- **데이터 타입**: float32 (정밀도/속도 균형)

## 🛠️ 문제해결

### 일반적인 문제들

#### 1. **메모리 부족 오류**
```python
# 배치 크기 감소
batch_size = 16  # 기본값 32에서 감소

# 또는 이미지 크기 감소 (권장하지 않음)
img_size = 192  # 기본값 224에서 감소
```

#### 2. **CUDA 오류**
```bash
# CPU 모드로 강제 실행
export CUDA_VISIBLE_DEVICES=""
```

#### 3. **한글 경로 문제**
```python
# PIL로 이미지 로드
from PIL import Image
pil_img = Image.open(img_path)
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
```

#### 4. **모델 로드 실패**
```python
# 절대 경로 사용
model_path = r'D:\full\path\to\model.h5'
model = tf.keras.models.load_model(model_path)
```

### 성능 개선 방법

#### 1. **데이터 품질 향상**
- 고해상도 이미지 사용
- 다양한 조명/각도 데이터 수집
- 노이즈 이미지 제거

#### 2. **하이퍼파라미터 튜닝**
```python
# 학습률 조정
learning_rate_stage1 = 0.002  # 기본: 0.001
learning_rate_stage2 = 0.0002  # 기본: 0.0001

# 클래스 가중치 조정
class_weight = {0: 1.0, 1: 3.0}  # 더 강한 가중치

# 증강 강도 조정
target_per_class = 300  # 더 많은 증강 데이터
```

#### 3. **모델 앙상블**
```python
# 여러 모델 조합
models = [model1, model2, model3]
predictions = [model.predict(X) for model in models]
ensemble_pred = np.mean(predictions, axis=0)
```



## 📚 참고 자료

### 논문
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Facial Expression Recognition in Online Learning](https://ieeexplore.ieee.org/document/9846390)

### 기술 문서
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Albumentations Documentation](https://albumentations.ai/docs/)

### 관련 프로젝트
- [FER2013 Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [AffectNet Dataset](http://mohammadmahoor.com/affectnet/)

