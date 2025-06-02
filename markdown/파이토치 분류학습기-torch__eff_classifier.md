# EfficientNet-B0 PyTorch 전이학습 이진분류 시스템 - 설명서

## 🎯 프로젝트 개요

본 프로젝트는 **PyTorch와 EfficientNet-B0**을 활용한 전이학습 기반 얼굴 표정 이진분류 시스템입니다. 침착맨의 "약올리기 vs 비약올리기" 표정을 자동으로 분류하는 모델을 구축합니다.

**핵심 특징**:
- ImageNet 사전훈련된 EfficientNet-B0 백본 활용
- 2단계 전이학습 (백본 동결 → 점진적 해제)
- Mixed Precision Training으로 메모리 최적화
- 클래스 불균형 해결을 위한 가중치 및 데이터 증강

## 🚀 실행 방법

### 환경 요구사항
```bash
pip install torch torchvision timm albumentations scikit-learn matplotlib seaborn tqdm pillow opencv-python
```

### 기본 실행
```bash
# 코드 파일 실행
python torch_eff_classifier.py
```

### 데이터 구조
```
data/processed/
├── teasing/          # 약올리기 이미지들
│   ├── image1.jpg
│   └── ...
└── non_teasing/      # 비약올리기 이미지들
    ├── image1.jpg
    └── ...
```

## 🧠 시스템 아키텍처

### 1. 전체 파이프라인

```
데이터 로딩 → 전처리 → 2단계 전이학습 → 성능 평가 → 결과 저장
     ↓           ↓           ↓            ↓          ↓
   원본분할   →  증강처리  →  백본동결    →  검증    →  시각화
                           →  미세조정
```

### 2. 모델 구조 (EfficientNetClassifier)

```python
class EfficientNetClassifier(nn.Module):
    def __init__(self):
        # 백본: timm.efficientnet_b0 (ImageNet 사전훈련)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        
        # 분류 헤드: Dropout + Linear
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 2)  # 이진분류
        )
```

**장점**:
- **경량성**: EfficientNet-B0는 파라미터 효율적
- **성능**: ImageNet 사전훈련으로 강력한 특징 추출 능력
- **유연성**: 백본과 분류 헤드 분리로 단계별 학습 가능

## 📊 2단계 전이학습 전략

### Stage 1: 백본 동결 + 분류 헤드 학습
```python
# 백본 완전 동결
model.freeze_backbone()

# 분류 헤드만 학습
optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 15 에포크 학습
```

**목적**: 새로운 도메인(얼굴 표정)에 맞는 분류 헤드를 안정적으로 학습

### Stage 2: 백본 일부 해제 + 미세조정
```python
# 마지막 3개 블록만 해제
model.unfreeze_backbone(layers_to_unfreeze=3)

# 더 작은 학습률로 미세조정
optimizer_stage2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# 20 에포크 미세조정
```

**목적**: 도메인 특화 특징을 학습하면서 사전훈련된 지식 보존

## 🔧 핵심 최적화 기법

### 1. 데이터 증강 및 불균형 해결
```python
# Albumentations 데이터 증강
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.3)
])

# 클래스별 균형 맞추기 (각 클래스 250개로 증강)
X_train_aug, y_train_aug = augment_data_pytorch(X_train_raw, y_train_raw, target_per_class=250)

# 클래스 가중치 적용
class_weights = torch.tensor([1.0, 2.0])  # teasing 클래스 가중치 증가
```

### 2. Mixed Precision Training
```python
# GPU 메모리 및 속도 최적화
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

### 3. 조기 종료 및 학습률 스케줄링
```python
# 성능 정체 시 자동 종료
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# 검증 손실 기준 조기 종료
if patience_counter >= patience:
    break
```

## 📈 데이터 흐름

### 1. 데이터 로딩 및 전처리
```python
# 1. 원본 데이터 로딩
X_non_teasing, y_non_teasing = load_images_robust(non_teasing_path, 0)
X_teasing, y_teasing = load_images_robust(teasing_path, 1)

# 2. 데이터 누수 방지를 위한 원본 분할
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. 훈련 데이터만 증강 (검증 데이터는 원본 유지)
X_train_aug, y_train_aug = augment_data_pytorch(X_train_raw, y_train_raw)
```

### 2. PyTorch Dataset & DataLoader
```python
# Transform 정의
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# Dataset 생성
train_dataset = FacialExpressionDataset(X_train_aug, y_train_aug, transform=train_transform)
val_dataset = FacialExpressionDataset(X_val_raw, y_val_raw, transform=val_transform)

# DataLoader 생성 (배치 처리)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
```

## 🎯 성능 모니터링

### 1. 실시간 학습 모니터링
```python
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        # Mixed Precision Training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 진행률 업데이트
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
```

### 2. 다중 지표 평가
```python
# 분류 성능 지표
f1 = f1_score(final_labels, final_predictions)
precision = precision_score(final_labels, final_predictions)
recall = recall_score(final_labels, final_predictions)

# 혼동 행렬
cm = confusion_matrix(final_labels, final_predictions)
```

## 📊 결과 시각화 및 저장

### 1. 학습 과정 시각화
- **2단계 학습 곡선**: 정확도/손실 변화 추이
- **혼동 행렬**: 분류 성능 세부 분석
- **예측 확률 분포**: 모델 신뢰도 분석

### 2. 자동 결과 저장
```
results/pytorch_efficientnet/
├── models/                    # 훈련된 모델 파일들
│   ├── best_model_stage1_*.pth
│   ├── best_model_stage2_*.pth
│   └── final_model_*.pth
├── plots/                     # 시각화 그래프
├── metrics/                   # 성능 요약 텍스트
└── logs/                      # 상세 학습 로그 (JSON)
```

## ⚙️ 하이퍼파라미터 설정

| 구분 | Stage 1 | Stage 2 | 설명 |
|------|---------|---------|------|
| **학습률** | 0.001 | 0.0001 | 미세조정 시 10배 감소 |
| **에포크** | 15 | 20 | 조기 종료로 자동 조절 |
| **배치 크기** | 32 | 32 | GPU 메모리 고려 |
| **Dropout** | 0.3 | 0.3 | 과적합 방지 |
| **가중치 감쇠** | 1e-4 | 1e-4 | 정규화 |

## 🔧 문제 해결

### 메모리 부족
```python
# 배치 크기 줄이기
batch_size = 16  # 또는 8

# Mixed Precision 활용
scaler = GradScaler()
```

### 성능 저하
```python
# 클래스 가중치 조정
class_weights = torch.tensor([1.0, 3.0])  # 소수 클래스 가중치 증가

# 데이터 증강 강화
target_per_class = 500  # 증강 데이터 수 증가
```

### 과적합
```python
# Dropout 증가
dropout_rate = 0.5

# 조기 종료 patience 감소
patience = 3
```

## 📈 기대 성능

**목표 성능**:
- 검증 정확도: **85%** 이상
- F1 Score: **0.85** 이상
- 기존 TensorFlow 모델 대비 **5-10%** 향상

**성능 개선 요인**:
1. **EfficientNet-B0**: 파라미터 효율적인 아키텍처
2. **2단계 전이학습**: 안정적인 학습 과정
3. **Mixed Precision**: 메모리 효율성과 안정성
4. **데이터 증강**: 일반화 성능 향상

## 💡 다음 단계 제안

### 성능이 목표에 미달할 경우
- **VGGFace2 사전훈련 모델 적용** (논문 기법)
- **강건한 최적화(Robust Optimization) 도입** (논문 Algorithm 1)
- **앙상블 모델** 구성
- **추가 데이터 수집** 및 라벨링 품질 개선

### 성능이 목표를 달성한 경우
- **실제 비디오 처리 시스템 통합**
- **모델 경량화** (모바일 배포 고려)
- **온라인 학습** 기능 추가
- **다중 감정 분류**로 확장

이 시스템은 논문의 이론적 기반을 실용적인 PyTorch 구현으로 발전시킨 사례로, 전이학습의 효과를 극대화하여 높은 분류 성능을 달성하는 것을 목표로 합니다.