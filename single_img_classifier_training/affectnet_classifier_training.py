# affectnet_simple_training.py
# AffectNet 사전학습 모델을 활용한 단순화된 얼굴 표정 이진분류
# 전처리된 260x260 정렬 이미지 사용

import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import time
from pathlib import Path

# PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

# 분석 및 시각화
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# 랜덤 시드 고정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

# CuDNN 결정적 동작 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=== 🚀 AffectNet 기반 얼굴 표정 이진분류 (단순화 버전) ===")
print("전처리된 260x260 정렬 이미지 → AffectNet 특징 → 분류")

# =================================
# GPU 설정 및 최적화
# =================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n=== 🔧 GPU 설정 ===")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 메모리 최적화
    torch.cuda.empty_cache()
    print("✅ CUDA 메모리 캐시 정리 완료")

# =================================
# AffectNet 사전학습 모델 로딩
# =================================
def load_affectnet_model(model_path, device):
    """AffectNet 사전학습 모델 로딩"""
    print(f"\n=== 🤖 AffectNet 모델 로딩 ===")
    print(f"모델 경로: {model_path}")
    
    try:
        # 모델 로딩
        affectnet_model = torch.load(model_path, map_location=device, weights_only=False)
        affectnet_model.eval()
        print("✅ AffectNet 모델 로딩 성공!")
        
        # 모델 정보 확인
        if hasattr(affectnet_model, 'classifier'):
            classifier = affectnet_model.classifier
            feature_dim = classifier.in_features
            num_classes = classifier.out_features
            print(f"   - 특징 차원: {feature_dim}")
            print(f"   - 원본 클래스: {num_classes}")
        else:
            raise ValueError("분류기를 찾을 수 없습니다.")
        
        return affectnet_model, feature_dim
        
    except Exception as e:
        print(f"❌ AffectNet 모델 로딩 실패: {e}")
        raise

# AffectNet 모델 로딩
affectnet_model_path = "./models/affectnet_emotions/enet_b2_8.pt"
affectnet_model, feature_dim = load_affectnet_model(affectnet_model_path, device)

# =================================
# 단순화된 Dataset 클래스
# =================================
class SimpleAffectNetDataset(Dataset):
    """전처리된 얼굴 이미지를 위한 단순 Dataset"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): 전처리된 이미지 경로 리스트
            labels (list): 레이블 리스트
            transform (transforms): 데이터 변환
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 전처리된 260x260 이미지 로드
            image = Image.open(img_path).convert('RGB')
            
            # Transform 적용
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패 {img_path}: {e}")
            # 실패 시 더미 이미지 반환
            dummy_image = torch.zeros(3, 260, 260)
            return dummy_image, torch.tensor(label, dtype=torch.long)

# =================================
# AffectNet 기반 분류 모델 정의 (수정된 버전)
# =================================
class AffectNetBinaryClassifier(nn.Module):
    """AffectNet 사전학습 모델 기반 이진분류기 (수정된 버전)"""
    
    def __init__(self, affectnet_model, feature_dim=1408, num_classes=2, dropout_rate=0.3):
        super(AffectNetBinaryClassifier, self).__init__()
        
        # AffectNet 백본에서 classifier 제거
        self.backbone = nn.ModuleList(list(affectnet_model.children())[:-1])
        
        # 새로운 이진분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # 특징 차원 저장
        self.feature_dim = feature_dim
        
        print(f"✅ AffectNet 이진분류기 생성 완료")
        print(f"   - 백본: AffectNet EfficientNet-B2")
        print(f"   - 특징 차원: {feature_dim}")
        print(f"   - 분류 헤드: Dropout({dropout_rate}) → Linear({feature_dim}, {num_classes})")
        
    def forward(self, x):
        # 백본을 통과하여 특징 추출
        for module in self.backbone:
            x = module(x)
        
        # 이미 Global Pooling이 적용된 상태라고 가정
        # x shape: [batch_size, feature_dim]
        
        # 만약 4D tensor라면 flatten
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        
        # 분류
        outputs = self.classifier(x)
        return outputs
    
    def extract_features(self, x):
        """특징 벡터만 추출 (분류 헤드 제외)"""
        with torch.no_grad():
            for module in self.backbone:
                x = module(x)
            
            if len(x.shape) > 2:
                x = torch.flatten(x, 1)
            
            return x
    
    def freeze_backbone(self):
        """백본 가중치 동결"""
        for module in self.backbone:
            for param in module.parameters():
                param.requires_grad = False
        print("🔒 AffectNet 백본 가중치 동결됨")
    
    def unfreeze_backbone(self, layers_to_unfreeze=3):
        """백본 일부 레이어 해제"""
        # 모든 백본 파라미터를 일단 동결
        for module in self.backbone:
            for param in module.parameters():
                param.requires_grad = False
        
        # 마지막 몇 개 모듈만 해제
        total_modules = len(self.backbone)
        unfreeze_from = max(0, total_modules - layers_to_unfreeze)
        
        for i in range(unfreeze_from, total_modules):
            for param in self.backbone[i].parameters():
                param.requires_grad = True
        
        print(f"🔓 백본 마지막 {layers_to_unfreeze}개 모듈 해제됨 ({unfreeze_from}번부터)")
        
        # 분류 헤드는 항상 학습 가능
        for param in self.classifier.parameters():
            param.requires_grad = True

# =================================
# 모델 생성 및 GPU 이동
# =================================
print(f"\n=== 🏗️ 모델 생성 ===")

# 모델 인스턴스 생성
model = AffectNetBinaryClassifier(
    affectnet_model=affectnet_model,
    feature_dim=feature_dim,
    num_classes=2,
    dropout_rate=0.3
)

# GPU로 이동
model = model.to(device)

# 모델 정보 출력
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 모델 정보:")
print(f"   - 총 파라미터: {total_params:,}")
print(f"   - 학습 가능 파라미터: {trainable_params:,}")
print(f"   - 특징 차원: {model.feature_dim}")

# =================================
# Transform 정의
# =================================
print(f"\n=== 🔄 Transform 설정 ===")

# 훈련용 Transform (데이터 증강)
train_transform = transforms.Compose([
    transforms.Resize((260, 260)),  # 이미 260이지만 안전을 위해
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# 검증용 Transform (증강 없음)
val_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✅ Transform 설정 완료")
print("   - 훈련용: Resize(260) + RandomFlip + ColorJitter + Rotation + 정규화")
print("   - 검증용: Resize(260) + 정규화만")

# =================================
# 데이터 로딩 함수
# =================================
def collect_preprocessed_images(folder_path, label, extensions=('.jpg', '.jpeg', '.png')):
    """전처리된 이미지 경로 수집"""
    image_paths = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"❌ {folder_path} 폴더가 존재하지 않습니다.")
        return image_paths, labels
    
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(extensions)]
    
    print(f"📁 {os.path.basename(folder_path)} 폴더:")
    print(f"   전체 파일: {len(all_files)}개")
    print(f"   이미지 파일: {len(image_files)}개")
    
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        image_paths.append(img_path)
        labels.append(label)
    
    return image_paths, labels

# 전처리된 데이터 경로 설정
processed_base_path = r'D:\my_projects\calmman-facial-classification\data\affectnet_processed'
teasing_processed_path = os.path.join(processed_base_path, 'teasing')
non_teasing_processed_path = os.path.join(processed_base_path, 'non_teasing')

print("\n=== 📁 전처리된 데이터 수집 ===")
# 비약올리기 전처리 이미지 경로 수집 (라벨: 0)
X_non_teasing_paths, y_non_teasing = collect_preprocessed_images(non_teasing_processed_path, 0)

# 약올리기 전처리 이미지 경로 수집 (라벨: 1)
X_teasing_paths, y_teasing = collect_preprocessed_images(teasing_processed_path, 1)

# 데이터 합치기
all_image_paths = X_non_teasing_paths + X_teasing_paths
all_labels = y_non_teasing + y_teasing

print(f"전처리된 데이터 수집 완료:")
print(f"  비약올리기: {len(X_non_teasing_paths)}개")
print(f"  약올리기: {len(X_teasing_paths)}개")
print(f"  총 이미지: {len(all_image_paths)}개")

if len(all_image_paths) == 0:
    print("❌ 수집된 이미지가 없습니다.")
    print("전처리 스크립트를 먼저 실행해주세요!")
    exit()

# =================================
# 데이터 분할
# =================================
print(f"\n=== ✂️ 데이터 분할 ===")

# 클래스 분포 확인
y_array = np.array(all_labels)
print(f"클래스 분포: {np.bincount(y_array)}")

# 훈련/검증 분할
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    all_image_paths, all_labels, 
    test_size=0.2, 
    random_state=SEED, 
    stratify=all_labels
)

print(f"훈련 데이터: {len(X_train_paths)}개")
print(f"검증 데이터: {len(X_val_paths)}개")
print(f"훈련 클래스 분포: {np.bincount(y_train)}")
print(f"검증 클래스 분포: {np.bincount(y_val)}")

# =================================
# Dataset 및 DataLoader 생성
# =================================
print(f"\n=== 📦 Dataset 및 DataLoader 생성 ===")

# Dataset 생성
train_dataset = SimpleAffectNetDataset(X_train_paths, y_train, transform=train_transform)
val_dataset = SimpleAffectNetDataset(X_val_paths, y_val, transform=val_transform)

print(f"✅ Dataset 생성 완료")
print(f"   - 훈련 Dataset: {len(train_dataset)}개")
print(f"   - 검증 Dataset: {len(val_dataset)}개")

# DataLoader 생성
batch_size = 32  # 전처리된 이미지로 배치 크기 증가 가능
num_workers = 0  # Windows 호환

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

print(f"✅ DataLoader 생성 완료")
print(f"   - 배치 크기: {batch_size}")
print(f"   - 훈련 배치 수: {len(train_loader)}")
print(f"   - 검증 배치 수: {len(val_loader)}")

# =================================
# 1단계 학습 설정 (백본 동결)
# =================================
print(f"\n=== 🎯 1단계 학습 설정 (백본 동결) ===")

# 백본 동결
model.freeze_backbone()

# 1단계 손실함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer_stage1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    weight_decay=1e-4
)

# 학습률 스케줄러
scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_stage1,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Mixed Precision 스케일러
scaler = GradScaler()

print("✅ 1단계 학습 설정 완료")
print(f"   - 손실함수: CrossEntropyLoss")
print(f"   - 옵티마이저: Adam (lr=0.001)")
print(f"   - 스케줄러: ReduceLROnPlateau")
print(f"   - Mixed Precision: 활성화")

# 학습 가능한 파라미터 확인
trainable_params_stage1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   - 1단계 학습 파라미터: {trainable_params_stage1:,}")

# =================================
# 학습 및 검증 함수
# =================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """한 에포크 훈련"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 통계 계산
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 진행률 업데이트
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """한 에포크 검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 예측값 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_predictions, all_labels

# =================================
# 결과 저장 디렉토리 생성
# =================================
# 결과 저장 경로
results_base = "results/affectnet_simple"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 디렉토리 생성
dirs_to_create = [
    f"{results_base}/models",
    f"{results_base}/plots", 
    f"{results_base}/metrics",
    f"{results_base}/logs"
]

for dir_path in dirs_to_create:
    os.makedirs(dir_path, exist_ok=True)

print(f"\n=== 📁 결과 저장 디렉토리 생성 ===")
print(f"기본 경로: {results_base}")

# =================================
# 1단계 학습 실행
# =================================
print(f"\n=== 🚀 1단계 학습: 백본 동결 + 분류 헤드 학습 ===")

# 1단계 설정
stage1_epochs = 15
best_val_loss = float('inf')
patience_counter = 0
patience = 5

print(f"1단계 학습 시작:")
print(f"  - 에포크: {stage1_epochs}")
print(f"  - 백본: 완전 동결")
print(f"  - 학습률: {optimizer_stage1.param_groups[0]['lr']}")
print(f"  - 조기 종료: patience={patience}")

# 1단계 학습 기록
stage1_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(stage1_epochs):
    print(f"\n📊 Epoch {epoch+1}/{stage1_epochs}")
    
    # 훈련
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer_stage1, scaler, device
    )
    
    # 검증
    val_loss, val_acc, val_predictions, val_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # 기록 저장
    stage1_history['train_loss'].append(train_loss)
    stage1_history['train_acc'].append(train_acc)
    stage1_history['val_loss'].append(val_loss)
    stage1_history['val_acc'].append(val_acc)
    
    # 스케줄러 업데이트
    scheduler_stage1.step(val_loss)
    
    # 결과 출력
    print(f"💯 Results - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"💯 Results - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"📚 LR: {optimizer_stage1.param_groups[0]['lr']:.6f}")
    
    # 조기 종료 체크
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 최고 모델 저장
        stage1_model_path = f"{results_base}/models/best_model_stage1_{timestamp}.pth"
        torch.save(model.state_dict(), stage1_model_path)
        print(f"💾 새로운 최고 모델 저장됨")
    else:
        patience_counter += 1
        print(f"⏰ Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("🛑 조기 종료!")
            break

# 최고 모델 로드
stage1_model_path = f"{results_base}/models/best_model_stage1_{timestamp}.pth"
checkpoint = torch.load(stage1_model_path, weights_only=False)

# OrderedDict인지 전체 모델인지 확인
if isinstance(checkpoint, dict):
    # state_dict인 경우
    model.load_state_dict(checkpoint)
else:
    # 전체 모델인 경우
    model = checkpoint

print(f"\n✅ 1단계 완료! 최고 검증 손실: {best_val_loss:.4f}")

# =================================
# 2단계 학습: 백본 일부 해제 + 미세조정
# =================================
print(f"\n=== 🔬 2단계 학습: 백본 일부 해제 + 미세조정 ===")

# 백본 일부 해제
model.unfreeze_backbone(layers_to_unfreeze=3)

# 2단계 옵티마이저 (더 작은 학습률)
optimizer_stage2 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001,  # 10배 작은 학습률
    weight_decay=1e-4
)

# 2단계 스케줄러
scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_stage2,
    mode='min',
    factor=0.3,
    patience=3,
    min_lr=1e-8
)

# 2단계 설정
stage2_epochs = 20
best_val_loss_stage2 = float('inf')
patience_counter_stage2 = 0
patience_stage2 = 7

# 학습 가능한 파라미터 확인
trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"2단계 학습 시작:")
print(f"  - 에포크: {stage2_epochs}")
print(f"  - 학습 파라미터: {trainable_params_stage2:,}")
print(f"  - 학습률: {optimizer_stage2.param_groups[0]['lr']}")
print(f"  - 조기 종료: patience={patience_stage2}")

# 2단계 학습 기록
stage2_history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(stage2_epochs):
    print(f"\n📊 Stage2 Epoch {epoch+1}/{stage2_epochs}")
    
    # 훈련
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer_stage2, scaler, device
    )
    
    # 검증
    val_loss, val_acc, val_predictions, val_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # 기록 저장
    stage2_history['train_loss'].append(train_loss)
    stage2_history['train_acc'].append(train_acc)
    stage2_history['val_loss'].append(val_loss)
    stage2_history['val_acc'].append(val_acc)
    
    # 스케줄러 업데이트
    scheduler_stage2.step(val_loss)
    
    # 결과 출력
    print(f"💯 Results - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"💯 Results - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"📚 LR: {optimizer_stage2.param_groups[0]['lr']:.6f}")
    
    # 조기 종료 체크
    if val_loss < best_val_loss_stage2:
        best_val_loss_stage2 = val_loss
        patience_counter_stage2 = 0
        # 최고 모델 저장
        stage2_model_path = f"{results_base}/models/best_model_stage2_{timestamp}.pth"
        torch.save(model.state_dict(), stage2_model_path)
        print(f"💾 새로운 최고 모델 저장됨")
    else:
        patience_counter_stage2 += 1
        print(f"⏰ Patience: {patience_counter_stage2}/{patience_stage2}")
        
        if patience_counter_stage2 >= patience_stage2:
            print("🛑 조기 종료!")
            break

# 최고 모델 로드
stage2_model_path = f"{results_base}/models/best_model_stage2_{timestamp}.pth"
checkpoint = torch.load(stage2_model_path, weights_only=False)

# OrderedDict인지 전체 모델인지 확인
if isinstance(checkpoint, dict):
    # state_dict인 경우
    model.load_state_dict(checkpoint)
else:
    # 전체 모델인 경우
    model = checkpoint

print(f"\n✅ 2단계 완료! 최고 검증 손실: {best_val_loss_stage2:.4f}")

# =================================
# 최종 성능 평가
# =================================
print(f"\n=== 📊 최종 성능 평가 ===")

# 최종 검증 수행
model.eval()
final_val_loss, final_val_acc, final_predictions, final_labels = validate_epoch(
    model, val_loader, criterion, device
)

print(f"최종 검증 결과:")
print(f"  - 검증 정확도: {final_val_acc:.2f}%")
print(f"  - 검증 손실: {final_val_loss:.4f}")

# F1 Score 계산
f1 = f1_score(final_labels, final_predictions)
precision = precision_score(final_labels, final_predictions)
recall = recall_score(final_labels, final_predictions)

print(f"  - F1 Score: {f1:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")

# 분류 보고서
classes = ['non_teasing', 'teasing']
print(f"\n분류 보고서:")
print(classification_report(final_labels, final_predictions, target_names=classes))

# 혼동 행렬
cm = confusion_matrix(final_labels, final_predictions)
print(f"\n혼동 행렬:")
print(cm)

# =================================
# 결과 시각화
# =================================
print(f"\n=== 📈 결과 시각화 ===")

# 전체 학습 기록 합치기
total_epochs_stage1 = len(stage1_history['train_loss'])
total_epochs_stage2 = len(stage2_history['train_loss'])

# 연속된 에포크로 변환
all_train_loss = stage1_history['train_loss'] + stage2_history['train_loss']
all_train_acc = stage1_history['train_acc'] + stage2_history['train_acc']
all_val_loss = stage1_history['val_loss'] + stage2_history['val_loss']
all_val_acc = stage1_history['val_acc'] + stage2_history['val_acc']

# 에포크 범위
epochs_stage1 = list(range(1, total_epochs_stage1 + 1))
epochs_stage2 = list(range(total_epochs_stage1 + 1, total_epochs_stage1 + total_epochs_stage2 + 1))
all_epochs = epochs_stage1 + epochs_stage2

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('AffectNet-based Binary Classification Results (Simple Version)', fontsize=16)

# 1. 정확도 그래프
axes[0,0].plot(epochs_stage1, stage1_history['train_acc'], 'b-', label='Stage 1 Training', linewidth=2)
axes[0,0].plot(epochs_stage1, stage1_history['val_acc'], 'b--', label='Stage 1 Validation', linewidth=2)
axes[0,0].plot(epochs_stage2, stage2_history['train_acc'], 'g-', label='Stage 2 Training', linewidth=2)
axes[0,0].plot(epochs_stage2, stage2_history['val_acc'], 'g--', label='Stage 2 Validation', linewidth=2)
axes[0,0].axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='Random Baseline')
axes[0,0].axvline(x=total_epochs_stage1, color='red', linestyle=':', alpha=0.7, label='Stage Transition')
axes[0,0].set_title('Model Accuracy (AffectNet 2-Stage Training)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy (%)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. 손실 그래프
axes[0,1].plot(epochs_stage1, stage1_history['train_loss'], 'b-', label='Stage 1 Training', linewidth=2)
axes[0,1].plot(epochs_stage1, stage1_history['val_loss'], 'b--', label='Stage 1 Validation', linewidth=2)
axes[0,1].plot(epochs_stage2, stage2_history['train_loss'], 'g-', label='Stage 2 Training', linewidth=2)
axes[0,1].plot(epochs_stage2, stage2_history['val_loss'], 'g--', label='Stage 2 Validation', linewidth=2)
axes[0,1].axvline(x=total_epochs_stage1, color='red', linestyle=':', alpha=0.7, label='Stage Transition')
axes[0,1].set_title('Model Loss (AffectNet 2-Stage Training)')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. 혼동 행렬 히트맵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=classes, yticklabels=classes, ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Actual')

# 4. 예측 확률 분포
model.eval()
with torch.no_grad():
    # 샘플 배치로 확률 분포 확인
    sample_images, sample_labels = next(iter(val_loader))
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
    sample_outputs = model(sample_images)
    sample_probs = torch.softmax(sample_outputs, dim=1)[:, 1].cpu().numpy()  # teasing 클래스 확률
    sample_labels_cpu = sample_labels.cpu().numpy()

axes[1,1].hist(sample_probs[sample_labels_cpu==0], alpha=0.5, label='non_teasing', bins=20, color='blue')
axes[1,1].hist(sample_probs[sample_labels_cpu==1], alpha=0.5, label='teasing', bins=20, color='orange')
axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
axes[1,1].set_title('Prediction Probability Distribution')
axes[1,1].set_xlabel('Predicted Probability (Teasing Class)')
axes[1,1].set_ylabel('Count')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()

# 그래프 저장
plot_path = f"{results_base}/plots/affectnet_simple_results_{timestamp}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"📈 학습 결과 그래프 저장: {plot_path}")

plt.show()

# =================================
# 성능 개선 분석
# =================================
print(f"\n=== 🎯 성능 개선 분석 ===")

# 기준선과 비교 (기존 ImageNet 모델 기준)
baseline_accuracy = 77.5  # 기존 ImageNet EfficientNet 성능
improvement = final_val_acc - baseline_accuracy
improvement_percent = (improvement / baseline_accuracy) * 100

print(f"✅ AffectNet vs ImageNet 비교:")
print(f"   - 기존 ImageNet EfficientNet: {baseline_accuracy}%")
print(f"   - AffectNet EfficientNet: {final_val_acc:.1f}%")
print(f"   - 개선폭: {improvement:+.1f}% ({improvement_percent:+.1f}%)")
print(f"   - F1 Score: {f1:.4f}")
print(f"   - 총 학습 에포크: {total_epochs_stage1 + total_epochs_stage2}회")

# 성능 평가
if final_val_acc > 90:
    print(f"🎉 뛰어난 성능! AffectNet 사전학습이 매우 효과적")
elif final_val_acc > 85:
    print(f"🎉 목표 초과 달성! AffectNet 전이학습이 매우 효과적") 
elif final_val_acc > 82:
    print(f"👍 목표 달성! AffectNet 전이학습 효과 확인")
elif final_val_acc > baseline_accuracy:
    print(f"✨ 성능 향상! AffectNet 사전학습 도움됨")
else:
    print(f"🤔 성능 정체. 추가 조정 필요")

# =================================
# 결과 저장
# =================================
print(f"\n=== 💾 결과 저장 ===")

# 최종 모델 저장
final_model_path = f"{results_base}/models/final_affectnet_simple_model_{timestamp}.pth"
torch.save(model.state_dict(), final_model_path)
print(f"🤖 최종 모델 저장: {final_model_path}")

# 학습 기록 저장
training_log = {
    'timestamp': timestamp,
    'model_config': {
        'backbone': 'AffectNet_EfficientNet_B2',
        'feature_dim': feature_dim,
        'num_classes': 2,
        'dropout_rate': 0.3,
        'input_size': 260,
        'preprocessed': True
    },
    'training_config': {
        'stage1_epochs': total_epochs_stage1,
        'stage2_epochs': total_epochs_stage2,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'mixed_precision': True,
        'face_alignment': True,
        'preprocessed_data': True
    },
    'final_metrics': {
        'val_accuracy': float(final_val_acc),
        'val_loss': float(final_val_loss),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'improvement_over_baseline': float(improvement)
    },
    'data_info': {
        'train_samples': len(X_train_paths),
        'val_samples': len(X_val_paths),
        'total_samples': len(all_image_paths)
    },
    'stage1_history': {
        'train_loss': [float(x) for x in stage1_history['train_loss']],
        'train_acc': [float(x) for x in stage1_history['train_acc']],
        'val_loss': [float(x) for x in stage1_history['val_loss']],
        'val_acc': [float(x) for x in stage1_history['val_acc']]
    },
    'stage2_history': {
        'train_loss': [float(x) for x in stage2_history['train_loss']],
        'train_acc': [float(x) for x in stage2_history['train_acc']],
        'val_loss': [float(x) for x in stage2_history['val_loss']],
        'val_acc': [float(x) for x in stage2_history['val_acc']]
    }
}

log_path = f"{results_base}/logs/affectnet_simple_training_log_{timestamp}.json"
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(training_log, f, indent=2, ensure_ascii=False)
print(f"📊 학습 로그 저장: {log_path}")

# 성능 지표 저장
metrics_text = f"""AffectNet 기반 얼굴 표정 이진분류 결과 (단순화 버전)
================================================
실행 시간: {timestamp}

최종 성능:
- 검증 정확도: {final_val_acc:.2f}%
- 검증 손실: {final_val_loss:.4f}
- F1 Score: {f1:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}

학습 설정:
- 1단계 에포크: {total_epochs_stage1}
- 2단계 에포크: {total_epochs_stage2}
- 총 에포크: {total_epochs_stage1 + total_epochs_stage2}
- 배치 크기: {batch_size}
- 입력 크기: 260x260
- 전처리된 데이터 사용: Yes
- 얼굴 정렬: 사전 적용됨
- Mixed Precision: 활성화

데이터 정보:
- 훈련 샘플: {len(X_train_paths)}개
- 검증 샘플: {len(X_val_paths)}개
- 총 샘플: {len(all_image_paths)}개

기존 대비 개선:
- 기존 ImageNet EfficientNet: {baseline_accuracy}%
- AffectNet EfficientNet: {final_val_acc:.1f}%
- 개선폭: {improvement:+.1f}% ({improvement_percent:+.1f}%)

혼동 행렬:
{cm}

분류 보고서:
{classification_report(final_labels, final_predictions, target_names=classes)}
"""

metrics_path = f"{results_base}/metrics/affectnet_simple_performance_summary_{timestamp}.txt"
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"📈 성능 요약 저장: {metrics_path}")

# =================================
# 특징 벡터 분석 (옵션)
# =================================
print(f"\n=== 🔬 특징 벡터 분석 ===")

# 검증 데이터로 특징 벡터 추출
model.eval()
all_features = []
all_feature_labels = []

print("특징 벡터 추출 중...")
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Extracting features"):
        images = images.to(device)
        features = model.extract_features(images)
        
        all_features.append(features.cpu().numpy())
        all_feature_labels.extend(labels.numpy())

# 특징 벡터 합치기
all_features = np.vstack(all_features)
all_feature_labels = np.array(all_feature_labels)

print(f"추출된 특징 벡터 형태: {all_features.shape}")
print(f"특징 벡터 범위: [{all_features.min():.3f}, {all_features.max():.3f}]")
print(f"특징 벡터 평균: {all_features.mean():.3f}")
print(f"특징 벡터 표준편차: {all_features.std():.3f}")

# 클래스별 특징 통계
for class_idx, class_name in enumerate(classes):
    class_features = all_features[all_feature_labels == class_idx]
    print(f"{class_name} 클래스:")
    print(f"  - 샘플 수: {len(class_features)}")
    print(f"  - 평균: {class_features.mean():.3f}")
    print(f"  - 표준편차: {class_features.std():.3f}")

# 특징 벡터 저장 (나중에 분석용)
features_save_path = f"{results_base}/features/extracted_features_{timestamp}.npz"
os.makedirs(os.path.dirname(features_save_path), exist_ok=True)
np.savez(features_save_path, 
         features=all_features, 
         labels=all_feature_labels, 
         class_names=classes)
print(f"🔬 특징 벡터 저장: {features_save_path}")

print(f"\n💡 다음 단계 제안:")
if final_val_acc < 80:
    print(f"   - 데이터 수집 확대")
    print(f"   - 하이퍼파라미터 튜닝")
    print(f"   - 더 강력한 데이터 증강")
    print(f"   - 클래스 가중치 조정")
elif final_val_acc < 85:
    print(f"   - 앙상블 모델 시도")
    print(f"   - 추가 미세조정")
    print(f"   - 테스트 타임 증강(TTA)")
elif final_val_acc < 90:
    print(f"   - 모델 앙상블")
    print(f"   - 실제 서비스 적용 테스트")
    print(f"   - 추론 최적화")
else:
    print(f"   - 실제 서비스 배포 준비")
    print(f"   - 모델 경량화 및 최적화")
    print(f"   - 실시간 추론 파이프라인 구축")
    print(f"   - A/B 테스트 준비")

print(f"\n🎉 AffectNet 기반 얼굴 표정 분류 학습 완료!")
print(f"📁 모든 결과가 {results_base}/ 에 저장되었습니다.")
print(f"🚀 논문의 핵심 기법을 성공적으로 적용했습니다!")
print(f"   - ✅ AffectNet 사전학습 모델 활용")
print(f"   - ✅ penultimate layer 특징 추출 (1408차원)")
print(f"   - ✅ 2단계 전이학습 (동결 → 해제)")
print(f"   - ✅ 전처리된 정렬 얼굴 이미지 사용")
print(f"   - ✅ Mixed Precision Training")
print(f"   - ✅ 포괄적인 성능 분석 및 시각화")

# =================================
# 모델 사용 예시 코드 생성
# =================================
inference_code = f'''
# 학습된 모델 사용 예시
import torch
from PIL import Image
from torchvision import transforms

# 모델 로딩
model = AffectNetBinaryClassifier(affectnet_model, feature_dim={feature_dim})
model.load_state_dict(torch.load('{final_model_path}'))
model.eval()
model.to(device)

# 전처리 파이프라인
preprocess = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 추론 함수
def predict_emotion(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    class_names = {classes}
    return class_names[predicted_class], confidence

# 사용 예시
# result, confidence = predict_emotion('path/to/face_image.jpg')
# print(f"예측: {{result}}, 신뢰도: {{confidence:.3f}}")
'''

inference_code_path = f"{results_base}/inference_example_{timestamp}.py"
with open(inference_code_path, 'w', encoding='utf-8') as f:
    f.write(inference_code)
print(f"📝 추론 예시 코드 저장: {inference_code_path}")