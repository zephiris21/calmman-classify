# EfficientNet-B0 PyTorch 전이학습으로 얼굴 표정 이진분류
# 약올리기 vs 비약올리기 분류

import os
import cv2
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

# PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import timm

# 데이터 증강
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

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

print("=== 🚀 EfficientNet-B0 PyTorch 전이학습 이진분류 ===")
print("ImageNet 사전학습 → 얼굴 표정 분류 전이학습")

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
# PyTorch Dataset 클래스
# =================================
class FacialExpressionDataset(Dataset):
    """얼굴 표정 분류를 위한 커스텀 Dataset"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 이미지가 numpy array인 경우 PIL로 변환
        if isinstance(image, np.ndarray):
            # [0, 255] uint8로 변환
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # 데이터 증강 적용
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# =================================
# EfficientNet-B0 모델 정의
# =================================
class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 기반 이진분류 모델"""
    
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.3):
        super(EfficientNetClassifier, self).__init__()
        
        # timm에서 EfficientNet-B0 로드
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained,
            num_classes=0,  # 분류 헤드 제거
            drop_rate=dropout_rate
        )
        
        # 특징 차원 얻기
        self.feature_dim = self.backbone.num_features
        
        # 커스텀 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        print(f"✅ EfficientNet-B0 모델 생성 완료")
        print(f"   - 백본: timm.efficientnet_b0 (pretrained={pretrained})")
        print(f"   - 특징 차원: {self.feature_dim}")
        print(f"   - 분류 헤드: Dropout({dropout_rate}) → Linear({self.feature_dim}, {num_classes})")
        
    def forward(self, x):
        # 백본을 통과하여 특징 추출
        features = self.backbone(x)
        # 분류 헤드를 통과
        outputs = self.classifier(features)
        return outputs
    
    def freeze_backbone(self):
        """백본 가중치 동결"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🔒 백본 가중치 동결됨")
    
    def unfreeze_backbone(self, layers_to_unfreeze=3):
        """백본 일부 레이어 해제"""
        # 모든 백본 파라미터를 일단 동결
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 마지막 몇 개 레이어만 해제
        backbone_children = list(self.backbone.children())
        
        # EfficientNet의 마지막 블록들 해제
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
            total_blocks = len(blocks)
            unfreeze_from = max(0, total_blocks - layers_to_unfreeze)
            
            for i in range(unfreeze_from, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            
            print(f"🔓 백본 마지막 {layers_to_unfreeze}개 블록 해제됨 ({unfreeze_from}번부터)")
        
        # 배치 정규화와 분류 헤드는 항상 학습 가능
        for param in self.classifier.parameters():
            param.requires_grad = True

# =================================
# 모델 생성 및 GPU 이동
# =================================
print(f"\n=== 🏗️ 모델 생성 ===")

# 모델 인스턴스 생성
model = EfficientNetClassifier(
    num_classes=2,
    pretrained=True,
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
print(f"   - 백본 특징 차원: {model.feature_dim}")

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
# 데이터 로딩 (기존 함수 재사용)
# =================================
def load_images_robust(folder_path, label, max_images=None):
    """더 강력한 이미지 로딩 함수 (기존과 동일)"""
    images = []
    labels = []
    failed_files = []
    
    if not os.path.exists(folder_path):
        print(f"❌ {folder_path} 폴더가 존재하지 않습니다.")
        return images, labels
    
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    all_files = os.listdir(folder_path)
    image_files = [f for f in all_files if f.lower().endswith(extensions)]
    
    print(f"📁 {os.path.basename(folder_path)} 폴더:")
    print(f"   전체 파일: {len(all_files)}개")
    print(f"   이미지 파일: {len(image_files)}개")
    
    if max_images and len(image_files) > max_images:
        image_files = image_files[:max_images]
        print(f"   처리 대상: {len(image_files)}개 (제한됨)")
    
    for fname in tqdm(image_files, desc=f"Loading {os.path.basename(folder_path)}"):
        img_path = os.path.join(folder_path, fname)
        
        try:
            # 방법 1: OpenCV
            img = cv2.imread(img_path)
            
            if img is None:
                # 방법 2: PIL (한글 경로 문제 해결)
                pil_img = Image.open(img_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                failed_files.append(fname)
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0  # [0,1] 정규화
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            failed_files.append(f"{fname}: {str(e)}")
            continue
    
    print(f"   ✅ 성공: {len(images)}개")
    print(f"   ❌ 실패: {len(failed_files)}개")
    
    if failed_files:
        print(f"   실패 파일들: {failed_files[:3]}...")
    
    return images, labels

# 데이터 경로 설정 (기존과 동일)
base_path = 'data/processed'
teasing_path = os.path.join(base_path, 'teasing')
non_teasing_path = os.path.join(base_path, 'non_teasing')

print("\n=== 📁 데이터 로딩 ===")
# 비약올리기 이미지 로드 (라벨: 0)
X_non_teasing, y_non_teasing = load_images_robust(non_teasing_path, 0)

# 약올리기 이미지 로드 (라벨: 1)
X_teasing, y_teasing = load_images_robust(teasing_path, 1)

# 데이터 합치기
X = X_non_teasing + X_teasing
y = y_non_teasing + y_teasing

print(f"로딩 완료:")
print(f"  비약올리기: {len(X_non_teasing)}개")
print(f"  약올리기: {len(X_teasing)}개")
print(f"  총 이미지: {len(X)}개")

if len(X) == 0:
    print("❌ 로드된 이미지가 없습니다.")
    exit()

# =================================
# 원본 데이터 분할 (데이터 누수 방지)
# =================================
print(f"\n=== ✂️ 원본 데이터 분할 ===")

X_raw = np.array(X)
y_raw = np.array(y)

print(f"원본 데이터: {X_raw.shape}")
print(f"원본 클래스 분포: {np.bincount(y_raw)}")

# 원본 데이터를 먼저 분할
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=SEED, stratify=y_raw
)

print(f"원본 훈련 데이터: {X_train_raw.shape}")
print(f"원본 검증 데이터: {X_val_raw.shape}")
print(f"원본 훈련 클래스 분포: {np.bincount(y_train_raw)}")
print(f"원본 검증 클래스 분포: {np.bincount(y_val_raw)}")

# =================================
# PyTorch Transform 정의
# =================================
from torchvision import transforms

# 훈련용 Transform (데이터 증강)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# 검증용 Transform (증강 없음)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"\n=== 🔄 Transform 설정 ===")
print("✅ 훈련용: Resize + RandomFlip + ColorJitter + 정규화")
print("✅ 검증용: Resize + 정규화만")

# =================================
# 데이터 증강 (훈련 데이터만)
# =================================
def augment_data_pytorch(X_array, y_array, target_per_class=250):
    """PyTorch용 데이터 증강 함수"""
    
    class_0_indices = np.where(y_array == 0)[0]
    class_1_indices = np.where(y_array == 1)[0]
    
    class_0_data = [X_array[i] for i in class_0_indices]
    class_1_data = [X_array[i] for i in class_1_indices]
    
    print(f"\n=== 🔄 훈련 데이터 증강 ===")
    print(f"증강 전: 비약올리기 {len(class_0_data)}개, 약올리기 {len(class_1_data)}개")
    
    # Albumentations 사용 (기존과 동일)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
    ])

    final_class_0 = class_0_data.copy()
    final_class_1 = class_1_data.copy()

    # 비약올리기 증강
    if len(class_0_data) < target_per_class:
        need_count = target_per_class - len(class_0_data)
        print(f"비약올리기 {need_count}개 증강 중...")
        
        for i in tqdm(range(need_count)):
            base_img = class_0_data[i % len(class_0_data)]
            # uint8로 변환
            base_img_uint8 = (base_img * 255).astype(np.uint8)
            # 증강 적용
            augmented = transform(image=base_img_uint8)
            aug_img = augmented['image']
            # 다시 float32로 변환
            aug_img = aug_img.astype(np.float32) / 255.0
            final_class_0.append(aug_img)

    # 약올리기 증강
    if len(class_1_data) < target_per_class:
        need_count = target_per_class - len(class_1_data)
        print(f"약올리기 {need_count}개 증강 중...")
        
        for i in tqdm(range(need_count)):
            base_img = class_1_data[i % len(class_1_data)]
            # uint8로 변환
            base_img_uint8 = (base_img * 255).astype(np.uint8)
            # 증강 적용
            augmented = transform(image=base_img_uint8)
            aug_img = augmented['image']
            # 다시 float32로 변환
            aug_img = aug_img.astype(np.float32) / 255.0
            final_class_1.append(aug_img)
    
    # 최종 데이터
    final_X = final_class_0 + final_class_1
    final_y = [0] * len(final_class_0) + [1] * len(final_class_1)
    
    print(f"증강 후: 비약올리기 {len(final_class_0)}개, 약올리기 {len(final_class_1)}개")
    print(f"총 훈련 데이터: {len(final_X)}개")
    
    return final_X, final_y

# 훈련 데이터만 증강 (검증 데이터는 원본 유지!)
X_train_aug, y_train_aug = augment_data_pytorch(X_train_raw, y_train_raw, target_per_class=250)

# =================================
# Dataset 및 DataLoader 생성
# =================================
print(f"\n=== 📦 Dataset 및 DataLoader 생성 ===")

# Dataset 생성
train_dataset = FacialExpressionDataset(X_train_aug, y_train_aug, transform=train_transform)
val_dataset = FacialExpressionDataset(X_val_raw, y_val_raw, transform=val_transform)

print(f"✅ Dataset 생성 완료")
print(f"   - 훈련 Dataset: {len(train_dataset)}개")
print(f"   - 검증 Dataset: {len(val_dataset)}개")

# DataLoader 생성 (Windows 호환)
batch_size = 32
num_workers = 0  # Windows에서 멀티프로세싱 문제 방지

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # GPU 전송 최적화
    drop_last=True    # 마지막 배치 크기 통일
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
print(f"   - Worker 수: {num_workers}")
print(f"   - 훈련 배치 수: {len(train_loader)}")
print(f"   - 검증 배치 수: {len(val_loader)}")
print(f"   - GPU 메모리 최적화: pin_memory=True")

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
            
            # 예측값 저장 (나중에 분석용)
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
import os
from datetime import datetime

# 결과 저장 경로
results_base = "results/pytorch_efficientnet"
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
for dir_path in dirs_to_create:
    print(f"✅ {dir_path}")

# =================================
# 1단계 학습: 백본 동결 + 분류 헤드 학습
# =================================
print(f"\n=== 🚀 1단계 학습: 백본 동결 + 분류 헤드 학습 ===")

# 1단계 설정
stage1_epochs = 15
best_val_loss = float('inf')
patience_counter = 0
patience = 5

# 클래스 가중치 설정
class_counts = np.bincount(y_train_aug)
class_weights = torch.tensor([1.0, 2.0], dtype=torch.float).to(device)  # teasing 클래스 가중치 증가
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)

print(f"1단계 학습 시작:")
print(f"  - 에포크: {stage1_epochs}")
print(f"  - 백본: 완전 동결")
print(f"  - 학습률: {optimizer_stage1.param_groups[0]['lr']}")
print(f"  - 클래스 가중치: {class_weights}")
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
        model, train_loader, criterion_weighted, optimizer_stage1, scaler, device
    )
    
    # 검증
    val_loss, val_acc, val_predictions, val_labels = validate_epoch(
        model, val_loader, criterion_weighted, device
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
        # 최고 모델 저장 (정확한 경로로)
        stage1_model_path = f"{results_base}/models/best_model_stage1_{timestamp}.pth"
        torch.save(model.state_dict(), stage1_model_path)
        print(f"💾 새로운 최고 모델 저장됨: {stage1_model_path}")
    else:
        patience_counter += 1
        print(f"⏰ Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print("🛑 조기 종료!")
            break

# 최고 모델 로드
stage1_model_path = f"{results_base}/models/best_model_stage1_{timestamp}.pth"
model.load_state_dict(torch.load(stage1_model_path))
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
        model, train_loader, criterion_weighted, optimizer_stage2, scaler, device
    )
    
    # 검증
    val_loss, val_acc, val_predictions, val_labels = validate_epoch(
        model, val_loader, criterion_weighted, device
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
        # 최고 모델 저장 (정확한 경로로)
        stage2_model_path = f"{results_base}/models/best_model_stage2_{timestamp}.pth"
        torch.save(model.state_dict(), stage2_model_path)
        print(f"💾 새로운 최고 모델 저장됨: {stage2_model_path}")
    else:
        patience_counter_stage2 += 1
        print(f"⏰ Patience: {patience_counter_stage2}/{patience_stage2}")
        
        if patience_counter_stage2 >= patience_stage2:
            print("🛑 조기 종료!")
            break

# 최고 모델 로드
stage2_model_path = f"{results_base}/models/best_model_stage2_{timestamp}.pth"
model.load_state_dict(torch.load(stage2_model_path))
print(f"\n✅ 2단계 완료! 최고 검증 손실: {best_val_loss_stage2:.4f}")

# =================================
# 결과 저장 디렉토리 생성
# =================================
import os
from datetime import datetime

# 결과 저장 경로
results_base = "results/pytorch_efficientnet"
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
for dir_path in dirs_to_create:
    print(f"✅ {dir_path}")

# =================================
# 최종 성능 평가
# =================================
print(f"\n=== 📊 최종 성능 평가 ===")

# 최종 검증 수행
model.eval()
final_val_loss, final_val_acc, final_predictions, final_labels = validate_epoch(
    model, val_loader, criterion_weighted, device
)

print(f"최종 검증 결과:")
print(f"  - 검증 정확도: {final_val_acc:.2f}%")
print(f"  - 검증 손실: {final_val_loss:.4f}")

# F1 Score 계산
from sklearn.metrics import f1_score, precision_score, recall_score

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
fig.suptitle('EfficientNet-B0 PyTorch 2-Stage Training Results', fontsize=16)

# 1. 정확도 그래프
axes[0,0].plot(epochs_stage1, stage1_history['train_acc'], 'b-', label='Stage 1 Training', linewidth=2)
axes[0,0].plot(epochs_stage1, stage1_history['val_acc'], 'b--', label='Stage 1 Validation', linewidth=2)
axes[0,0].plot(epochs_stage2, stage2_history['train_acc'], 'g-', label='Stage 2 Training', linewidth=2)
axes[0,0].plot(epochs_stage2, stage2_history['val_acc'], 'g--', label='Stage 2 Validation', linewidth=2)
axes[0,0].axhline(y=50, color='gray', linestyle=':', alpha=0.7, label='Random Baseline')
axes[0,0].axvline(x=total_epochs_stage1, color='red', linestyle=':', alpha=0.7, label='Stage Transition')
axes[0,0].set_title('Model Accuracy (2-Stage Training)')
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
axes[0,1].set_title('Model Loss (2-Stage Training)')
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

# 4. 예측 확률 분포 (마지막 배치 기준)
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
axes[1,1].set_title('Prediction Probability Distribution (Sample)')
axes[1,1].set_xlabel('Predicted Probability (Teasing Class)')
axes[1,1].set_ylabel('Count')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()

# 그래프 저장
plot_path = f"{results_base}/plots/training_results_{timestamp}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"📈 학습 결과 그래프 저장: {plot_path}")

plt.show()

# =================================
# 성능 개선 분석
# =================================
print(f"\n=== 🎯 성능 개선 분석 ===")

# 기준선과 비교 (기존 TensorFlow 모델 기준)
baseline_accuracy = 77.5  # 기존 CNN 성능
improvement = final_val_acc - baseline_accuracy
improvement_percent = (improvement / baseline_accuracy) * 100

print(f"✅ PyTorch EfficientNet-B0 전이학습 결과:")
print(f"   - 기존 TensorFlow CNN: {baseline_accuracy}%")
print(f"   - PyTorch EfficientNet: {final_val_acc:.1f}%")
print(f"   - 개선폭: {improvement:+.1f}% ({improvement_percent:+.1f}%)")
print(f"   - F1 Score: {f1:.4f}")
print(f"   - 총 학습 에포크: {total_epochs_stage1 + total_epochs_stage2}회")

# 성능 평가
if final_val_acc > 85:
    print(f"🎉 목표 초과 달성! 전이학습이 매우 효과적")
elif final_val_acc > 82:
    print(f"🎉 목표 달성! 전이학습이 매우 효과적") 
elif final_val_acc > 80:
    print(f"👍 목표 근접! 전이학습 효과 확인")
elif final_val_acc > baseline_accuracy:
    print(f"✨ 성능 향상! 전이학습 도움됨")
else:
    print(f"🤔 성능 정체. 추가 조정 필요")

# =================================
# 결과 저장
# =================================
print(f"\n=== 💾 결과 저장 ===")

# 최종 모델 저장
final_model_path = f"{results_base}/models/final_model_{timestamp}.pth"
torch.save(model.state_dict(), final_model_path)
print(f"🤖 최종 모델 저장: {final_model_path}")

# 학습 기록 저장
import json
training_log = {
    'timestamp': timestamp,
    'model_config': {
        'backbone': 'efficientnet_b0',
        'num_classes': 2,
        'dropout_rate': 0.3
    },
    'training_config': {
        'stage1_epochs': total_epochs_stage1,
        'stage2_epochs': total_epochs_stage2,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'mixed_precision': True
    },
    'final_metrics': {
        'val_accuracy': float(final_val_acc),
        'val_loss': float(final_val_loss),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall)
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

log_path = f"{results_base}/logs/training_log_{timestamp}.json"
with open(log_path, 'w', encoding='utf-8') as f:
    json.dump(training_log, f, indent=2, ensure_ascii=False)
print(f"📊 학습 로그 저장: {log_path}")

# 성능 지표 저장
metrics_text = f"""PyTorch EfficientNet-B0 전이학습 결과
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
- Mixed Precision: 활성화

기존 대비 개선:
- 기존 TensorFlow CNN: {baseline_accuracy}%
- PyTorch EfficientNet: {final_val_acc:.1f}%
- 개선폭: {improvement:+.1f}% ({improvement_percent:+.1f}%)

혼동 행렬:
{cm}

분류 보고서:
{classification_report(final_labels, final_predictions, target_names=classes)}
"""

metrics_path = f"{results_base}/metrics/performance_summary_{timestamp}.txt"
with open(metrics_path, 'w', encoding='utf-8') as f:
    f.write(metrics_text)
print(f"📈 성능 요약 저장: {metrics_path}")

print(f"\n💡 다음 단계 제안:")
if final_val_acc < 80:
    print(f"   - 데이터 수집 확대")
    print(f"   - VGGFace2 사전학습 모델 시도 (논문 기법)")
    print(f"   - 하이퍼파라미터 튜닝")
    print(f"   - 강건한 최적화(Robust Optimization) 적용")
else:
    print(f"   - 모델 앙상블 시도")
    print(f"   - 테스트 데이터로 최종 검증")
    print(f"   - 실제 서비스 적용 고려")
    print(f"   - 논문 기법(VGGFace2 + 강건한 최적화) 적용으로 추가 향상")

print(f"\n🚀 PyTorch EfficientNet-B0 전이학습 완료!")
print(f"📁 모든 결과가 {results_base}/ 에 저장되었습니다.")