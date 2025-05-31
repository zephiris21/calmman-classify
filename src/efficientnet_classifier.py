# EfficientNet-B0 전이학습으로 얼굴 표정 이진분류
# 약올리기 vs 비약올리기 분류

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 랜덤 시드 고정
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("=== 🚀 EfficientNet-B0 전이학습 이진분류 ===")
print("ImageNet 사전학습 → 얼굴 표정 분류 전이학습")

# =================================
# 1. 데이터 로딩
# =================================
def load_images_robust(folder_path, label, max_images=None):
    """더 강력한 이미지 로딩 함수"""
    from PIL import Image
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
            img = img.astype('float32') # [0,255] 유지
            
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

# 데이터 경로 설정
base_path = r'D:\my_projects\calmman-facial-classification\data\processed'
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
# 2. 원본 데이터 분할 (데이터 누수 방지)
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
# 3. 데이터 증강 (훈련 데이터만!)
# =================================
def augment_data_efficientnet(X_array, y_array, target_per_class=250):
    """EfficientNet용 데이터 증강 함수"""
    
    class_0_indices = np.where(y_array == 0)[0]
    class_1_indices = np.where(y_array == 1)[0]
    
    class_0_data = [X_array[i] for i in class_0_indices]
    class_1_data = [X_array[i] for i in class_1_indices]
    
    print(f"\n=== 🔄 훈련 데이터 증강 ===")
    print(f"증강 전: 비약올리기 {len(class_0_data)}개, 약올리기 {len(class_1_data)}개")
    
    # 얼굴 표정에 적합한 증강 (보수적)
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
            # uint8로 변환 (Albumentations 요구사항)
            base_img_uint8 = (base_img * 255).astype(np.uint8)
            # 증강 적용
            augmented = transform(image=base_img_uint8)
            aug_img = augmented['image']
            # 다시 float32로 변환 [0,1] 범위
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
X_train_aug, y_train_aug = augment_data_efficientnet(X_train_raw, y_train_raw, target_per_class=250)

# =================================
# 4. 최종 데이터 준비
# =================================
print(f"\n=== 📦 최종 데이터 준비 ===")

# 훈련 데이터: 증강된 데이터 사용
X_train = np.array(X_train_aug)
y_train = np.array(y_train_aug)

# 검증 데이터: 원본 데이터 그대로 사용
X_val = X_val_raw
y_val = y_val_raw

print(f"최종 훈련 데이터: {X_train.shape}")
print(f"최종 검증 데이터: {X_val.shape}")
print(f"최종 훈련 클래스 분포: {np.bincount(y_train)}")
print(f"최종 검증 클래스 분포: {np.bincount(y_val)}")

print(f"✅ 데이터 누수 방지: 검증 데이터는 훈련 중 본 적 없는 원본 이미지만 사용")

# =================================
# 5. EfficientNet-B0 모델 구성
# =================================
print(f"\n=== 🏗️ EfficientNet-B0 전이학습 모델 구성 ===")

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=1):
    """EfficientNet-B0 기반 전이학습 모델 생성"""
    
    # 입력 레이어
    inputs = tf.keras.Input(shape=input_shape)
    
    # EfficientNet 전용 전처리 (ImageNet 정규화)
    x = preprocess_input(inputs)
    
    # EfficientNet-B0 백본 (사전학습된 가중치 사용, 분류 헤드 제외)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=x
    )
    
    # 백본 동결
    base_model.trainable = False
    
    # 분류 헤드 구성
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)  # 보수적인 정규화
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)  # 이진분류
    
    model = tf.keras.Model(inputs, outputs)
    
    print(f"✅ EfficientNet-B0 백본 로드 완료")
    print(f"   - 백본 상태: 동결됨")
    print(f"   - 분류 헤드: GAP → Dropout(0.3) → Dense(1)")
    
    return model, base_model

# 모델 생성
model, base_model = create_efficientnet_model()

# 모델 요약
print("\n모델 구조:")
model.summary()

# =================================
# 6. 1단계 학습: 백본 동결 + 분류 헤드 학습
# =================================
print(f"\n=== 🚀 1단계 학습: 백본 동결 + 분류 헤드 학습 ===")

# 1단계 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# 콜백 설정
callbacks_stage1 = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("1단계 학습 시작...")
print(f"  - 백본: 완전 동결")
print(f"  - 학습률: 0.001")
print(f"  - 목표: 분류 헤드 가중치 최적화")


print(f"  - 클래스 가중치 설정정")
class_weight = {0: 1.0, 1: 2.0}  # teasing 가중치 증가

# 1단계 학습 실행
history_stage1 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks_stage1,
    class_weight=class_weight,
    verbose=1
)

# 1단계 결과
stage1_loss, stage1_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n1단계 완료:")
print(f"  - 검증 정확도: {stage1_acc:.4f}")
print(f"  - 검증 손실: {stage1_loss:.4f}")

# =================================
# 7. 2단계 학습: 백본 일부 해제 + 미세조정
# =================================
print(f"\n=== 🔬 2단계 학습: 백본 일부 해제 + 미세조정 ===")

# 백본 일부 해제 (보수적으로 마지막 3개 레이어만)
base_model.trainable = True

# 마지막 3개 레이어만 학습 가능하게 설정
trainable_layers = 3
for layer in base_model.layers[:-trainable_layers]:
    layer.trainable = False

print(f"백본 레이어 해제:")
print(f"  - 전체 레이어: {len(base_model.layers)}개")
print(f"  - 해제 레이어: 마지막 {trainable_layers}개")
print(f"  - 동결 레이어: {len(base_model.layers) - trainable_layers}개")

# 2단계 컴파일 (더 작은 학습률)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 10배 작은 학습률
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# 콜백 설정 (더 보수적)
callbacks_stage2 = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

print("2단계 학습 시작...")
print(f"  - 백본: 마지막 {trainable_layers}개 레이어 해제")
print(f"  - 학습률: 0.0001 (10배 감소)")
print(f"  - 목표: 얼굴 표정에 특화된 미세조정")

# 2단계 학습 실행
history_stage2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks_stage2,
    class_weight=class_weight, 
    verbose=1
)

# =================================
# 8. 최종 성능 평가
# =================================
print(f"\n=== 📊 최종 성능 평가 ===")

# 최종 평가
final_loss, final_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"최종 검증 정확도: {final_accuracy:.4f}")
print(f"최종 검증 손실: {final_loss:.4f}")

# 예측 수행
y_pred_prob = model.predict(X_val, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 분류 보고서
classes = ['non_teasing', 'teasing']
print(f"\n분류 보고서:")
print(classification_report(y_val, y_pred, target_names=classes))

# F1 Score 계산
f1 = f1_score(y_val, y_pred)
print(f"\nF1 Score: {f1:.4f}")

# 혼동 행렬
print(f"\n혼동 행렬:")
cm = confusion_matrix(y_val, y_pred)
print(cm)

# 예측 분포 확인
print(f"\n예측 분포:")
print(f"예측 확률 범위: {y_pred_prob.min():.4f} ~ {y_pred_prob.max():.4f}")
print(f"예측 확률 표준편차: {y_pred_prob.std():.4f}")
print(f"예측 클래스 분포: {np.bincount(y_pred)}")

# =================================
# 9. 결과 시각화
# =================================
print(f"\n=== 📈 결과 시각화 ===")

# 두 단계 학습 과정 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1단계 정확도
axes[0,0].plot(history_stage1.history['binary_accuracy'], label='Stage 1 Training', color='blue')
axes[0,0].plot(history_stage1.history['val_binary_accuracy'], label='Stage 1 Validation', color='orange')
if 'binary_accuracy' in history_stage2.history:
    stage1_epochs = len(history_stage1.history['binary_accuracy'])
    stage2_epochs = range(stage1_epochs, stage1_epochs + len(history_stage2.history['binary_accuracy']))
    axes[0,0].plot(stage2_epochs, history_stage2.history['binary_accuracy'], label='Stage 2 Training', color='green')
    axes[0,0].plot(stage2_epochs, history_stage2.history['val_binary_accuracy'], label='Stage 2 Validation', color='red')
axes[0,0].axhline(y=0.5, color='gray', linestyle='--', label='Random Baseline')
axes[0,0].axvline(x=len(history_stage1.history['binary_accuracy']), color='purple', linestyle=':', label='Stage Transition')
axes[0,0].set_title('Model Accuracy (2-Stage Training)')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Accuracy')
axes[0,0].legend()
axes[0,0].grid(True)

# 1단계 손실
axes[0,1].plot(history_stage1.history['loss'], label='Stage 1 Training', color='blue')
axes[0,1].plot(history_stage1.history['val_loss'], label='Stage 1 Validation', color='orange')
if 'loss' in history_stage2.history:
    stage1_epochs = len(history_stage1.history['loss'])
    stage2_epochs = range(stage1_epochs, stage1_epochs + len(history_stage2.history['loss']))
    axes[0,1].plot(stage2_epochs, history_stage2.history['loss'], label='Stage 2 Training', color='green')
    axes[0,1].plot(stage2_epochs, history_stage2.history['val_loss'], label='Stage 2 Validation', color='red')
axes[0,1].axvline(x=len(history_stage1.history['loss']), color='purple', linestyle=':', label='Stage Transition')
axes[0,1].set_title('Model Loss (2-Stage Training)')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Loss')
axes[0,1].legend()
axes[0,1].grid(True)

# 혼동 행렬 히트맵
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=classes, yticklabels=classes, ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Actual')

# 예측 확률 분포
axes[1,1].hist(y_pred_prob[y_val==0], alpha=0.5, label='non_teasing', bins=20, color='blue')
axes[1,1].hist(y_pred_prob[y_val==1], alpha=0.5, label='teasing', bins=20, color='orange')
axes[1,1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
axes[1,1].set_title('Prediction Probability Distribution')
axes[1,1].set_xlabel('Predicted Probability')
axes[1,1].set_ylabel('Count')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.show()

# =================================
# 10. 성능 개선 분석
# =================================
print(f"\n=== 🎯 성능 개선 분석 ===")

baseline_accuracy = 0.775  # 기본 CNN 성능
improvement = final_accuracy - baseline_accuracy
improvement_percent = (improvement / baseline_accuracy) * 100

print(f"✅ EfficientNet-B0 전이학습 결과:")
print(f"   - 기본 CNN: 77.5%")
print(f"   - EfficientNet: {final_accuracy:.1%}")
print(f"   - 개선폭: {improvement:+.1%} ({improvement_percent:+.1f}%)")
print(f"   - F1 Score: {f1:.4f}")
print(f"   - 총 학습 에포크: {len(history_stage1.history['loss']) + len(history_stage2.history['loss'])}회")

if final_accuracy > 0.82:
    print(f"🎉 목표 달성! 전이학습이 매우 효과적")
elif final_accuracy > 0.80:
    print(f"👍 목표 근접! 전이학습 효과 확인")
elif final_accuracy > baseline_accuracy:
    print(f"✨ 성능 향상! 전이학습 도움됨")
else:
    print(f"🤔 성능 정체. 추가 조정 필요")

print(f"\n💡 다음 단계 제안:")
if final_accuracy < 0.80:
    print(f"   - 데이터 수집 확대")
    print(f"   - VGGFace2 사전학습 모델 시도")
    print(f"   - 하이퍼파라미터 튜닝")
else:
    print(f"   - 모델 앙상블 시도")
    print(f"   - 테스트 데이터로 최종 검증")
    print(f"   - 실제 서비스 적용 고려")

print(f"\n🚀 EfficientNet-B0 전이학습 완료!")