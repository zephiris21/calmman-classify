# 기본 CNN으로 얼굴 표정 이진분류 테스트
# 사전학습 없이 처음부터 학습하는 방식

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 랜덤 시드 고정
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

print("=== 🎯 기본 CNN 이진분류 테스트 ===")
print("사전학습 없이 처음부터 학습하는 방식")

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
            img = img.astype('float32') / 255.0
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            failed_files.append(f"{fname}: {str(e)}")
            continue
    
    print(f"   ✅ 성공: {len(images)}개")
    print(f"   ❌ 실패: {len(failed_files)}개")
    
    if failed_files:
        print(f"   실패 파일들: {failed_files[:3]}...")  # 처음 3개만 표시
    
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
def augment_data_simple(X_array, y_array, target_per_class=250):
    """훈련 데이터만 증강하는 함수"""
    
    class_0_indices = np.where(y_array == 0)[0]
    class_1_indices = np.where(y_array == 1)[0]
    
    class_0_data = [X_array[i] for i in class_0_indices]
    class_1_data = [X_array[i] for i in class_1_indices]
    
    print(f"\n=== 🔄 훈련 데이터 증강 ===")
    print(f"증강 전: 비약올리기 {len(class_0_data)}개, 약올리기 {len(class_1_data)}개")
    
    
    # 얼굴 표정에 적합한 증강
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
X_train_aug, y_train_aug = augment_data_simple(X_train_raw, y_train_raw, target_per_class=250)

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
# 5. 기본 CNN 모델 구성
# =================================
print(f"\n=== 🏗️ 기본 CNN 모델 구성 ===")

model = keras.Sequential([
    # 입력 정규화 (이미 정규화했지만 명시적으로)
    layers.Rescaling(1.0, input_shape=(224, 224, 3)),
    
    # 첫 번째 컨볼루션 블록
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    
    # 두 번째 컨볼루션 블록
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    
    # 세 번째 컨볼루션 블록
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    
    # 분류 헤드
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # 이진분류
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# 모델 요약
print("모델 구조:")
model.summary()

# =================================
# 6. 학습
# =================================
print(f"\n=== 🚀 모델 학습 ===")

# 콜백 설정
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]

# 학습 실행
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# =================================
# 7. 성능 평가
# =================================
print(f"\n=== 📊 성능 평가 ===")

# 최종 평가
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"최종 검증 정확도: {val_accuracy:.4f}")
print(f"최종 검증 손실: {val_loss:.4f}")

# 예측 수행
y_pred_prob = model.predict(X_val, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 분류 보고서
classes = ['non_teasing', 'teasing']
print(f"\n분류 보고서:")
print(classification_report(y_val, y_pred, target_names=classes))

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
# 8. 결과 시각화
# =================================
print(f"\n=== 📈 결과 시각화 ===")

# 학습 과정 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 정확도 그래프
axes[0].plot(history.history['binary_accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random Baseline')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# 손실 그래프
axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# =================================
# 9. 결과 요약
# =================================
print(f"\n=== 🎯 결과 요약 ===")
print(f"✅ 사전학습 없는 기본 CNN 결과:")
print(f"   - 최종 검증 정확도: {val_accuracy:.1%}")
print(f"   - 학습 에포크: {len(history.history['loss'])}회")
print(f"   - 예측 다양성: {y_pred_prob.std():.4f}")

if val_accuracy > 0.7:
    print(f"🎉 우수한 성능! 기본 CNN만으로도 좋은 결과")
elif val_accuracy > 0.6:
    print(f"👍 괜찮은 성능! 전이학습으로 더 개선 가능")
elif val_accuracy > 0.5:
    print(f"⚠️ 랜덤보다는 나음. 모델 개선 필요")
else:
    print(f"❌ 성능 부족. 데이터나 모델 재검토 필요")

print(f"\n전이학습과 비교해보시겠어요? 🤔")