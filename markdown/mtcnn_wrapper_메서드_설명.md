## 📋 **MTCNN 메서드들 로직 정리**

### **🔍 기본 감지 메서드들**

#### **1. `detect_faces_with_landmarks(img)`** (단일)
```python
input: PIL.Image 1개
↓
self.mtcnn.detect(img, landmarks=True)
↓
output: (boxes, probs, landmarks) - 1개 이미지의 결과
```

#### **2. `detect_faces_with_landmarks_batch(pil_images)`** (배치)
```python
input: [PIL.Image1, PIL.Image2, ...] 여러개
↓
self.mtcnn.detect(pil_images, landmarks=True)  # 리스트 전달
↓
output: (batch_boxes, batch_probs, batch_landmarks) - 배치 차원 추가
```

### **🎨 처리 메서드들**

#### **3. `process_image(img)`** (단일 - 기존)
```python
input: PIL.Image 1개
↓
detect_faces_with_landmarks() 호출
↓
각 얼굴별로: 크롭 → 리사이즈 → 정렬 → PIL.Image 변환
↓
output: [face_img1, face_img2, ...] PIL 이미지 리스트
```

#### **4. `process_image_batch(pil_images, metadata)`** (배치 - 신규)
```python
input: [PIL.Image1, PIL.Image2, ...], [metadata1, metadata2, ...]
↓
detect_faces_with_landmarks_batch() 호출
↓
2중 루프: 각 이미지 → 각 얼굴별로 _process_single_face() 호출
↓
output: [{face_image, frame_number, timestamp, ...}, ...] 딕셔너리 리스트
```

### **🛠️ 헬퍼 메서드들**

#### **5. `_process_single_face(img, box, landmark, ...)`**
```python
input: 원본이미지, 얼굴박스, 랜드마크
↓
크롭(box + margin) → 리사이즈(224x224) → 정렬(옵션)
↓
output: PIL.Image 얼굴 1개
```

## 🔄 **핵심 차이점**

| 구분 | 단일 처리 | 배치 처리 |
|------|-----------|-----------|
| **입력** | 이미지 1개 | 이미지 N개 |
| **MTCNN 호출** | N번 | 1번 |
| **출력** | PIL 이미지 리스트 | 메타데이터 포함 딕셔너리 |
| **성능** | 느림 | 10배+ 빠름 |

**핵심: 배치는 MTCNN을 1번만 호출해서 속도 향상!**