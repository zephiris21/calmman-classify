# 간단한 얼굴 이미지 처리 스크립트

이 스크립트는 지정된 디렉토리에서 얼굴 이미지를 감지하고 처리하여 `processed/new` 디렉토리에 저장합니다.

## 사용 방법

```bash
python simple_process_images.py 입력_디렉토리_경로
```

### 예시

```bash
# 프로젝트 루트 디렉토리에서 실행
python src/data_processing/simple_process_images.py data/raw/몰입

# 절대 경로 사용
python src/data_processing/simple_process_images.py D:/my_projects/calmman-facial-classification/data/raw/웃김
```

## 기능

- 지정된 입력 디렉토리에서 모든 이미지 파일(jpg, jpeg, png)을 탐색합니다.
- 각 이미지에서 얼굴을 감지하고 추출합니다.
- 추출된 얼굴 이미지는 224x224 크기로 조정됩니다.
- 모든 처리된 이미지는 `data/processed/new` 디렉토리에 저장됩니다.

## 참고사항

- 이 스크립트는 FaceDetector 클래스를 사용하며, MTCNN 모델을 활용합니다.
- 이미지당 여러 얼굴이 감지될 경우 각 얼굴은 별도의 파일로 저장됩니다.
- 처리 결과는 콘솔에 출력됩니다. 