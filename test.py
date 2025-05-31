import torch
from facenet_pytorch import MTCNN


# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
print(f"MTCNN 모델 설정: {device}")
