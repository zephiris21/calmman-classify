#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class AffectNetBinaryClassifier(nn.Module):
    """AffectNet 사전학습 모델 기반 이진분류기 (실제 저장된 구조와 일치)"""
    
    def __init__(self, affectnet_model=None, feature_dim=1408, num_classes=2, dropout_rate=0.3):
        super(AffectNetBinaryClassifier, self).__init__()
        
        if affectnet_model is not None:
            # 학습 시: AffectNet 백본에서 classifier 제거
            self.backbone = nn.ModuleList(list(affectnet_model.children())[:-1])
            print(f"✅ AffectNet 백본 초기화 완료: {len(self.backbone)} 레이어")
        else:
            # 추론 시: 더미 백본 (state_dict 로드로 덮어씌워짐)
            self.backbone = nn.Sequential(nn.Identity())
            print("⚠️ 더미 백본 사용 (성능 저하 가능성)")
        
        # 이진분류 헤드 (저장된 구조: classifier.1이 Linear)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),  # classifier.0
            nn.Linear(feature_dim, num_classes)  # classifier.1
        )
        
        # 특징 차원 저장
        self.feature_dim = feature_dim
    
    def forward(self, x):
        # 백본을 통과하여 특징 추출
        if isinstance(self.backbone, nn.ModuleList):
            # ModuleList 타입인 경우
            for module in self.backbone:
                x = module(x)
        else:
            # Sequential 타입인 경우
            x = self.backbone(x)
        
        # 만약 4D tensor라면 flatten
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        
        # 분류
        outputs = self.classifier(x)
        return outputs


class TorchFacialClassifier:
    """AffectNet 기반 얼굴 표정 분류기"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = torch.device(config['classifier']['device'] 
                                  if torch.cuda.is_available() 
                                  else 'cpu')
        
        self.logger.info(f"디바이스: {self.device}")
        
        # 모델 로드
        self.model = self._load_model()
        
        # 전처리 파이프라인 (260x260 크기로 변경)
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),  # AffectNet 학습 크기
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 배치 처리 설정
        self.batch_size = config['classifier']['batch_size']
        self.batch_timeout = config['classifier']['batch_timeout']
        
        self.logger.info(f"AffectNet 분류기 초기화 완료 (배치: {self.batch_size}, 크기: 260x260)")
    
    def _load_model(self) -> nn.Module:
        """모델 로드 (전체 모델 우선)"""
        model_dir = self.config['classifier']['model_path']
        
        # 상대 경로를 절대 경로로 변환 (프로젝트 루트 기준)
        if not os.path.isabs(model_dir):
            # 현재 스크립트 위치에서 상위로 올라가서 프로젝트 루트 찾기
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # src의 상위 = 프로젝트 루트
            model_dir = os.path.join(project_root, model_dir)
            model_dir = os.path.normpath(model_dir)
        
        self.logger.info(f"모델 디렉토리 경로: {model_dir}")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"모델 디렉토리가 존재하지 않습니다: {model_dir}")
        
        # 모델 파일 찾기
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        
        if not model_files:
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_dir}")
        
        # stage2 모델 우선 선택 (기존 로직 유지)
        stage2_models = [f for f in model_files if 'stage2' in f]
        if stage2_models:
            latest_model = sorted(stage2_models)[-1]
        else:
            # final 모델 다음 우선
            final_models = [f for f in model_files if 'final' in f]
            if final_models:
                latest_model = sorted(final_models)[-1]
            else:
                latest_model = sorted(model_files)[-1]
        
        model_full_path = os.path.join(model_dir, latest_model)
        self.logger.info(f"모델 로드: {model_full_path}")
        
        try:
            # 전체 모델 로드 시도
            checkpoint = torch.load(model_full_path, map_location=self.device, weights_only=False)
            
            # 전체 모델인지 확인
            if hasattr(checkpoint, 'eval') and hasattr(checkpoint, 'forward'):
                # 전체 모델인 경우
                self.logger.info("✅ 전체 모델 감지 - 직접 사용")
                model = checkpoint.to(self.device)
                model.eval()
                return model
            
            elif isinstance(checkpoint, dict):
                # state_dict인 경우 - timm 사용하여 EfficientNet 백본 생성
                self.logger.info("✅ state_dict 감지 - EfficientNet 백본 생성")
                
                try:
                    # timm 라이브러리 임포트 시도
                    import timm
                    self.logger.info(f"timm 버전: {timm.__version__}")
                    
                    # 특징 차원 자동 감지
                    classifier_weight_key = 'classifier.1.weight'
                    if classifier_weight_key in checkpoint:
                        feature_dim = checkpoint[classifier_weight_key].shape[1]
                        self.logger.info(f"감지된 특징 차원: {feature_dim}")
                    else:
                        # 대안: classifier 관련 키 찾기
                        classifier_keys = [k for k in checkpoint.keys() if 'classifier' in k and 'weight' in k]
                        if classifier_keys:
                            feature_dim = checkpoint[classifier_keys[0]].shape[1]
                            self.logger.info(f"대안으로 감지된 특징 차원: {feature_dim}")
                        else:
                            feature_dim = 1408  # AffectNet EfficientNet-B2 모델의 특징 차원
                            self.logger.info(f"기본 특징 차원 사용: {feature_dim}")
                    
                    # AffectNet 모델과 동일한 EfficientNet-B2 생성
                    pretrained_model = timm.create_model('efficientnet_b2', pretrained=False)
                    
                    # 감지된 차원으로 모델 생성
                    model_instance = AffectNetBinaryClassifier(
                        affectnet_model=pretrained_model,
                        feature_dim=feature_dim,
                        num_classes=2,
                        dropout_rate=0.3
                    )
                    
                    # 가중치 로드 (백본은 무시하고 classifier만 로드)
                    model_instance.load_state_dict(checkpoint, strict=False)
                    model = model_instance.to(self.device)
                    model.eval()
                    
                    self.logger.info("✅ EfficientNet-B2 백본 + 학습된 분류기 로드 성공")
                    return model
                    
                except ImportError:
                    self.logger.warning("⚠️ timm 라이브러리를 찾을 수 없습니다. 더미 백본 사용")
                    
                    # timm 없는 경우 기존 더미 백본 사용
                    model_instance = AffectNetBinaryClassifier(
                        affectnet_model=None,
                        feature_dim=feature_dim,
                        num_classes=2,
                        dropout_rate=0.3
                    )
                    
                    # 가중치 로드 (더미 백본, classifier만 로드)
                    model_instance.load_state_dict(checkpoint, strict=False)
                    model = model_instance.to(self.device)
                    model.eval()
                    
                    self.logger.info("✅ state_dict 로드 성공 (strict=False 적용, 더미 백본)")
                    return model
            
            else:
                raise ValueError(f"알 수 없는 모델 형태: {type(checkpoint)}")
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise
    
    def preprocess_batch(self, face_images: List[Image.Image]) -> torch.Tensor:
        """이미지 배치 전처리 (260x260)"""
        batch_tensors = []
        
        for img in face_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            tensor = self.transform(img)
            batch_tensors.append(tensor)
        
        return torch.stack(batch_tensors).to(self.device)
    
    def predict_batch(self, face_images: List[Image.Image]) -> List[Dict]:
        """배치 예측"""
        if not face_images:
            return []
        
        try:
            # 전처리
            batch_tensor = self.preprocess_batch(face_images)
            
            # 추론
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model(batch_tensor)
                inference_time = time.time() - start_time
                
                # 소프트맥스로 확률 변환
                probabilities = torch.softmax(outputs, dim=1)
                
                # 킹받는 확률 (클래스 1)
                angry_probs = probabilities[:, 1].cpu().numpy()
            
            # 결과 구성
            results = []
            for i, prob in enumerate(angry_probs):
                results.append({
                    'confidence': float(prob),
                    'is_angry': prob > self.config['classifier']['confidence_threshold']
                })
            
            # 로깅
            if self.config['logging']['batch_summary']:
                angry_count = sum(1 for r in results if r['is_angry'])
                avg_confidence = np.mean([r['confidence'] for r in results])
                
                self.logger.info(
                    f"AffectNet 분류: {len(face_images)}개 → 킹받음 {angry_count}개 "
                    f"(평균 신뢰도: {avg_confidence:.3f}, {inference_time:.3f}초)"
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"배치 예측 오류: {e}")
            return [{'confidence': 0.0, 'is_angry': False} for _ in face_images]
    
    def get_memory_usage(self) -> Dict:
        """GPU 메모리 사용량 조회"""
        if self.device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}