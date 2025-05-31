#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from mtcnn_wrapper import FaceDetector


class CalmmanVideoProcessor:
    """
    침착맨 킹받는 순간 탐지를 위한 비디오 프로세서
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        비디오 프로세서 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 성능 모니터링 변수
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'angry_moments': 0,
            'processing_start_time': None,
            'last_stats_time': time.time()
        }
        
        # 종료 플래그 추가
        self.stop_flag = False
        self.face_detection_done = False
        self.classification_done = False
        
        # 큐 생성
        self.frame_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        self.face_queue = queue.Queue(maxsize=self.config['performance']['max_queue_size'])
        self.result_queue = queue.Queue()
        
        # 얼굴 탐지기 초기화
        self._init_face_detector()
        
        # 분류 모델 로드
        self._load_classifier()
        
        # 출력 디렉토리 생성
        self._create_output_dirs()
        
        print("✅ CalmmanVideoProcessor 초기화 완료")
        self._print_config_summary()
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 설정 파일 로드: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            raise
    
    def _init_face_detector(self):
        """MTCNN 얼굴 탐지기 초기화"""
        mtcnn_config = self.config['mtcnn']
        
        self.face_detector = FaceDetector(
            image_size=mtcnn_config['image_size'],
            margin=mtcnn_config['margin'],
            prob_threshold=mtcnn_config['prob_threshold'],
            align_faces=mtcnn_config['align_faces']
        )
        
        print(f"✅ MTCNN 초기화 완료 (배치 크기: {mtcnn_config['batch_size']})")
    
    def _load_classifier(self):
        """분류 모델 로드"""
        try:
            model_path = self.config['classifier']['model_path']
            self.classifier = tf.keras.models.load_model(model_path)
            print(f"✅ 분류 모델 로드: {model_path}")
        except Exception as e:
            print(f"❌ 분류 모델 로드 실패: {e}")
            raise
    
    def _create_output_dirs(self):
        """출력 디렉토리 생성"""
        if self.config['output']['save_highlights']:
            os.makedirs(self.config['output']['highlights_dir'], exist_ok=True)
        
        if self.config['output']['save_timestamps']:
            os.makedirs(self.config['output']['timestamps_dir'], exist_ok=True)
    
    def _print_config_summary(self):
        """설정 요약 출력"""
        print("\n📋 설정 요약:")
        print(f"   프레임 스킵: {self.config['video']['frame_skip']}프레임마다")
        print(f"   MTCNN 배치: {self.config['mtcnn']['batch_size']}")
        print(f"   분류 워커: {self.config['classifier']['workers']}개")
        print(f"   큐 크기: {self.config['performance']['max_queue_size']}")
        print(f"   RAM 제한: {self.config['performance']['max_ram_gb']}GB")
        print()
    
    def process_video(self, video_path: str) -> Dict:
        """
        비디오 파일 처리
        
        Args:
            video_path (str): 처리할 비디오 파일 경로
            
        Returns:
            Dict: 처리 결과 요약
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        print(f"\n🎬 비디오 처리 시작: {os.path.basename(video_path)}")
        
        # 비디오 정보 가져오기
        video_info = self._get_video_info(video_path)
        print(f"   길이: {video_info['duration']:.1f}초, FPS: {video_info['fps']:.1f}")
        
        # 결과 저장용 및 플래그 초기화
        self.angry_moments = []
        self.stats['processing_start_time'] = time.time()
        self.face_detection_done = False
        self.classification_done = False
        self.stop_flag = False
        
        # 스레드 시작
        threads = []
        
        # 1. 프레임 읽기 스레드
        frame_thread = threading.Thread(
            target=self._frame_reader_worker, 
            args=(video_path,)
        )
        threads.append(frame_thread)
        
        # 2. 얼굴 탐지 스레드
        face_thread = threading.Thread(target=self._face_detection_worker)
        threads.append(face_thread)
        
        # 3. 분류 스레드
        classify_thread = threading.Thread(target=self._classification_worker)
        threads.append(classify_thread)
        
        # 4. 성능 모니터링 스레드 (데몬으로 설정)
        monitor_thread = threading.Thread(target=self._performance_monitor)
        monitor_thread.daemon = True  # 데몬 스레드로 설정
        threads.append(monitor_thread)
        
        # 모든 스레드 시작
        for thread in threads:
            thread.start()
        
        # 프레임 읽기 완료 대기
        frame_thread.join()
        print("✅ 프레임 읽기 완료")
        
        # 두 작업 모두 완료될 때까지 대기
        print("⏳ 얼굴 탐지 및 분류 완료 대기 중...")
        while not (self.face_detection_done and self.classification_done):
            time.sleep(0.1)
        
        print("✅ 모든 처리 완료!")
        
        # 종료 플래그 설정
        self.stop_flag = True
        
        # 워커 스레드들 정리
        face_thread.join(timeout=2)
        classify_thread.join(timeout=2)
        
        # 결과 저장
        results = self._save_results(video_path, video_info)
        
        # 최종 통계 출력
        self._print_final_stats()
        
        return results
    
    def _get_video_info(self, video_path: str) -> Dict:
        """비디오 정보 추출"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
    
    def _frame_reader_worker(self, video_path: str):
        """프레임 읽기 워커"""
        cap = cv2.VideoCapture(video_path)
        frame_skip = self.config['video']['frame_skip']
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 스킵
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # BGR to RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 타임스탬프 계산
                timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)
                
                # 큐에 추가
                frame_data = {
                    'frame': frame_rgb,
                    'frame_number': frame_count,
                    'timestamp': timestamp
                }
                
                self.frame_queue.put(frame_data)
                frame_count += 1
                
        except Exception as e:
            print(f"❌ 프레임 읽기 오류: {e}")
        finally:
            cap.release()
            # 종료 신호
            self.frame_queue.put(None)
            print("📹 프레임 읽기 워커 종료")
    
    def _face_detection_worker(self):
        """얼굴 탐지 워커"""
        batch_size = self.config['mtcnn']['batch_size']
        frame_batch = []
        
        try:
            while True:
                frame_data = self.frame_queue.get()
                
                if frame_data is None:  # 종료 신호
                    # 남은 배치 처리
                    if frame_batch:
                        self._process_face_batch(frame_batch)
                    self.face_queue.put(None)  # 다음 워커에 종료 신호
                    break
                
                frame_batch.append(frame_data)
                
                # 배치가 찼으면 처리
                if len(frame_batch) >= batch_size:
                    self._process_face_batch(frame_batch)
                    frame_batch = []
                
                self.frame_queue.task_done()
                
        except Exception as e:
            print(f"❌ 얼굴 탐지 오류: {e}")
        finally:
            self.face_detection_done = True  # 완료 플래그 설정
            print("✅ 얼굴 탐지 완료")
    
    def _process_face_batch(self, frame_batch: List[Dict]):
        """프레임 배치에서 얼굴 탐지"""
        for frame_data in frame_batch:
            try:
                # PIL 이미지로 변환
                pil_image = Image.fromarray(frame_data['frame'])
                
                # 얼굴 탐지
                face_images = self.face_detector.process_image(pil_image)
                
                if face_images:
                    for face_img in face_images:
                        face_data = {
                            'face_image': face_img,
                            'frame_number': frame_data['frame_number'],
                            'timestamp': frame_data['timestamp']
                        }
                        self.face_queue.put(face_data)
                        self.stats['faces_detected'] += 1
                
                self.stats['frames_processed'] += 1
                
            except Exception as e:
                print(f"⚠️ 프레임 {frame_data['frame_number']} 처리 실패: {e}")
    
    def _classification_worker(self):
        """분류 워커"""
        try:
            while True:
                face_data = self.face_queue.get()
                
                if face_data is None:  # 종료 신호
                    break
                
                try:
                    # 이미지를 numpy 배열로 변환
                    img_array = np.array(face_data['face_image'])
                    img_array = img_array.astype('float32')  # [0,255] 유지
                    img_array = np.expand_dims(img_array, axis=0)

                    # EfficientNet 전용 정규화
                    img_array = preprocess_input(img_array)
                    
                    # 분류 예측
                    prediction = self.classifier.predict(img_array, verbose=0)
                    confidence = float(prediction[0][0])  # 킹받는 확률
                    
                    # 임계값 확인
                    threshold = self.config['classifier']['confidence_threshold']
                    is_angry = confidence > threshold
                    
                    if is_angry:
                        angry_moment = {
                            'timestamp': face_data['timestamp'],
                            'frame_number': face_data['frame_number'],
                            'confidence': confidence
                        }
                        
                        self.angry_moments.append(angry_moment)
                        self.stats['angry_moments'] += 1
                        
                        # 킹받는 프레임 저장 (옵션)
                        if self.config['output']['save_highlights']:
                            self._save_highlight_image(face_data, confidence)
                        
                        if self.config['debug']['verbose']:
                            timestamp_str = str(timedelta(seconds=int(face_data['timestamp'])))
                            print(f"😡 킹받는 순간 발견! {timestamp_str} (신뢰도: {confidence:.3f})")
                
                except Exception as e:
                    print(f"⚠️ 분류 오류: {e}")
                
                self.face_queue.task_done()
                
        except Exception as e:
            print(f"❌ 분류 워커 오류: {e}")
        finally:
            self.classification_done = True  # 완료 플래그 설정
            print("✅ 분류 처리 완료")
    
    def _save_highlight_image(self, face_data: Dict, confidence: float):
        """킹받는 순간 이미지 저장"""
        try:
            timestamp_str = f"{int(face_data['timestamp']):05d}"
            filename = f"angry_{timestamp_str}_{confidence:.3f}.jpg"
            
            save_path = os.path.join(
                self.config['output']['highlights_dir'],
                filename
            )
            
            face_data['face_image'].save(save_path)
            
        except Exception as e:
            print(f"⚠️ 하이라이트 이미지 저장 실패: {e}")
    
    def _performance_monitor(self):
        """성능 모니터링"""
        interval = self.config['performance']['monitoring_interval']
        
        while not self.stop_flag:  # 종료 플래그 확인
            time.sleep(interval)
            
            if self.stats['processing_start_time'] is None:
                continue
            
            # 현재 통계
            elapsed = time.time() - self.stats['processing_start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            
            # 큐 상태
            frame_queue_size = self.frame_queue.qsize()
            face_queue_size = self.face_queue.qsize()
            
            if self.config['debug']['performance_log']:
                print(f"📊 [{elapsed:.1f}s] "
                      f"프레임: {self.stats['frames_processed']} "
                      f"({fps:.1f} FPS), "
                      f"얼굴: {self.stats['faces_detected']}, "
                      f"킹받음: {self.stats['angry_moments']}, "
                      f"큐: {frame_queue_size}/{face_queue_size}")
        
        print("📊 성능 모니터링 종료")
    
    def _save_results(self, video_path: str, video_info: Dict) -> Dict:
        """결과 저장"""
        results = {
            'video_path': video_path,
            'video_info': video_info,
            'processing_stats': self.stats.copy(),
            'angry_moments': self.angry_moments,
            'total_angry_moments': len(self.angry_moments)
        }
        
        # 타임스탬프 JSON 저장
        if self.config['output']['save_timestamps']:
            video_name = Path(video_path).stem
            timestamp_file = os.path.join(
                self.config['output']['timestamps_dir'],
                f"{video_name}_angry_moments.json"
            )
            
            with open(timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 결과 저장: {timestamp_file}")
        
        return results
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        elapsed = time.time() - self.stats['processing_start_time']
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        print(f"\n🎯 처리 완료!")
        print(f"   총 처리 시간: {elapsed:.1f}초")
        print(f"   처리된 프레임: {self.stats['frames_processed']}개 ({fps:.1f} FPS)")
        print(f"   탐지된 얼굴: {self.stats['faces_detected']}개")
        print(f"   킹받는 순간: {self.stats['angry_moments']}개")
        print()


def main():
    """메인 실행 함수"""
    import sys
    
    # 기본 설정
    video_path = "yt_download/downloads/웃음이 필요할 때 보는 킹받는 침착맨.mp4"
    
    try:
        # 프로세서 초기화
        processor = CalmmanVideoProcessor()
        
        # 비디오 처리
        results = processor.process_video(video_path)
        
        print("✅ 모든 처리가 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 강제 종료
        print("🔚 프로그램을 종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()