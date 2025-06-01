#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import yaml
import time
import json
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

from mtcnn_wrapper import FaceDetector
from pytorch_classifier import TorchFacialClassifier


class TorchVideoProcessor:
    """
    PyTorch 기반 침착맨 킹받는 순간 탐지를 위한 비디오 프로세서
    """
    
    def __init__(self, config_path: str = "../config/config_torch.yaml"):
        """
        비디오 프로세서 초기화
        
        Args:
            config_path (str): 설정 파일 경로
        """
        # 설정 로드
        self.config = self._load_config(config_path)
        
        # 로깅 시스템 초기화
        self._setup_logging()
        
        # 성능 모니터링 변수
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'angry_moments': 0,
            'processing_start_time': None,
            'last_stats_time': time.time(),
            'batch_count': 0,
            'total_inference_time': 0
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
        
        # PyTorch 분류 모델 로드
        self._load_classifier()
        
        # 출력 디렉토리 생성
        self._create_output_dirs()
        
        self.logger.info("✅ TorchVideoProcessor 초기화 완료")
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
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        # 로거 생성
        self.logger = logging.getLogger('TorchVideoProcessor')
        self.logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 포매터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러 (옵션)
        if self.config['logging']['save_logs']:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"torch_processing_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"로그 파일: {log_file}")
    
    def _init_face_detector(self):
        """MTCNN 얼굴 탐지기 초기화"""
        mtcnn_config = self.config['mtcnn']
        
        self.face_detector = FaceDetector(
            image_size=mtcnn_config['image_size'],
            margin=mtcnn_config['margin'],
            prob_threshold=mtcnn_config['prob_threshold'],
            align_faces=mtcnn_config['align_faces']
        )
        
        self.logger.info(f"✅ MTCNN 초기화 완료 (배치 크기: {mtcnn_config['batch_size']})")
    
    def _load_classifier(self):
        """PyTorch 분류 모델 로드"""
        try:
            self.classifier = TorchFacialClassifier(self.config)
            self.logger.info("✅ PyTorch 분류 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"❌ 분류 모델 로드 실패: {e}")
            raise
    
    def _create_output_dirs(self):
        """출력 디렉토리 생성"""
        base_dir = self.config['output']['base_dir']
        os.makedirs(base_dir, exist_ok=True)
        
        if self.config['logging']['save_logs']:
            os.makedirs("logs", exist_ok=True)
    
    def _create_video_output_dir(self, video_path: str) -> str:
        """영상별 출력 디렉토리 생성"""
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # 특수문자 제거
        safe_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        video_dir = os.path.join(
            self.config['output']['base_dir'],
            f"{timestamp}_{safe_name}"
        )
        
        # 하위 디렉토리 생성
        if self.config['output']['save_highlights']:
            os.makedirs(os.path.join(video_dir, "highlights"), exist_ok=True)
        
        if self.config['output']['save_timestamps']:
            os.makedirs(os.path.join(video_dir, "timestamps"), exist_ok=True)
        
        if self.config['output']['save_processing_log']:
            os.makedirs(os.path.join(video_dir, "logs"), exist_ok=True)
        
        return video_dir
    
    def _print_config_summary(self):
        """설정 요약 출력"""
        self.logger.info("📋 설정 요약:")
        self.logger.info(f"   프레임 스킵: {self.config['video']['frame_skip']}프레임마다")
        self.logger.info(f"   MTCNN 배치: {self.config['mtcnn']['batch_size']}")
        self.logger.info(f"   분류 배치: {self.config['classifier']['batch_size']}")
        self.logger.info(f"   배치 타임아웃: {self.config['classifier']['batch_timeout']}초")
        self.logger.info(f"   큐 크기: {self.config['performance']['max_queue_size']}")
        self.logger.info(f"   디바이스: {self.config['classifier']['device']}")
    
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
        
        self.logger.info(f"🎬 비디오 처리 시작: {os.path.basename(video_path)}")
        
        # 영상별 출력 디렉토리 생성
        self.video_output_dir = self._create_video_output_dir(video_path)
        self.logger.info(f"📁 출력 디렉토리: {self.video_output_dir}")
        
        # 비디오 정보 가져오기
        video_info = self._get_video_info(video_path)
        self.logger.info(f"   길이: {video_info['duration']:.1f}초, FPS: {video_info['fps']:.1f}")
        
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
        
        # 3. 배치 분류 스레드
        classify_thread = threading.Thread(target=self._batch_classification_worker)
        threads.append(classify_thread)
        
        # 4. 성능 모니터링 스레드 (데몬으로 설정)
        monitor_thread = threading.Thread(target=self._performance_monitor)
        monitor_thread.daemon = True
        threads.append(monitor_thread)
        
        # 모든 스레드 시작
        for thread in threads:
            thread.start()
        
        # 프레임 읽기 완료 대기
        frame_thread.join()
        self.logger.info("✅ 프레임 읽기 완료")
        
        # 두 작업 모두 완료될 때까지 대기
        self.logger.info("⏳ 얼굴 탐지 및 분류 완료 대기 중...")
        while not (self.face_detection_done and self.classification_done):
            time.sleep(0.1)
        
        self.logger.info("✅ 모든 처리 완료!")
        
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
            self.logger.error(f"❌ 프레임 읽기 오류: {e}")
        finally:
            cap.release()
            # 종료 신호
            self.frame_queue.put(None)
            self.logger.info("📹 프레임 읽기 워커 종료")
    
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
            self.logger.error(f"❌ 얼굴 탐지 오류: {e}")
        finally:
            self.face_detection_done = True
            self.logger.info("✅ 얼굴 탐지 완료")
    
    def _process_face_batch(self, frame_batch: List[Dict]):
        """프레임 배치에서 얼굴 탐지"""
        batch_start_time = time.time()
        faces_in_batch = 0
        
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
                        faces_in_batch += 1
                
                self.stats['frames_processed'] += 1
                
            except Exception as e:
                self.logger.warning(f"⚠️ 프레임 {frame_data['frame_number']} 처리 실패: {e}")
        
        batch_time = time.time() - batch_start_time
        self.stats['faces_detected'] += faces_in_batch
        
        # 배치 단위 로깅
        if self.config['logging']['batch_summary']:
            self.logger.info(
                f"얼굴 탐지 배치: {len(frame_batch)}프레임 → {faces_in_batch}개 얼굴 "
                f"({batch_time:.2f}초)"
            )
    
    def _batch_classification_worker(self):
        """배치 분류 워커"""
        batch_size = self.config['classifier']['batch_size']
        timeout = self.config['classifier']['batch_timeout']
        
        face_batch = []
        last_batch_time = time.time()
        
        try:
            while True:
                try:
                    # 타임아웃 설정으로 얼굴 데이터 가져오기
                    remaining_timeout = max(0.1, timeout - (time.time() - last_batch_time))
                    face_data = self.face_queue.get(timeout=remaining_timeout)
                    
                    if face_data is None:  # 종료 신호
                        # 남은 배치 처리
                        if face_batch:
                            self._process_classification_batch(face_batch)
                        break
                    
                    face_batch.append(face_data)
                    
                    # 배치 처리 조건 확인
                    should_process = (
                        len(face_batch) >= batch_size or  # 배치가 가득 참
                        (self.face_detection_done and len(face_batch) > 0) or  # 탐지 완료 + 남은 배치
                        (time.time() - last_batch_time) >= timeout  # 타임아웃
                    )
                    
                    if should_process:
                        self._process_classification_batch(face_batch)
                        face_batch = []
                        last_batch_time = time.time()
                    
                    self.face_queue.task_done()
                    
                except queue.Empty:
                    # 타임아웃 발생 - 현재 배치 처리
                    if face_batch:
                        if self.config['logging']['batch_summary']:
                            self.logger.info(f"타임아웃으로 배치 처리: {len(face_batch)}개")
                        self._process_classification_batch(face_batch)
                        face_batch = []
                        last_batch_time = time.time()
                
        except Exception as e:
            self.logger.error(f"❌ 분류 워커 오류: {e}")
        finally:
            self.classification_done = True
            self.logger.info("✅ 분류 처리 완료")
    
    def _process_classification_batch(self, face_batch: List[Dict]):
        """분류 배치 처리"""
        if not face_batch:
            return
        
        try:
            # 얼굴 이미지들 추출
            face_images = [face_data['face_image'] for face_data in face_batch]
            
            # 배치 예측
            batch_start_time = time.time()
            predictions = self.classifier.predict_batch(face_images)
            batch_time = time.time() - batch_start_time
            
            self.stats['batch_count'] += 1
            self.stats['total_inference_time'] += batch_time
            
            # 결과 처리
            angry_count = 0
            for face_data, prediction in zip(face_batch, predictions):
                if prediction['is_angry']:
                    angry_moment = {
                        'timestamp': face_data['timestamp'],
                        'frame_number': face_data['frame_number'],
                        'confidence': prediction['confidence']
                    }
                    
                    self.angry_moments.append(angry_moment)
                    self.stats['angry_moments'] += 1
                    angry_count += 1
                    
                    # 킹받는 프레임 저장 (옵션)
                    if self.config['output']['save_highlights']:
                        self._save_highlight_image(face_data, prediction['confidence'])
                    
                    if self.config['debug']['timing_detailed']:
                        timestamp_str = str(timedelta(seconds=int(face_data['timestamp'])))
                        self.logger.info(f"😡 킹받는 순간! {timestamp_str} (신뢰도: {prediction['confidence']:.3f})")
            
        except Exception as e:
            self.logger.error(f"⚠️ 배치 분류 오류: {e}")
    
    def _save_highlight_image(self, face_data: Dict, confidence: float):
        """킹받는 순간 이미지 저장"""
        try:
            timestamp_str = f"{int(face_data['timestamp']):05d}"
            filename = f"angry_{timestamp_str}_{confidence:.3f}.jpg"
            
            save_path = os.path.join(
                self.video_output_dir,
                "highlights",
                filename
            )
            
            face_data['face_image'].save(save_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 하이라이트 이미지 저장 실패: {e}")
    
    def _performance_monitor(self):
        """성능 모니터링"""
        interval = self.config['performance']['monitoring_interval']
        
        while not self.stop_flag:
            time.sleep(interval)
            
            if self.stats['processing_start_time'] is None:
                continue
            
            # 현재 통계
            elapsed = time.time() - self.stats['processing_start_time']
            fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
            
            # 큐 상태
            frame_queue_size = self.frame_queue.qsize()
            face_queue_size = self.face_queue.qsize()
            
            # GPU 메모리 사용량
            memory_info = self.classifier.get_memory_usage()
            
            if self.config['logging']['performance_tracking']:
                avg_batch_time = (self.stats['total_inference_time'] / self.stats['batch_count'] 
                                 if self.stats['batch_count'] > 0 else 0)
                
                self.logger.info(
                    f"📊 [{elapsed:.1f}s] "
                    f"프레임: {self.stats['frames_processed']} ({fps:.1f} FPS), "
                    f"얼굴: {self.stats['faces_detected']}, "
                    f"킹받음: {self.stats['angry_moments']}, "
                    f"큐: {frame_queue_size}/{face_queue_size}, "
                    f"GPU: {memory_info['allocated']:.1f}GB, "
                    f"배치 평균: {avg_batch_time:.3f}초"
                )
        
        self.logger.info("📊 성능 모니터링 종료")
    
    def _save_results(self, video_path: str, video_info: Dict) -> Dict:
        """결과 저장"""
        results = {
            'video_path': video_path,
            'video_info': video_info,
            'processing_stats': self.stats.copy(),
            'angry_moments': self.angry_moments,
            'total_angry_moments': len(self.angry_moments),
            'config': self.config
        }
        
        # 타임스탬프 JSON 저장
        if self.config['output']['save_timestamps']:
            timestamp_file = os.path.join(
                self.video_output_dir,
                "timestamps",
                "angry_moments.json"
            )
            
            with open(timestamp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 결과 저장: {timestamp_file}")
        
        return results
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        elapsed = time.time() - self.stats['processing_start_time']
        fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        avg_batch_time = (self.stats['total_inference_time'] / self.stats['batch_count'] 
                         if self.stats['batch_count'] > 0 else 0)
        
        self.logger.info("🎯 처리 완료!")
        self.logger.info(f"   총 처리 시간: {elapsed:.1f}초")
        self.logger.info(f"   처리된 프레임: {self.stats['frames_processed']}개 ({fps:.1f} FPS)")
        self.logger.info(f"   탐지된 얼굴: {self.stats['faces_detected']}개")
        self.logger.info(f"   킹받는 순간: {self.stats['angry_moments']}개")
        self.logger.info(f"   분류 배치: {self.stats['batch_count']}회 (평균 {avg_batch_time:.3f}초)")
        
        # GPU 메모리 최종 사용량
        memory_info = self.classifier.get_memory_usage()
        self.logger.info(f"   최대 GPU 메모리: {memory_info['max_allocated']:.1f}GB")


def main():
    """메인 실행 함수"""
    import sys
    import argparse
    
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='침착맨 킹받는 순간 탐지 (PyTorch)')
    parser.add_argument('filename', nargs='?', help='처리할 비디오 파일명 (확장자 포함)')
    parser.add_argument('--dir', '--directory', help='비디오 파일 디렉토리 경로')
    parser.add_argument('--config', default='config/config_torch.yaml', help='설정 파일 경로')
    
    args = parser.parse_args()
    
    try:
        # 프로세서 초기화
        processor = TorchVideoProcessor(args.config)
        
        # 비디오 경로 결정
        if args.filename:
            # 명령줄에서 파일명 제공
            video_dir = args.dir if args.dir else processor.config['video']['default_directory']
            video_path = os.path.join(video_dir, args.filename)
        else:
            # config에서 기본값 사용
            video_dir = processor.config['video']['default_directory']
            video_filename = processor.config['video']['default_filename']
            video_path = os.path.join(video_dir, video_filename)
        
        processor.logger.info(f"🎬 처리할 영상: {video_path}")
        
        # 비디오 처리
        results = processor.process_video(video_path)
        
        processor.logger.info("✅ 모든 처리가 완료되었습니다!")
        
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