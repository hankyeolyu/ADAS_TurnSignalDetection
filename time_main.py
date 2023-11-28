import cv2
import torch
import numpy as np
import pygame
import time

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\project\lane_change_warning_system/best.pt')

# 객체 목록 및 클래스 레이블
class_labels = ['car_1', 'car_2', 'car_3', 'car_4', 'light1_1', 'light1_2', 'light2_1', 'light2_2', 'light3_1', 'light3_2', 'light4_1', 'light4_2']

# 동영상 파일 경로
video_path = 'C:\project\DataSet/carla6.mp4'

# 동영상 파일 재생을 위한 VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

threshold = 80
threshold_upper_bound = 1200

def warning_image(image_path):
    # 이미지 파일 읽기
    image = cv2.imread(image_path)

    # 이미지 창 생성 및 이미지 표시
    cv2.imshow('Image', image)

    # 키 입력 대기
    cv2.waitKey(0)

    # 이미지 창 닫기
    cv2.destroyAllWindows()

def warning_audio():
    # 오디오 파일 경로
    audio_path = 'C:\project\warning_image\warning.mp3'

    # 오디오 초기화
    pygame.mixer.init()

    # 오디오 파일 로드
    pygame.mixer.music.load(audio_path)

    # 오디오 재생
    pygame.mixer.music.play()

    # 재생 종료 대기
    while pygame.mixer.music.get_busy():
        continue

# 방향 지시등 인식 함수
def detect_traffic_sign(frame):
    # BGR 컬러 스페이스로 변환
    bgr_frame = frame

    # 객체 위치 식별
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 경계 상자 그리기
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h  # 영역 계산
        if area > threshold and area < threshold_upper_bound:  # 일정 영역 이상인 경우에만 인식
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, yellow_mask

# 타이머 초기화
last_detection_time = time.time()

# 동영상 파일 프레임 처리
while cap.isOpened():
    # 영상 프레임 읽기
    ret, frame = cap.read()

    # 동영상 파일 재생이 끝나면 종료
    if not ret:
        break

    # YOLOv5를 사용하여 객체 인식 수행
    results = model(frame)
    
    # 객체 감지 결과에 bounding box 그리기
    for bbox in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_index = bbox
        x1, y1, x2, y2 = round(x1.item()), round(y1.item()), round(x2.item()), round(y2.item())
        class_label = class_labels[int(class_index)]
        
        # 방향 지시등 객체에 대한 처리
        if class_label == 'light1_1':
            # 방향 지시등 영역 추출을 위해 이미지를 HSV 컬러 스페이스로 변환
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 노란색 범위에 해당하는 마스크 생성
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)

            # 방향 지시등 인식 함수 호출
            frame, yellow_mask = detect_traffic_sign(frame)

            # 타이머를 사용하여 경고 사운드 제어
            current_time = time.time()
            if current_time - last_detection_time > 2:  # 마지막 인식 이후로 2초 이상 경과한 경우
                # 경고 이미지 및 사운드 재생
                #warning_image('C:\project\warning_image\warning.jpg')
                warning_audio()

                # 타이머 업데이트
                last_detection_time = current_time
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 프레임 표시
    cv2.imshow('Frame', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 동영상 파일과 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()
