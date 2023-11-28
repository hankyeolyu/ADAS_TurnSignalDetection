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

#우측전방 carla6
#좌측전방 carla1
#좌측후방 carla2
#우측후방 carla3

# 동영상 파일 재생을 위한 VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

threshold = 50
f_threshold = 50
b_threshold = 100
threshold_upper_bound = 1200

def warning_audio(audio_path):

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

    # 객체 감지 및 분류 로직 작성
    # 노란색 방향 지시등을 감지하기 위해 색상 범위를 지정합니다.
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # BGR 이미지를 HSV 컬러 스페이스로 변환합니다.
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # 노란색 범위에 해당하는 픽셀을 마스크로 생성합니다.
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

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

        conf_threshold = 0.5
        # bounding box 그리기
        if confidence > conf_threshold:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 객체 레이블과 신뢰도 표시
        label = class_labels[int(class_index)]
        if confidence > conf_threshold:
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if label.startswith('light1_1'):
            # 객체 내부 사각형 추출
            object_roi = frame[y1:y2, x1:x2]
            # 방향 지시등 인식 수행
            object_roi_with_traffic_sign, yellow_mask_roi = detect_traffic_sign(object_roi)
            # 원본 이미지에 방향 지시등 인식 결과 반영
            frame[y1:y2, x1:x2] = object_roi_with_traffic_sign
            
            current_time = time.time()
            yellow_pixel_count = cv2.countNonZero(yellow_mask_roi)
            if yellow_pixel_count > f_threshold and yellow_pixel_count < threshold_upper_bound and current_time - last_detection_time > 2.5:
                audio_path = 'C:\project\warning_image\warning1.mp3'
                warning_audio(audio_path)
                last_detection_time = current_time
                print("우측 전방 차량 인식")
                continue

        if label.startswith('light2_1'):
            # 객체 내부 사각형 추출
            object_roi = frame[y1:y2, x1:x2]
            # 방향 지시등 인식 수행
            object_roi_with_traffic_sign, yellow_mask_roi = detect_traffic_sign(object_roi)
            # 원본 이미지에 방향 지시등 인식 결과 반영
            frame[y1:y2, x1:x2] = object_roi_with_traffic_sign

            current_time = time.time()
            yellow_pixel_count = cv2.countNonZero(yellow_mask_roi)
            if yellow_pixel_count > f_threshold and yellow_pixel_count < threshold_upper_bound and current_time - last_detection_time > 2.5:
                audio_path = 'C:\project\warning_image\warning2.mp3'
                warning_audio(audio_path)
                last_detection_time = current_time
                print("좌측 전방 차량 인식")
                continue

        if label.startswith('light3_1'):
            # 객체 내부 사각형 추출
            object_roi = frame[y1:y2, x1:x2]
            # 방향 지시등 인식 수행
            object_roi_with_traffic_sign, yellow_mask_roi = detect_traffic_sign(object_roi)
            # 원본 이미지에 방향 지시등 인식 결과 반영
            frame[y1:y2, x1:x2] = object_roi_with_traffic_sign

            current_time = time.time()
            yellow_pixel_count = cv2.countNonZero(yellow_mask_roi)
            if yellow_pixel_count > b_threshold and yellow_pixel_count < threshold_upper_bound and current_time - last_detection_time > 2.5:
                audio_path = 'C:\project\warning_image\warning3.mp3'
                warning_audio(audio_path)
                last_detection_time = current_time
                print("좌측 후방 차량 인식")
                continue

        if label.startswith('light4_1'):
            # 객체 내부 사각형 추출
            object_roi = frame[y1:y2, x1:x2]
            # 방향 지시등 인식 수행
            object_roi_with_traffic_sign, yellow_mask_roi = detect_traffic_sign(object_roi)
            # 원본 이미지에 방향 지시등 인식 결과 반영
            frame[y1:y2, x1:x2] = object_roi_with_traffic_sign

            current_time = time.time()
            yellow_pixel_count = cv2.countNonZero(yellow_mask_roi)
            if yellow_pixel_count > b_threshold and yellow_pixel_count < threshold_upper_bound and current_time - last_detection_time > 2.5:
                audio_path = 'C:\project\warning_image\warning4.mp3'
                warning_audio(audio_path)
                last_detection_time = current_time
                print("우측 후방 차량 인식")
                continue

    # 동영상 파일 재생
    cv2.imshow('Video', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 동영상 파일 재생 종료
cap.release()
cv2.destroyAllWindows()
