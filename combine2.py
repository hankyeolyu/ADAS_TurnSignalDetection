import cv2
import torch
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\project\lane_change_warning_system/best.pt')

# 객체 목록 및 클래스 레이블
class_labels = ['car', 'car_1', 'car_2', 'car_3', 'car_4', 'light', 'light_1', 'light_2', 'light_3', 'light_4']

# 동영상 파일 경로
video_path = 'C:\project\DataSet\carla4.mp4'

# 동영상 파일 재생을 위한 VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

# 방향 지시등 검출을 위한 변수 초기화
prev_yellow_mask = None
prev_yellow_detected = False
yellow_count = 0
blink_threshold = 2
blink_interval = 5  # 프레임 간의 시간 간격 (단위: 프레임)

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

        # bounding box 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 객체 레이블과 신뢰도 표시
        label = class_labels[int(class_index)]
        cv2.putText(frame, f'{label}: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 객체 인식 메시지 출력
        if label == 'car_1':
            print("우측 전방 차량 인식")
        elif label == 'car_2':
            print("좌측 전방 차량 인식")
        elif label == 'car_3':
            print("좌측 후방 차량 인식")
        elif label == 'car_4':
            print("우측 후방 차량 인식")

    # 방향 지시등 검출
    bgr_frame = frame

    # 노란색 방향 지시등을 감지하기 위해 색상 범위를 지정합니다.
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # BGR 이미지를 HSV 컬러 스페이스로 변환합니다.
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # 노란색 범위에 해당하는 픽셀을 마스크로 생성합니다.
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # 이전 프레임과 비교하여 노란색 영역 검출 여부 확인
    if prev_yellow_mask is not None:
        yellow_diff = cv2.absdiff(yellow_mask, prev_yellow_mask)
        yellow_diff_count = np.count_nonzero(yellow_diff)

        if yellow_diff_count > 0:
            if not prev_yellow_detected:
                yellow_count += 1
                prev_yellow_detected = True
        else:
            prev_yellow_detected = False

    # 현재 프레임의 노란색 영역을 이전 프레임에 저장
    prev_yellow_mask = yellow_mask.copy()

    # 일정 시간 간격으로 깜빡이며 2회 검출되면 방향 지시등 인식 메시지 출력
    if yellow_count >= blink_threshold and yellow_count % blink_interval == 0:
        print("방향 지시등 인식")

    # 프레임 출력
    cv2.imshow('Real-time Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 종료 시 리소스 해제
cap.release()
cv2.destroyAllWindows()

