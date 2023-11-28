import cv2
import numpy as np

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


# 비디오 열기
video_capture = cv2.VideoCapture('C:\project\DataSet/carla4.mp4')

while True:
    # 프레임 읽기
    ret, frame = video_capture.read()

    if not ret:
        break

    # 방향 지시등 인식
    result = detect_traffic_sign(frame)

    # 화면에 출력
    cv2.imshow('Video', result)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 비디오 종료
video_capture.release()
cv2.destroyAllWindows()

