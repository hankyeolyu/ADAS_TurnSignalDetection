import cv2
import torch
import numpy as np
from yolov5 import YOLOv5
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = YOLOv5('C:\project\lane_change_warning_system/best.pt')

class_labels = ['car', 'car_1', 'car_2', 'car_3', 'car_4', 'light', 'light_1', 'light_2', 'light_3', 'light_4']

def send_notifictation(object_class1, object_class2):
    if object_class1 == 'car':
        object_class1 = 'car_1'
    if object_class1 == 'light':
        object_class1 = 'light_1'
    if object_class2 == 'car':
        object_class2 = 'car_1'
    if object_class2 == 'light':
        object_class2 = 'light_1'
    if object_class1 == 'car_1':
        if object_class2 == 'light_1':
            print("우측 전방 인식")
    elif object_class1 == 'car_2':
        if object_class2 == 'light_2':
            print("좌측 전방 인식")
    elif object_class1 == 'car_3':
        if object_class2 == 'light_3':
            print("좌측 후방 인식")
    elif object_class1 == 'car_4':
        if object_class2 == 'light_4':
            print("우측 후방 인식")
    elif object_class1 == 'light_1':
        if object_class2 == 'car_1':
            print("우측 전방 인식")
    elif object_class1 == 'light_2':
        if object_class2 == 'car_2':
            print("좌측 전방 인식")
    elif object_class1 == 'light_3':
        if object_class2 == 'car_3':
            print("좌측 후방 인식")
    elif object_class1 == 'light_4':
        if object_class2 == 'car_4':
            print("우측 후방 인식")
def draw_results(frame, predictions, predicted_classes):
    _, _, h, w = predictions.shape

    for idx, obj_class in enumerate(predicted_classes):
        # (예시) 객체의 경계 상자 좌표를 가져와서 사각형 그리기
        box = predictions[0, idx, :4] * torch.tensor([w, h, w, h])
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 객체 클래스 레이블 텍스트 작성
        label = obj_class

        # 텍스트 배경 상자 그리기
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1 = max(y1, label_size[1])
        cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (0, 255, 0), cv2.FILLED)

        # 텍스트 그리기
        cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame
def preprocess(frame):
    # (예시) 필요한 전처리 작업을 수행하고 모델에 입력할 수 있는 형태로 변환
    # PyTorch의 경우, 텐서로 변환하고 채널 순서를 변경해야 할 수도 있습니다.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame.transpose((2, 0, 1))).float()
    return frame.unsqueeze(0)
def get_video_stream():
    return cv2.VideoCapture("C:\project\DataSet\car_data.mp4")
def object_detection():
    video_stream = get_video_stream()

    while True:
        ret, frame = video_stream.read()
        if not ret:
            break
        preprocessed_frame = preprocess(frame)

        with torch.no_grad():
            predictions = model.predict(preprocessed_frame)
        _, predicted_classes = predictions.topk(2)
        predicted_classes = [class_labels[idx.item()] for idx in predicted_classes]
        predicted_class1, predicted_class2 = predicted_classes
        send_notifictation(predicted_class1, predicted_class2)
        
        frame = draw_results(frame, predictions, predicted_classes)
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    video_stream.release()
    cv2.destroyAllWindows()

object_detection()
        


