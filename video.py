import torch
import numpy as np
import cv2
import time
import pygame

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")


class ObjectDetection:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)
        self.model = self.load_model()
        self.classes = self.model.names
        self.alarm_triggered = False  

    def load_model(self):
        model = torch.hub.load('./yolov5', 'custom', path="weapon.pt", source='local')
        model = model.to(self.device).eval()  
        return model

    def score_frame(self, frame):
        frame = torch.from_numpy(frame).to(self.device).float()
        frame = frame.permute(2, 0, 1)  
        frame = frame.unsqueeze(0)  
        results = self.model(frame)  

        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.85:  
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

                if not self.alarm_triggered:
                    alarm_sound.play()
                    self.alarm_triggered = True

                label_and_confidence = f"{self.class_to_label(labels[i])} {row[4] * 100:.1f}%"
                cv2.putText(frame, label_and_confidence, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()

            if not ret:
                break

            frame_resized = cv2.resize(frame, (640, 480))

            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0

            results = self.score_frame(frame_rgb)

            frame_with_boxes = self.plot_boxes(results, frame_resized)

            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame_with_boxes, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow("img", frame_with_boxes)

            if results[0].size == 0:  
                self.alarm_triggered = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

detection = ObjectDetection()
detection()
