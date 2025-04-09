import sys
import time
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QPushButton, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# BiSeNetV2 구조 불러오기 (경로에 맞게 import)
from model.BiSeNet.lib.models.bisenetv2 import BiSeNetV2


class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("보행가능구역-bisenet")
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # UI 구성 요소
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["카메라 0", "카메라 1", "카메라 2"])
        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.toggle_camera)
        self.fps_label = QLabel("FPS: 0.00")

        self.orig_label = QLabel()
        self.orig_label.setFixedSize(1024, 1024)
        self.seg_label = QLabel()
        self.seg_label.setFixedSize(1024, 1024)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.camera_selector)
        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.fps_label)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.orig_label)
        image_layout.addWidget(self.seg_label)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

        # 모델 로드
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        self.model = self.load_model()

    def load_model(self):
        model = BiSeNetV2(n_classes=19)
        checkpoint = torch.load("model_final_v2_city.pth", map_location='cpu')
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def toggle_camera(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.capture:
                self.capture.release()
            self.start_button.setText("시작")
        else:
            camera_index = self.camera_selector.currentIndex()
            self.capture = cv2.VideoCapture(camera_index)
            self.timer.start(30)  # 약 30fps
            self.start_button.setText("중지")

    def update_frame(self):
        start_time = time.time()

        ret, frame = self.capture.read()

        frame_height, frame_width, _ = frame.shape

        if not ret:
            return

        input_tensor = self.preprocess(frame).to(self.device)
        #print(f"input_tensor shape: {input_tensor.shape}")

        with torch.no_grad():
            output = self.model(input_tensor)[0]
            seg_map = output.argmax(dim=1)[0].byte().cpu().numpy()

            #print(f"model output shape: {output.shape if output is not None else 'None'}")
            #print(f"seg_map shape: {seg_map.shape}, dtype: {seg_map.dtype}")

        # 결과를 grayscale로 normalize
        #seg_image = cv2.normalize(seg_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        seg_image = cv2.cvtColor(seg_map, cv2.COLOR_GRAY2BGR)
        orig_frame = cv2.resize(frame, (frame_width, frame_height))
        seg_image = cv2.resize(seg_image, (frame_width, frame_height))

        # 이미지 보여주기
        self.display_image(orig_frame, self.orig_label)
        self.display_image(seg_image, self.seg_label)
        #print(f"orig_frame shape: {orig_frame.shape}, dtype: {orig_frame.dtype}")
        #print(f"seg_map shape: {seg_image.shape}, dtype: {seg_image.dtype}")

        # FPS 계산
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        self.fps_label.setText("FPS: {:.2f}".format(fps))

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 512))  # BiSeNetV2의 기본 입력 크기
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img).unsqueeze(0)
        return tensor

    def display_image(self, img, label):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec_())