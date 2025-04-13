import cv2
import time
import torch
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from bise.bise.models.bisenetv2 import BiSeNetv2 as BiSeNetV2
from model.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 모델 설정
MODEL_PATH = "best_model_finetuned_epoch12.pth"
DEPTH_MODEL_PATH = "depth_anything_v2_vits.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리용 transform
transform = A.Compose([
    A.Resize(352, 352),
    A.Normalize(mean=(0.56, 0.56, 0.54), std=(0.20, 0.19, 0.19)),
    ToTensorV2()
])

def preprocess(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    return image_tensor

def postprocess(mask):
    mask = (torch.sigmoid(mask) > 0.5).cpu().numpy().astype(np.uint8)[0, 0]
    return (mask * 255).astype(np.uint8)

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Segmentation with BiSeNetV2, Depth-Anything-V2")

        # 카메라 선택
        self.camera_index = tk.IntVar()
        self.camera_selector = ttk.Combobox(root, values=list(range(3)))
        self.camera_selector.current(0)
        self.camera_selector.pack(pady=5)

        # 버튼
        self.start_btn = ttk.Button(root, text="Start", command=self.start)
        self.start_btn.pack(pady=5)

        self.stop_btn = ttk.Button(root, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

        # FPS 라벨
        self.fps_label = ttk.Label(root, text="FPS: 0.00")
        self.fps_label.pack(pady=5)

        # 비디오 캔버스
        self.canvas = tk.Canvas(root, width=1056, height=352)
        self.canvas.pack()

        # 상태 플래그
        self.running = False

        # 모델 로드
        self.model = BiSeNetV2(num_class=1).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vits'

        self.depth_model = DepthAnythingV2(**model_configs[encoder])
        self.depth_model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location=DEVICE))
        self.depth_model.to(DEVICE).eval()

    def get_dynamic_kernel_size(self, depth_value, max_size=51, scale=3):
        return max(3, min(max_size, int(depth_value * scale)))

    def apply_dynamic_dilation(self, edge_mask, depth_map, max_size=51, scale=2):    
        # 최종 결과 이미지 (엣지 크기 조정 결과)
        dilated_result = np.zeros_like(edge_mask, dtype=np.uint8)
    
        # 엣지 픽셀 위치 가져오기
        edge_points = np.column_stack(np.where(edge_mask > 0))
        height, width = edge_mask.shape
        valid_points = edge_points[
            (edge_points[:, 0] > 10) & (edge_points[:, 0] < height - 10) &  # y 좌표 필터링
            (edge_points[:, 1] > 10) & (edge_points[:, 1] < width - 10)    # x 좌표 필터링
        ]

        for y, x in valid_points:
            depth_value = depth_map[y, x]  # 현재 픽셀의 Depth 값 가져오기
            kernel_size = self.get_dynamic_kernel_size(depth_value, max_size, scale)

            # 커널 생성 (홀수 크기로 보정)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            # 현재 위치에서 팽창 수행
            roi_y1, roi_y2 = max(0, y - kernel_size // 2), min(edge_mask.shape[0], y + kernel_size // 2 + 1)
            roi_x1, roi_x2 = max(0, x - kernel_size // 2), min(edge_mask.shape[1], x + kernel_size // 2 + 1)

            dilated_result[roi_y1:roi_y2, roi_x1:roi_x2] = 255  # 팽창 적용

        return dilated_result

    def start(self):
        self.cap = cv2.VideoCapture(int(self.camera_selector.get()))
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.update, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            start_time = time.time()

            # 원본 백업
            orig = cv2.resize(frame.copy(), (352, 352))

            # 전처리 및 예측
            with torch.no_grad():
                input_tensor = preprocess(orig)
                output = self.model(input_tensor)
                mask = postprocess(output)
                edges = cv2.Canny(mask, 100, 100)

                depth_mask = self.depth_model.infer_image(orig)
                dilated_road_mask = self.apply_dynamic_dilation(edges, depth_mask, 101, 5)
                dilated_road_mask = cv2.bitwise_and(dilated_road_mask, mask)

            # 시각화
            seg_rgb = cv2.cvtColor(dilated_road_mask, cv2.COLOR_GRAY2RGB)
            seg_rgb = cv2.resize(seg_rgb, (352, 352))
            blended = cv2.addWeighted(seg_rgb, 0.5, orig, 0.5, 0)
            combined = np.hstack((orig, seg_rgb, blended))

            # FPS 측정
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            self.fps_label.config(text=f"FPS: {fps:.2f}")

            # GUI 업데이트
            image = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()