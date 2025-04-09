import os
import cv2
import torch
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append("C:/Users/hayou/PycharmProjects/robot_segmentation")
from models.bisenetv2 import BiSeNetV2

# 경로 설정
video_path = r"C:\Users\hayou\PycharmProjects\robot_segmentation\test_video.mp4"
model_path = r"C:\Users\hayou\PycharmProjects\robot_segmentation\best_model_finetuned_epoch12.pth"
output_path = r"C:\Users\hayou\PycharmProjects\robot_segmentation\output_segmented.mp4"

walk_icon_path = "C:/Users/hayou/PycharmProjects/robot_segmentation/icons/walk_icon.png"
arrow_icon_path = "C:/Users/hayou/PycharmProjects/robot_segmentation/icons/arrow_icon.png"

walk_icon = cv2.imread(walk_icon_path, cv2.IMREAD_UNCHANGED)
arrow_icon = cv2.imread(arrow_icon_path, cv2.IMREAD_UNCHANGED)

# 아이콘 크기 조정
icon_size = (50, 50)
if walk_icon is not None:
    walk_icon = cv2.resize(walk_icon, icon_size)
if arrow_icon is not None:
    arrow_icon = cv2.resize(arrow_icon, icon_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = (0.56, 0.56, 0.54)
std = (0.20, 0.20, 0.19)
image_size = 352

model = BiSeNetV2(num_class=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = Compose([
    Resize(image_size, image_size),
    Normalize(mean=mean, std=std),
    ToTensorV2()
])

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🎞️ 세그멘테이션 영상 생성 중...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = transform(image=rgb)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(processed)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (prob > 0.5).astype(np.uint8)

    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # 초록색 영역 만들기
    green_mask = np.zeros_like(frame)
    green_mask[mask == 1] = (0, 255, 0)

    # 경계 추출
    kernel = np.ones((100, 100), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    border = mask - eroded
    yellow_mask = np.zeros_like(frame)
    yellow_mask[border == 1] = (0, 255, 255)

    # 합성
    combined_mask = cv2.add(green_mask, yellow_mask)
    overlay = cv2.addWeighted(frame, 0.7, combined_mask, 0.3, 0)

    border_area = np.sum(border)
    if border_area > 1000:
        green_coords = np.column_stack(np.where(mask == 1))

        if green_coords.size > 0:
            # 하단 중앙 좌표 계산
            bottom_y = np.max(green_coords[:, 0])
            candidates = green_coords[green_coords[:, 0] >= bottom_y - 10]
            center_x = int(np.mean(candidates[:, 1]))

            # 텍스트 설정
            text = "[Safe path detected]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 3
            text_color = (255, 20, 147)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            text_x = center_x - text_size[0] // 2
            text_y = bottom_y - 10

            # 텍스트 배경 박스
            box_x1 = text_x - 10
            box_y1 = text_y - text_size[1] - 10
            box_x2 = text_x + text_size[0] + 10
            box_y2 = text_y + 10
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)

            # 텍스트 그리기
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            # 아이콘 위치 계산
            if walk_icon is not None and arrow_icon is not None:
                icon_w, icon_h = icon_size
                icon_center_x = center_x - icon_w // 2

                walk_y1 = box_y1 - icon_h - 10
                walk_y2 = walk_y1 + icon_h
                arrow_y1 = walk_y1 - icon_h - 10
                arrow_y2 = arrow_y1 + icon_h

                # walk icon
                icon_rgb = walk_icon[:, :, :3]
                icon_alpha = walk_icon[:, :, 3] / 255.0
                for c in range(3):
                    overlay[walk_y1:walk_y2, icon_center_x:icon_center_x+icon_w, c] = \
                        icon_alpha * icon_rgb[:, :, c] + \
                        (1 - icon_alpha) * overlay[walk_y1:walk_y2, icon_center_x:icon_center_x+icon_w, c]

                # arrow icon
                icon_rgb = arrow_icon[:, :, :3]
                icon_alpha = arrow_icon[:, :, 3] / 255.0
                for c in range(3):
                    overlay[arrow_y1:arrow_y2, icon_center_x:icon_center_x+icon_w, c] = \
                        icon_alpha * icon_rgb[:, :, c] + \
                        (1 - icon_alpha) * overlay[arrow_y1:arrow_y2, icon_center_x:icon_center_x+icon_w, c]
    else:
        cv2.putText(overlay, "⚠️ Caution! Narrow walkable area", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if arrow_icon is not None:
            icon_rgb = arrow_icon[:, :, :3]
            icon_alpha = arrow_icon[:, :, 3] / 255.0
            y1, y2 = 10, 60
            x1, x2 = width - 60, width - 10
            for c in range(3):
                overlay[y1:y2, x1:x2, c] = icon_alpha * icon_rgb[:, :, c] + \
                                           (1 - icon_alpha) * overlay[y1:y2, x1:x2, c]

    out.write(overlay)
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"🖼️ {frame_count} 프레임 처리 중...")

cap.release()
out.release()
print(f"✅ 완료! 결과 저장: {output_path}")
