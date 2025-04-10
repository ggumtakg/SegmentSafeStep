import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import csv
from segment_anything import sam_model_registry, SamPredictor

# ========================
# 설정
# ========================
image_dir = r"1000_dataset_polygon_alley\polygon_alley"
mask_dir = r"1000_dataset_polygon_alley\masked_polygon_alley"
os.makedirs(mask_dir, exist_ok=True)

checkpoint = "sam_vit_h_4b8939.pth"  # SAM 모델 체크포인트
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

# SAM 모델 로드
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 이미지 목록 가져오기
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

# 클릭 좌표 저장용 CSV
csv_path = os.path.join(mask_dir, "click_points.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# 헤더 작성 (최대 10점)
header = ["filename"] + [f"x{i+1}" for i in range(10)] + [f"y{i+1}" for i in range(10)]
csv_writer.writerow(header)

# ========================
# 메인 반복 처리
# ========================
for fname in image_files:
    image_path = os.path.join(image_dir, fname)
    image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)

    print(f"\n 이미지: {fname} - 창에서 이면도로를 클릭하세요 (점 1~10개).\n닫으면 마스크가 생성됩니다.")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(f"{fname} - 이면도로 위 클릭 (최대 점 10개)")

    clicked_points = plt.ginput(n=10, timeout=0)  # 무제한 대기, 최대 점 10개
    plt.close()

    if len(clicked_points) == 0:
        print("클릭된 포인트 없음. 다음 이미지로 넘어감.")
        continue

    input_points = np.array([[int(x), int(y)] for x, y in clicked_points])
    input_labels = np.ones(len(input_points))  # foreground

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    # 마스크 저장
    save_mask = (masks[0] * 255).astype(np.uint8)
    mask_name = os.path.splitext(fname)[0] + ".png"
    mask_path = os.path.join(mask_dir, mask_name)
    Image.fromarray(save_mask).save(mask_path)
    print(f"마스크 저장됨: {mask_path}")

    # 좌표 저장 (x1, x2, ..., y1, y2, ...)
    flat_points = input_points.flatten().tolist()
    while len(flat_points) < 20:  # 총 점 10 = 100좌표
        flat_points.append("")  # 빈칸 채움
    x_coords = flat_points[0::2]
    y_coords = flat_points[1::2]
    csv_writer.writerow([fname] + x_coords + y_coords)

csv_file.close()
print("\n완료")
