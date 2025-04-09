import cv2
import os

# 이미지가 저장된 폴더 경로
folder_path = 'demoVideo/stuttgart_00'

# 이미지 파일들을 정렬된 순서로 불러오기
image_files = sorted([
    f for f in os.listdir(folder_path)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

window_name = "Image Sequence Player"
frame_delay = 30  # 재생 속도 (ms)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

paused = False
index = 0

while True:
    if not paused and index < len(image_files):
        image_path = os.path.join(folder_path, image_files[index])
        img = cv2.imread(image_path)

        if img is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            index += 1
            continue

        cv2.imshow(window_name, img)
        index += 1

    key = cv2.waitKey(frame_delay) & 0xFF

    if key == 27:  # ESC 키
        cv2.destroyAllWindows()
        break
    elif key == ord(' '):  # 스페이스바: 일시정지 / 재생
        paused = not paused
    elif key == ord('r'):  # r 키: 처음으로
        index = 0
        paused = False