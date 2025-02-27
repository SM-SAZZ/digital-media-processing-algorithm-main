import cv2
import sys
import os
import numpy as np

if __name__ == '__main__':
    cap = cv2.VideoCapture("IZ1/video/drift2.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cap.get(cv2.CAP_PROP_FOURCC)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    output_dir = "IZ1/results/modifications"
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_gray = cv2.VideoWriter(f"{output_dir}/GRAY.mp4", fourcc, fps, (int(width), int(height)),False)
    output_blurred = cv2.VideoWriter(f"{output_dir}/BLUR.mp4", fourcc, fps, (int(width), int(height)))
    output_contrast = cv2.VideoWriter(f"{output_dir}/CONTRAST.mp4", fourcc, fps, (int(width), int(height)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(frame, (15, 15), 100)
        contrast_frame = cv2.addWeighted(frame, 2.5, np.zeros_like(frame), 0, 0)
        output_gray.write(gray)
        output_blurred.write(blurred_frame)
        output_contrast.write(contrast_frame)
    cap.release()
    output_gray.release()
    output_contrast.release()
    output_blurred.release()
    cv2.destroyAllWindows()

    print("Обработка завершена!")
