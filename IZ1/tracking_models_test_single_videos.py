import cv2
import sys
import os

def list_files(directory):
    filelist = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            filelist.append(os.path.join(root, file))
    return filelist

TRACKER_TYPES = ['KCF', 'MEDIANFLOW', 'MOSSE']
file = 'IZ1/video/drift2.mp4'
tracker_type = 0

if __name__ == '__main__':
    video = cv2.VideoCapture(file)
    fps = video.get(cv2.CAP_PROP_FPS)
    codec = video.get(cv2.CAP_PROP_FOURCC)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    ret, frame = video.read()
    bbox = cv2.selectROI(frame, False)
    
    if tracker_type not in [0, 1, 2]:
        print("Invalid tracker type")
        sys.exit()
    elif tracker_type == 0:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 1:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 2:
        tracker = cv2.legacy.TrackerMOSSE_create()

    output_dir = f"IZ1/results/tracking"
    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(f"{output_dir}/{file[10:-4]}_{TRACKER_TYPES[tracker_type]}.avi", fourcc, fps, (int(width), int(height)))
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ret = tracker.init(frame, bbox)

    while True:
        ret, frame = video.read()
        if not ret:
            break
        timer = cv2.getTickCount()
        ret, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, TRACKER_TYPES[tracker_type] + " Tracker", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "FPS: " + str(int(fps)), (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        output.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    output.release()
    video.release()