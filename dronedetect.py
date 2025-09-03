import cv2
from ultralytics import YOLO
import datetime
import numpy as np

model = YOLO('yolov10n.pt')

#(use 4 for drone cam, 0 for webcam) # PC specific
cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

screen_width = 1920  
screen_height = 1080  

cv2.namedWindow('Drone', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Drone', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 30 #Adjust based on camera capability

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%d-%m-%Y_%H-%M-%S")
filename = f"recordings/{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(
        source=frame,
        conf=0.6,  #Confidence threshold for detection
        verbose=False  #Don't print results to console
    )
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{cls_name} {conf:.2f}"
            
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            
            cv2.putText(frame, label, (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    aspect_ratio = frame_width / frame_height
    if screen_width / screen_height > aspect_ratio:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
    
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    black_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    y_offset = (screen_height - new_height) // 2
    x_offset = (screen_width - new_width) // 2
    black_bg[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    video_writer.write(frame)
    cv2.imshow('Drone', black_bg)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()