import cv2
from ultralytics import YOLO

gravar = 0
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "video_416_p.avi"
cap = cv2.VideoCapture(video_path)

if gravar:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('./note_416_41fps.avi', fourcc, 24.0, (416, 416))
    
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, imgsz=416, conf=0.3)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if gravar:
            out.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
if gravar:
    out.release()
cv2.destroyAllWindows()
