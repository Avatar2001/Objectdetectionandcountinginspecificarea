import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Load YOLOv11 model
model = YOLO('yolo11s.pt')
names = model.names  # Get class names

# Open video file
cap = cv2.VideoCapture('pcount.mp4')

# Get width and height of video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer to save output
output_video_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))  # 20 is number of frames per second

# Define the polygon points for the region of interest
area = [(524, 282), (152, 331), (243, 459), (834, 427)]

# Main loop to process each frame in the video
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break  # Break the loop if no frame is read

    # Resize frame to fixed size
    frame = cv2.resize(frame, (1020, 500))

    # Get the object detection results from YOLO model
    results = model.track(frame, persist=True)

    # Persons count in ROI
    person_count = 0

    # Check if there are any boxes and IDs in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class, ID, track ID, and confidence
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding box coordinates
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Predicted classes
        track_ids = results[0].boxes.id.cpu().tolist()  # Track IDs
        confidence = results[0].boxes.conf.cpu().tolist()  # Confidence scores

        # Loop through each detected object
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidence):
            c = names[class_id]  # Get the class name (e.g., 'person', 'car', etc.)
            x1, y1, x2, y2 = box  # Get the bounding box coordinates

            # Check if the object is a person
            if 'person' in c:
                # Calculate the center of the bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check if the center of the bounding box is inside the polygon
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:  # If inside the polygon
                    person_count += 1  # Increment the person count

                    # Draw the bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Display class name and track ID
                    label = f'{c} {track_id}'
                    cvzone.putTextRect(frame, label, (x1, y1 - 10), scale=1, thickness=2, colorR=(0, 255, 0))

                    # Draw a circle at the center of the bounding box
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    # Draw the polygon on the frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)

    # Display the person count
    cvzone.putTextRect(frame, f'Person Count: {person_count}', (10, 30), scale=2, thickness=3, colorR=(0, 0, 255))

    # Write the frame to the output video
    video_writer.write(frame)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
video_writer.release()  # Release the video writer to save the output video
cv2.destroyAllWindows()
