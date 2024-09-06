from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLOv8x-worldv2 model
model = YOLO('yolov8x-worldv2.pt')

# Path to the image
image_path = '/home/neuromorph/htn/dataset/DSEC/htn_trial/event_image/yoloworld_trial/000000.png'

# Perform inference on the image
results = model.predict(image_path)

# Display the image with bounding boxes
image = cv2.imread(image_path)
for box in results[0].boxes.data:
    x1, y1, x2, y2, conf, cls = box
    label = model.names[int(cls)]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    cv2.putText(image, f'{label} {conf:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
