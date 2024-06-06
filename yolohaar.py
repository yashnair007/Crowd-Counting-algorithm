import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to load YOLO model
def load_yolo_model(cfg_path, weights_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
    return net, output_layers

# Function to apply YOLO
def apply_yolo(net, output_layers, image, confidence_threshold=0.5, nms_threshold=0.4):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    person_boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > confidence_threshold:  # Class ID for person in YOLO
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                person_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    if person_boxes:
        confidences = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(person_boxes, confidences, confidence_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            person_boxes = [person_boxes[i] for i in indices]
    
    return person_boxes, confidences

# Function to apply Haar Cascade
def apply_haar_cascade(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

# Function to draw bounding boxes
def draw_boxes(image, boxes, color, label):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Main function to load image and run both models
def main(image_path, yolo_cfg_path, yolo_weights_path, haar_cascade_path):
    image = cv2.imread(image_path)
    
    
    yolo_net, yolo_output_layers = load_yolo_model(yolo_cfg_path, yolo_weights_path)
    yolo_person_boxes, yolo_confidences = apply_yolo(yolo_net, yolo_output_layers, image)
    yolo_person_count = len(yolo_person_boxes)
    print(f"YOLO detected {yolo_person_count} people.")
    
    
    draw_boxes(image, yolo_person_boxes, (255, 0, 0), "YOLO Person")
    
    
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
  
    haar_person_boxes = apply_haar_cascade(image, face_cascade)
    haar_person_count = len(haar_person_boxes)
    print(f"Haar Cascade detected {haar_person_count} people.")
    
    
    draw_boxes(image, haar_person_boxes, (0, 255, 0), "Haar Face")
    
    
    combined_count = max(yolo_person_count, haar_person_count)
    print(f"Combined detected {combined_count} people.")
    
    
    output_image_path = "detection_results.jpg"
    cv2.imwrite(output_image_path, image)
    
    
    plt.figure(figsize=(10, 7))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Detected People')
    plt.show()
    
    
    plt.figure(figsize=(8, 5))
    plt.hist(yolo_confidences, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Confidence Scores')
    plt.grid(True)
    plt.show()
    
    
    labels = ['YOLO', 'Haar Cascade', 'Combined']
    counts = [yolo_person_count, haar_person_count, combined_count]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, color=['red', 'green', 'blue'])
    plt.xlabel('Detection Method')
    plt.ylabel('Number of People')
    plt.title('Detection Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

image_path = "PUT IN PATH OF THE IMAGE"
yolo_cfg_path = "PUT IN PATH OF YOLO CFG FILE"
yolo_weights_path = "PUT IN PATH OF YOLO WEIGHTS FILE"
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

main(image_path, yolo_cfg_path, yolo_weights_path, haar_cascade_path)
