import cv2
import numpy as np
from tqdm import tqdm
from depth_estimation import estimate_depth

# Load Yolo
net = cv2.dnn.readNet("config/yolov3.weights", "config/yolov3.cfg")
classes = ['person', '', 'car', 'motorbike']
class_num = [0, 2, 3]
layer_names = net.getLayerNames()
outputLayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float)

# Loading Video
cap = cv2.VideoCapture("data/test.avi")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video Writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('result.avi', fourcc, fps, (width, height))

for _ in tqdm(range(length)):
    if not cap.isOpened():
        break
    ret, frame = cap.read()

    # Detecting Object
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputLayers)

    # Showing Info on the screen
    class_ids = []
    confidences = []
    boxes = []
    depth = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id in class_num:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinate
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                depth.append(estimate_depth(center_x + h/2, center_y + h/2))

    # Duplicated Objects Detected
    indicies = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    numberDetected = len(boxes)

    for i in range(numberDetected):
        try:
            if i in indicies:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                color = colors[class_ids[i]]
                d = depth[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2)
                cv2.putText(frame, '%.2f m' % d, (x, y + 30 + h), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        except Exception as e:
            print(e)
            continue

    output.write(frame)
    # cv2.imshow('image', frame)
    # cv2.waitKey(1)

cap.release()
output.release()
