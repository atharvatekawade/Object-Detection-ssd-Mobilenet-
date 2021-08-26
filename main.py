import cv2

labels=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

cap.set(10,120)

while True:
    succ,img=cap.read()
    classes, confidences, boxes = net.detect(img, confThreshold=0.5)

    if(len(classes)!=0):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            cv2.rectangle(img, box, color=(0, 255, 0))
            cv2.putText(img, f'{labels[classId-1]} {round(confidence*100,2)} %', (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
