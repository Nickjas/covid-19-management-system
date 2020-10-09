from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import tkinter as tk
from tkinter import *
import csv
from PIL import Image,ImageTk
import pandas as pd
import datetime
import time
from imutils.video import VideoStream

def social_distance_detection():
    MIN_CONF = 0.3    # minimum object detection confidence
    NMS_THRESH = 0.3  # non-maxima suppression threshold

# boolean indicating if NVIDIA CUDA GPU should be used
    USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
    MIN_DISTANCE = 50
    def detect_people(frame, net, ln, personIdx=0):
        # grab the dimensions of the frame and  initialize the list of
    # results
        (H, W) = frame.shape[:2]
        results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
        boxes = []
        centroids = []
        confidences = []

    # loop over each of the layer outputs
        for output in layerOutputs:
        # loop over each of the detections  
            for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
                if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                # update our list of bounding box coordinates,
                # centroids, and confidences
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
    
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    
    # ensure at least one detection exists
        if len(idxs) > 0:
        # loop over the indexes we are keeping
            for i in idxs.flatten():
            # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
            
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)

    # return the list of results
        return results
        # derive the paths to the YOLO weights and model configuration
    weightsPath = 'C:/Users/HP/Downloads/yolov3.weights'
    configPath = 'C:/Users/HP/Desktop/covid/facemask/yolov3/cfg/yolov3.cfg'
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    labelsPath = 'C:/Users/HP/Desktop/covid/facemask/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    #print(LABELS[0])
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def video_stream():
        print("[INFO] accessing video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        #cap = cv2.VideoCapture('C:/Users/HP/Desktop/tensorflow/Social_Distancing-CV-master/Social_Distancing-CV-master/people.mp4')
        writer = None
        while True:
            # read the next frame from the file
            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end
            if not grabbed:
                break
            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                            personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the minimum social
    # distance
            violate = set()
            # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")
                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
                        if D[i, j] < MIN_DISTANCE:
                    # update our violation set with the indexes of
                    # the centroid pairs
                            violate.add(i)
                            violate.add(j)
            for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)
                # if the index pair exists within the violation set, then
        # update the color
                if i in violate: 
                    color = (0, 0, 255)
        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

         # draw the total number of social distancing violations on the
    # output frame
            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
                    # check to see if the output frame should be displayed to our
    # screen
            cv2.imshow("Frame", frame)
    
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    cv2.destroyAllWindows()
   
    

    window = tk.Tk()
    window.title("CMS-Covid Management System")

    window.geometry('520x400')
    window.configure(background='cyan')
    video_stream = tk.Button(window, text="live stream",command=video_stream,fg="white"  ,bg="blue2"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    takeImg.place(x=90, y=50)

    analysis = tk.Button(window, text="social distance analysis",fg="black",command=trainimg ,bg="lawn green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
    trainImg.place(x=390, y=50)

    btn.grid(column=2, row=0)
    window.mainloop()


windo = tk.Tk()
windo.title("CMS-Covid Management System")

windo.geometry('1280x720')
windo.configure(background='green')
btn = Button(windo, text="Click Me", command=social_distance_detection)

btn.grid(column=2, row=0)
windo.mainloop()