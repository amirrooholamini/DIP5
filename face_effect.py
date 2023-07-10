import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time

from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

fd = UltraLightFaceDetecion(
        "weights/RFB-320.tflite",
        conf_threshold=0.88)
fa = CoordinateAlignmentModel(
    "weights/coor_2d106.tflite")

cap = cv2.VideoCapture(0)
color = (125, 255, 125)

left_eye = [35, 36, 33, 37, 39, 42, 40, 41]
right_eye = [89, 90, 87, 91, 93, 96, 94, 95]
lib = [52, 55,56,53,59,58,61,68,67,71,63, 64]
left_eye_points = []
right_eye_points = []
lib_points = []
sticker = cv2.imread("watermelon.png", cv2.IMREAD_UNCHANGED)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
selection = ""

def set_large_image(frame, pred, indexs):
    points = []
    for i in indexs:
        points.append(pred[i])
    points = np.array(points, dtype = int)
    mask = np.zeros(frame.shape, dtype=np.uint8)
    try:
        cv2.drawContours(mask,[points], -1,(255,255,255), -1)
        x,y,w,h = cv2.boundingRect(points)
        mask = mask // 255
        object = mask * frame
        object = object[y:y+h, x:x+w]
        larg_img = cv2.resize(object, (0,0), fx=2, fy=2)
        center_c = (x+x+w)//2
        center_r = (y+y+h)//2

        R,C,_ = larg_img.shape
        if R % 2 == 0:
            start_r = center_r - (R//2)
        else:
            start_r = center_r - (R//2) - 1
        end_r = center_r + (R//2)

        if C % 2 == 0:
            start_c = center_c - (C//2)
        else:
            start_c = center_c - (C//2) - 1
        end_c = center_c + (C//2)

        start_r = max(0,start_r)
        start_c = max(0,start_c)

        end_r = min(end_r, frame.shape[0]-1)
        end_c = min(end_c, frame.shape[1]-1)

        for i in range(start_r, end_r):
            for j in range(start_c, end_c):
                x = i - start_r
                y = j - start_c
                if any(value > 0 for value in larg_img[x,y]):
                    if i < frame.shape[0] and j < frame.shape[1]:
                        frame[i,j] = larg_img[x,y]
    
        return frame
    except Exception as e:
        return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if selection == "fruit":
        left_eye_points = []
        right_eye_points = []
        lib_points = []
        boxes, scores = fd.inference(frame)
        for pred in fa.get_landmarks(frame, boxes):
            for i in left_eye:
                left_eye_points.append(pred[i])

            for i in right_eye:
                right_eye_points.append(pred[i])

            for i in lib:
                lib_points.append(pred[i])

            left_eye_points = np.array(left_eye_points, dtype = int)
            right_eye_points = np.array(right_eye_points, dtype = int)
            lib_points = np.array(lib_points, dtype = int)
        
        
        mask = np.zeros(frame.shape, dtype=np.uint8)
        try:
            cv2.drawContours(mask,[left_eye_points], -1,(255,255,255), -1)
            cv2.drawContours(mask,[right_eye_points], -1,(255,255,255), -1)
            cv2.drawContours(mask,[lib_points], -1,(255,255,255), -1)
            
        except:
            continue
        mask = mask // 255
        eyes = mask * frame
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            img = cv2.resize(sticker, (h,w))
            for r in range(y,y+h):
                for c in range(x,x+w):
                    a = r-y
                    b = c-x
                    if img[a,b,3]>0 and r>0 and r<frame.shape[0] and c>0 and c<frame.shape[1]:
                        frame[r,c] = img[a,b,:3]
        frame = cv2.resize(frame, (0,0), fx=2, fy=2)
        frame = np.where(mask == 0, frame, eyes)

    elif selection == "large":
        boxes, scores = fd.inference(frame)
        for pred in fa.get_landmarks(frame, boxes):
            frame = set_large_image(frame, pred, lib)
            frame = set_large_image(frame, pred, left_eye)
            frame = set_large_image(frame, pred, right_eye)
                

    cv2.imshow("result", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

    elif key == ord('1'):
        selection = "fruit"

    elif key == ord('2'):
        selection = "large"