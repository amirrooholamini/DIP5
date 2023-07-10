import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time

from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel

def rotate_180(image):
    R,C,_ = image.shape
    img = np.zeros(image.shape)
    print(R,C)
    for r in range(R):
        img[r,:,:] = image[R - r - 1, : , :]
    img = img.astype(np.uint8)
    return img


fd = UltraLightFaceDetecion(
        "weights/RFB-320.tflite",
        conf_threshold=0.88)
fa = CoordinateAlignmentModel(
    "weights/coor_2d106.tflite")


left_eye = [35, 36, 33, 37, 39, 42, 40, 41]
right_eye = [89, 90, 87, 91, 93, 96, 94, 95]
lib = [52, 55,56,53,59,58,61,68,67,71,63, 64]
left_eye_points = []
right_eye_points = []
lib_points = []

image = cv2.imread("rotation.jpg")

boxes, scores = fd.inference(image)
for pred in fa.get_landmarks(image, boxes):
    for i in left_eye:
        left_eye_points.append(pred[i])

    for i in right_eye:
        right_eye_points.append(pred[i])

    for i in lib:
        lib_points.append(pred[i])

    left_eye_points = np.array(left_eye_points, dtype = int)
    right_eye_points = np.array(right_eye_points, dtype = int)
    lib_points = np.array(lib_points, dtype = int)

    mask = np.zeros(image.shape, dtype=np.uint8)
    x,y,w,h = cv2.boundingRect(left_eye_points)
    padding = 5
    x -= padding
    y -= padding
    w += padding*2
    h += padding*2
    left_eye_image = image[y:y+h,x:x+w]
    rotated = rotate_180(left_eye_image)
    image[y:y+h,x:x+w] = rotated

    mask = np.zeros(image.shape, dtype=np.uint8)
    x,y,w,h = cv2.boundingRect(right_eye_points)
    x -= padding
    y -= padding
    w += padding*2
    h += padding*2
    right_eye_image = image[y:y+h,x:x+w]
    rotated = rotate_180(right_eye_image)
    image[y:y+h,x:x+w] = rotated

    mask = np.zeros(image.shape, dtype=np.uint8)
    x,y,w,h = cv2.boundingRect(lib_points)
    x -= padding
    y -= padding
    w += padding*2
    h += padding*2
    lib_image = image[y:y+h,x:x+w]
    rotated = rotate_180(lib_image)
    image[y:y+h,x:x+w] = rotated

    image = rotate_180(image)

    cv2.imshow("rotated", image)
    cv2.waitKey()