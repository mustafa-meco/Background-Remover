import cvzone
import numpy as np
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()


while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, (255, 0, 0), threshold=0.7)



    imgStacked = cvzone.stackImages([img, imgOut],2,1)
    #cv2.imshow("Image", img)
    #cv2.imshow("Image Out", imgOut)
    _, imgStacked = fpsReader.update(imgStacked, color= (0,0, 255))


    cv2.imshow("Image", imgStacked)
    cv2.waitKey(1)

