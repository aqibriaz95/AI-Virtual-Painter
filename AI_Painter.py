import mediapipe as mp
import cv2
import numpy as np
import time
import HandTrackingModule as htm
import os

folderPath = r"C:\Users\Aqib HOME\Documents\Python Scripts\Painter_Images"

myList = os.listdir(folderPath)
overlayList = []
brushThickness = 15
eraserThickness = 50

xp, yp = 0 , 0
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
imgCanvas = np.zeros((720,1280,3),np.uint8)
drawColor = (255,0,255)

cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    img[0:125,0:1280] = header
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList)!=0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
        fingers = detector.fingersUp()
        
        
        if fingers[1] and fingers[2]:
            xp,yp =0 , 0
            
            if y1 <125:
                if 350<x1<460:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 515<x1<630:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 700<x1<815:
                    header = overlayList[2]
                    drawColor = (0,255,0)
                elif 860<x1<980:
                    header = overlayList[3]
                    drawColor = (0,140,255)
                elif 1095<x1<1247:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img,(x1,y1-20),(x2,y2 +20),drawColor,cv2.FILLED)
        if fingers[1] and fingers[2]==False:
            
            cv2.circle(img,(x1,y1),8,drawColor,cv2.FILLED)
            if xp==0 and yp==0:
                xp, yp = x1 ,y1
                
            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,brushThickness)
            
            xp,yp = x1,y1
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,20,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    
    
    cv2.imshow('Image',img)
    cv2.waitKey(1)