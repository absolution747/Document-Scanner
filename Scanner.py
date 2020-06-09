# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:34:15 2020

@author: Karan Shetty
"""

import numpy as np
import cv2
import imutils
from transform import four_point_transform

#img = cv2.imread('page.jpg')
img = cv2.imread('notebook2.jpg')
scan = True
print(img.shape)
ratio = img.shape[0] / 500.0
orig = img.copy()
img = imutils.resize(img, height = 500)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edge = cv2.Canny(gray, 75, 200)
cv2.imshow("edge", edge)   


cv2.waitKey(0) 
cv2.destroyAllWindows()

cnts = cv2.findContours(edge, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    
    if len(approx) == 4:
        print("Document Found.")
        ScreenCnt = approx
        scan = True
        break
    else:
        print("No Documents Found.")
        scan = False
        break

if scan :
    print(ScreenCnt)

    print("STEP 2: Find contours of paper")
    cv2.drawContours(img, [ScreenCnt], -1, (255, 0, 0), 2)
    cv2.imshow("Outline", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    warped = four_point_transform(orig, ScreenCnt.reshape(4,2) * ratio)
    #warped = imutils.resize(warped, height = 500)
    cv2.imshow("Doc", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                  cv2.THRESH_BINARY,21,10)
    cv2.imshow("Documeny", warped)
    cv2.imwrite("Document.jpg", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

 

