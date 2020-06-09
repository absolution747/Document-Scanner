# -*- coding: utf-8 -*-
"""
Created on Sun May 31 00:21:04 2020

@author: Karan Shetty
"""

import numpy as np
import cv2

def order_points(points):
    
    rect = np.zeros((4,2), dtype = "float32")
    
    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    diff = np.diff(points)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    return rect

def four_point_transform(image, points):
    
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((tl[0] - tr[0])**2) + ((tl[1] - tr[1])**2))
    widthB = np.sqrt(((bl[0] - br[0])**2) + ((bl[1] - br[1])**2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    heightB = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0,0], [maxWidth - 1,0],[maxWidth - 1,maxHeight - 1],
                    [0,maxHeight - 1]], dtype = "float32") 
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped