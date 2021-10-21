import cv2 as cv
import numpy as np
img = cv.imread('MyShapes.jpeg')
blurred = cv.GaussianBlur(img, (5, 5),0)
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
threshold,thresh=cv.threshold(gray,100,255,cv.THRESH_BINARY)
canny = cv.Canny(thresh,200,255)
dilate = cv.dilate(canny, (3,3), iterations=1)
contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img,contours,-1,(0,255,0),2)
for contour in contours:
    blankimg = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    blankimg.fill(255)
    cv.drawContours(blankimg, contour, -1, (0, 0, 0), 3)
    blankimg = cv.cvtColor(blankimg, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(blankimg, 100, 0.2, 10)
    corners = np.int0(corners)
    average = np.average(corners, axis=0)
    average = np.int0(average)
    x ,y = average.ravel()
    fontsize=0.8
    font=cv.FONT_HERSHEY_COMPLEX
    if len(corners) == 3:
      cv.putText(img, "Triangle", (x, y), font, fontsize,(0, 0, 255), 1)
    elif len(corners) == 4:
      cv.putText(img, "Rectangle", (x, y), font, fontsize, (0, 0, 255), 1)
    elif len(corners) == 5:
      cv.putText(img, "Pentagon", (x, y), font, fontsize, (0, 0, 255), 1)
    else:
      cv.putText(img, "Circle", (x, y), font, fontsize, (0, 0, 255), 1)
cv.imshow('GeometricShapeRecognitonProject',img)

cv.waitKey(0)