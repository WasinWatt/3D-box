import cv2
import numpy as np
import glob

# Load previously saved data
with np.load('A.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def findCoor(color,hsv):
    if color == 'orange':
        lower = np.array([0,214,84])
        higher = np.array([16,255,255])

    if color == 'blue':
        lower = np.array([96,65,137])
        higher = np.array([103,255,255])

    if color == 'yellow':
        lower = np.array([20, 100, 100])
        higher = np.array([30, 255, 255])

    if color == 'light green':
        lower = np.array([33,54,0])
        higher = np.array([63,256,256])

    inranged = cv2.inRange(hsv,lower,higher)
    cv2.imshow('inraged',inranged)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(inranged,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    imgg,contours,_= cv2.findContours(dilation, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    maxsize = 1
    width = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > maxsize:
            maxsize = area
            # change needed in the future
            # calculate the center of the contour
            targetX,targetY,width,height = cv2.boundingRect(contour)

    if maxsize == 1:
        return False,0,0
    else:
        return True,targetX+width/2,targetY+height/2

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.array([[0,0,0],[21,0,0],[21,29.7,0],[0,29.7,0]],np.float32)
H = 10
axis = np.float32([[0,0,0],[21,0,0],[21,29.7,0],[0,29.7,0],[0,0,H],[21,0,H],[21,29.7,H],[0,29.7,H]])
# axis = np.float32([[0,0,0], [0,21,0], [30,21,0], [30,0,0],[0,0,-10],[0,21,-10],[30,21,-10],[30,0,-10] ])

# ==================== UNCOMMENT BELOW TO ENABLE REALTIME VISUALIZATION ===============

# cap = cv2.VideoCapture(0)
# count = 0
# while True:
#     ret,img = cap.read()
#     hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     f1, x1, y1 = findCoor('orange',hsv)
#     f2, x2, y2 = findCoor('blue',hsv)
#     f3, x3, y3 = findCoor('light green',hsv)
#     f4, x4, y4 = findCoor('yellow',hsv)
#     corners = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
#     corners2 = np.array(corners,np.float32)
#     # ret, corners = cv2.findChessboardCorners(gray, (30,21),None)
#     # Find the rotation and translation vectors.
#     if f1 & f2 & f3 & f4:
#         success, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
#         # project 3D points to image plane
#         imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
#         img = draw(img,corners2,imgpts)
#     cv2.imshow('img',img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# ==================== UNCOMMENT BELOW TO ENABLE STATIC VISUALIZATION ===============
# change an image
img = cv2.imread('3d5.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
f1, x1, y1 = findCoor('orange',hsv)
f2, x2, y2 = findCoor('blue',hsv)
f3, x3, y3 = findCoor('light green',hsv)
f4, x4, y4 = findCoor('yellow',hsv)
corners = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
corners2 = np.array(corners,np.float32)
# ret, corners = cv2.findChessboardCorners(gray, (30,21),None)
# Find the rotation and translation vectors.
if f1 & f2 & f3 & f4:
    success, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    img = draw(img,corners2,imgpts)
cv2.imshow('img',img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
