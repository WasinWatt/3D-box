import cv2
import numpy as np

back_meter_ratio = # constant
front_meter_ratio = # constant
back_meter = back_meter_ratio * height
front_meter = front_meter_ratio * height
y_back = # constant
y_front = # constant
def main():
    filename = '1-51m.jpg'
    # read input image
    img = cv2.imread(filename)
    # BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # find corners of the floor
    upleftX, upleftY = findCoor('color1')
    uprightX, uprightY = findCoor('color2')
    botleftX, botleftY = findCoor('color3')
    botrightX, botrightY = findCoor('color4')
    # set up the corners of the ceiling
    H_upleftX, H_upleftY = floorCoor(upleftX,upleftY,height)
    H_uprightX, H_uprightY = floorCoor(uprightX,uprightY,height)
    H_botleftX, H_botrightY = floorCoor(botleftX,botleftY,height)
    H_botrightX, H_botrightY = floorCoor(botrightX,botrightY,height)
    # draw all lines
        # horizontal lines
    cv2.line(img, (upleftX, upleftY), (uprightX, uprightY), (0,0,0))
    cv2.line(img, (botleftX, botleftY), (botrightX, botrightY), (0,0,0))
    cv2.line(img, (H_upleftX, H_upleftY), (H_uprightX, H_uprightY), (0,0,0))
    cv2.line(img, (H_botleftX, H_botleftY), (H_botrightX, H_botrightY), (0,0,0))
        # vertical lines
    cv2.line(img, (H_upleftX, H_upleftY), (upleftX, upleftY), (0,0,0))
    cv2.line(img, (H_uprightX, H_uprightY), (uprightX, uprightY), (0,0,0))
    cv2.line(img, (H_botleftX, H_botleftY), (botleftX, botleftY), (0,0,0))
    cv2.line(img, (H_botrightX, H_botrightY), (botrightX, botrightY), (0,0,0))
        # depth lines
    cv2.line(img, (upleftX, upleftY), (botleftX, botleftY), (0,0,0))
    cv2.line(img, (uprightX, uprightY), (botrightX, botrightY), (0,0,0))
    cv2.line(img, (H_upleftX, H_upleftY), (H_botleftX, H_botleftY), (0,0,0))
    cv2.line(img, (H_uprightX, H_uprightY), (H_botrightX, H_botrightY), (0,0,0))

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

def findCoor(color):
    if color == 'orange':
        lower = np.array([0,214,84])
        higher = np.array([16,256,256])
    if color == 'blue':
        lower = np.array([96,65,137])
        higher = np.array([103,256,256])
    if color == 'yellow':
        lower = np.array([26,119,137])
        higher = np.array([47,256,256])
    if color == 'light green':
        lower = np.array([33,54,0])
        higher = np.array([63,256,256])
    inranged = cv2.inRange(hsv,lower,higher)
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
	if(maxsize == 1):
		return 0,0
	else:
		return targetX+width/2,targetY+height/2

def floorCoor( x, y, height ):
    m = ( y_front - y_back ) / ( front_meter - back_meter )
    floorY = ( y - y_back ) / m + back_meter_ratio
    return x, floorY
