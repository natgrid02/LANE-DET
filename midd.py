import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('WhatsApp Video 2020-02-09 at 19.35.05.mp4')
cap.set(3,480)
cap.set(4,848)

#codec
'''fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('laneee.mp4',fourcc, 2000.0, (848,480))'''


# function define

def ROI(edges):
    width,height= edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges



def DLS(A):
#tuning min_threshold,minLineLength,maxLineGap is a trial and error process by hand
    rho = 10  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(A, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=10, maxLineGap=5)

    return line_segments



def ASI(img,B):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if B is None:
        
        return lane_lines
    

    height, width, _ = img.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    for line_segment in B:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                #logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(img, left_fit_average))
        print("l",left_fit_average)

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(img, right_fit_average))
        print("r",right_fit_average)

    #logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

def make_points(img, line):
    height, width, _ = img.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min( 2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min( 2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def DL(img, lines, line_color=(139, 0, 139), line_width=4):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            '''print(line)'''
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(img, 0.8, line_image, 1, 1) 
    return line_image

    
while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    '''img=cv2.imread("WhatsApp Video 2020-02-09 at 19.35.05.mp4",WhatsApp Video 2020-02-18 at 11.54.06.mp4 )'''
    '''img = cv2.flip(img,0)'''
    

    '''#blur
    blur=cv2.GaussianBlur(img,(5,5),0)'''
    

    
    mask = cv2.imread('WhatsApp Video 2020-02-09 at 19.35.05.mp4.png',0)

    dst_TELEA = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    dst_NS = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

    
    #hsv
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''# detecting colour
    lower_white = np.array([100,150, 130])
    upper_white = np.array([ 220,190,211])
    mask = cv2.inRange(hsv, lower_white, upper_white)'''

    

    
    # edges
    edges = cv2.Canny(hsv, 50, 250)
    
    #calling functions
    
    A=ROI(edges)
    B=DLS(A)
    C=ASI(img,B)
    L=DL(img,C)
    '''print(A.shape)'''
    '''print(B.shape)'''
    '''print(L.shape)'''
    '''print(edges.shape)'''
    
    
    '''#midpoint & lane
    _,  _,left_x2, _ = L[0][0]
    _, _, right_x2, _ = L[1][0]
    width=480
    height=848
    mid = int(width / 2)
    x_offset = (left_x2 + right_x2) / 2 - mid
    y_offset = int(height / 2)

    x1, _, x2, _ = L[0][0]
    x_offset = x2 - x1
    y_offset = int(height / 2)
    

    angle_to_mid_radian = math.atan(x_offset / y_offset) 
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi) 
    steering_angle = angle_to_mid_deg + 90'''

    

    # Display & capturing the resulting frame
    
    cv2.imshow("hsv",hsv)
    '''cv2.imshow('mask',mask)'''
    cv2.imshow(" canny edges",edges)
    cv2.imshow('L',L)
    '''cv2.imwrite("cap.png",L)'''

    #write frame
    '''out.write(L)'''

    
    
    if cv2.waitKey(1) & 0xFF == ord('3'):

        print("successfully marked")

        break
    
   



# When everything done, release the capture
'''cap.release()'''
'''out.release()'''
cv2.destroyAllWindows()











