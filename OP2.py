import numpy as np
import cv2

cap = cv2.VideoCapture(0)
'''#codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('edge11.avi',fourcc, 2000.0, (640,480))'''


# function define

def ROI(edges):
    '''print("second half edges")'''
    height, width = edges.shape
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
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 50  # minimal of votes
    line_segments = cv2.HoughLinesP(A, rho, angle, min_threshold, 
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments



def ASI(img,B):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if B is None:
        #logging.info('No line_segment segments detected')
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

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(img, right_fit_average))

    #logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines

def make_points(img, line):
    height, width, _ = img.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def DL(img, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
    return line_image

    
while(True):
    # Capture frame-by-frame
    '''ret, frame = cap.read()'''

    img=cv2.imread("IMG_5958 001.jpg", cv2.IMREAD_COLOR)

    
    #hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # detecting colour
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([145, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    
    # edges
    edges = cv2.Canny(mask, 100, 600)
    
    #calling functions
    A=ROI(edges)
    B=DLS(A)
    C=ASI(img,B)
    L=DL(img,C)
    print(A.shape)
    '''print(B.shape)'''
    print(L.shape)
    
    #midpoint & 1lane
   
    

    # Display & capturing the resulting frame
    cv2.imshow('second half edges displaying',L)
    '''cv2.imshow("hsv",hsv)'''
    cv2.imshow("edges",edges)
    '''cv2.imwrite("cap.png",L)

    #write frame
    out.write(L)'''

    
    
    if cv2.waitKey(1) & 0xFF == ord('1'):

        print("successfully marked")

        break
    
   



# When everything done, release the capture
'''cap.release()
out.release()'''
cv2.destroyAllWindows()










