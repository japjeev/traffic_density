#import the necessary packages
import cv2
import math
import numpy as np

##Globals
SHAPE = (1080, 1632)
#AREA_PTS = np.array([[420, 412], [675, 383], [1149, 703], [1486, 988], [600, 968]])
#AREA_PTS = np.array([[650, 500], [1178, 500], [1178, 500], [1025, 952], [11, 825]])
AREA_PTS = np.array([[560, 267], [1000, 267], [1000, 267], [288, 589], [7, 400]])
AREA_COLOR = (66, 183, 42)
fcount = 0
frame = None
occ_list = []

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('X','V','I','D'), 30, (1632,1080))

cam = cv2.VideoCapture('capacity2.mp4')

ret,frame = cam.read()
if ret is True:
    run = True
else:
    run = False

while(run):
    # Read a frame from the camera
    ret,frame = cam.read()

    # If the frame was properly read.
    if ret is True:
        fcount = fcount + 1
        
        base = np.zeros(SHAPE + (3,), dtype='uint8')
        area_mask = cv2.fillPoly(base, [AREA_PTS], (255, 255, 255))[:, :, 0]
        xall = np.count_nonzero(area_mask)
        base_frame = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(frame)
    
        edges = cv2.Canny(frame,50,70)
        edges = ~edges
        blur = cv2.bilateralFilter(cv2.blur(edges,(21,21), 100),9,200,200)
        _, threshold = cv2.threshold(blur,230, 255,cv2.THRESH_BINARY)
        
        t = cv2.bitwise_and(threshold,threshold,mask=area_mask)
        
        free = np.count_nonzero(t)
        capacity = 1 - float(free)/xall
        capacity = capacity*100
        occ_list.append(str(capacity))
        
        img = np.zeros(base_frame.shape, base_frame.dtype)
        img[:, :] = AREA_COLOR
        mask2 = cv2.bitwise_and(img, img, mask=area_mask)
        cv2.addWeighted(mask2, 1, base_frame, 1, 0, base_frame)

        cv2.putText(base_frame, "Frame#: " + str(fcount), (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, AREA_COLOR, 2)
        cv2.putText(base_frame, "Occupancy: " + str(round(capacity,0)) + "%", (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, AREA_COLOR, 2)
        cv2.imshow('CCTV Feed',base_frame)
        #cv2.imwrite("frame%d.jpg" % fcount, frame)
        # Write the frame to video file
        out.write(base_frame)
        key = cv2.waitKey(10) & 0xFF
    else:
        break

    if key == 27:
        break

cam.release()
out.release()
cv2.destroyAllWindows()

# Open File
resultFile = open("output.csv",'w')

for item in occ_list:
    resultFile.write(item + "\n")

resultFile.close()
