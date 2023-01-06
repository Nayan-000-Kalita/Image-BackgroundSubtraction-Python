import numpy as np
import cv2


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[1]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



cap = cv2.VideoCapture ('filename.mp4') #Enter file location
ret, frame = cap.read()
print('ret =', ret, 'Width =', frame.shape[1], 'Height =', frame.shape[0], 'channel =', frame.shape[2])


fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg1 = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg2 = cv2.createBackgroundSubtractorMOG2(history = 20, varThreshold = 150, detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
fgbg3 = cv2.createBackgroundSubtractorKNN()
fgbg4 = cv2.bgsegm.createBackgroundSubtractorGSOC()
fgbg5 = cv2.bgsegm.createBackgroundSubtractorCNT()
fgbg6=cv2.bgsegm.createBackgroundSubtractorLSBP()



while(cap.isOpened()):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask1 = fgbg1.apply(frame)
    fgmask2 = fgbg2.apply(frame)
    fgmask3 = fgbg3.apply(frame)
    fgmask4 = fgbg4.apply(frame)
    fgmask5 = fgbg5.apply(frame)
    fgmask6 = fgbg6.apply(frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (60, 60)
    fontScale = 3
    color = (255, 255, 255)
    thickness = 5
    cv2.putText(frame, 'Original', org, font, fontScale, (0,0,255), thickness, cv2.LINE_4)
    cv2.putText(fgmask, 'GMG', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(fgmask1, 'MOG', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(fgmask2, 'MOG2', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(fgmask3, 'KNN', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(fgmask4, 'GSOC', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(fgmask5, 'CNT', org, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(fgmask6, 'LSBP', org, font, fontScale, color, thickness, cv2.LINE_AA)



    stack = stackImages(0.3,([frame,fgmask6,fgmask2],[fgmask3,fgmask4,fgmask5]))
    #print('ret =', ret, 'Width =', frame.shape[1], 'Height =', frame.shape[0], 'channel =', frame.shape[2])

    #stackedImage = stackImages(0.3, ([frame,fgmask5,fgmask6]))



    cv2.imshow("Output", stack)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
