# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture("F:\pythonProject\filename.avi")

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):

    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images


        name = 'Frame' + str(currentframe) + '.png'
        print('Creating...' + name)
        ##Resizing frames to 480pixels
        width = 854
        height = 480
        imgResize = cv2.resize(frame, (width, height))

        ## Image cropping
        #imgCrop = imgResize[130:410, 0:854]
        # writing the extracted images
        cv2.imwrite(name, frame)


        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
