"""
cv_ex3.py - using webCam 

"""
import os

import cv2
if __name__ == "__main__":
    # set the argument to be 0
    # the index number is to specify which camera will be used
    cap = cv2.VideoCapture(0)

    name = 'your_name'
    path = os.getcwd() + "/dataset/" + name

    try:
        if not os.path.isdir(path):
            os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

    for i in range(1,4):
        if(cap.isOpened()):
            # press q to quit the webCam
            while(cv2.waitKey(1) != ord('q')):
                ret, frame = cap.read()
                cv2.imshow('webCam',frame)
            else:
                img_filename = path + '/picture' + str(i) + '.png'
                print(img_filename)
                cv2.imwrite(img_filename, frame)
        else:
            print('webCam is failed to use')

    cap.release()
    cv2.destroyAllWindows()
