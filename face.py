import sys
import numpy as np
import cv2

def detect_and_display(frame, cascade):
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_grey = cv2.equalizeHist(frame_grey)

    faces = cascade.detectMultiScale(
        frame_grey,
        scaleFactor = 1.1,
        minNeighbors = 1,
        flags = cv2.CASCADE_SCALE_IMAGE,
        minSize = (10, 10),
        maxSize = (300, 300)
    )
    print(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Display window", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def main():
    frame = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    
    cascade = cv2.CascadeClassifier("frontalface.xml")
    # cascade = cv2.CascadeClassifier("Noentrycascade/cascade.xml")

    detect_and_display(frame, cascade)
    
if __name__ == "__main__":
    main()