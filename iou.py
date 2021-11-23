import numpy as np
import cv2

NUM = 0

def get_face_cpp_rectangles():
    with open(f"faces_detected/coordinates/{NUM}.txt") as f:
        lines = f.readlines()
    return [list(map(int, line.strip().split(" "))) for line in lines]

def main():
    image = cv2.imread(f"No_entry/NoEntry{NUM}.bmp", cv2.IMREAD_COLOR)
    rectangles = get_face_cpp_rectangles()
    for (x, y, w, h) in rectangles:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Display window", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()