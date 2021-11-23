import numpy as np
import cv2

NUM = 1

def get_bounding_boxes(directory):
    with open(f"faces_detected/{directory}/{NUM}.txt") as f:
        lines = f.readlines()
    return [list(map(int, line.strip().split(" "))) for line in lines]

def main():
    image = cv2.imread(f"No_entry/NoEntry{NUM}.bmp", cv2.IMREAD_COLOR)
    face_cpp_bounding_boxes = get_bounding_boxes("face_cpp_bounding_boxes")
    for (x, y, w, h) in face_cpp_bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    ground_truth_bounding_boxes = get_bounding_boxes("ground_truth_bounding_boxes")
    for (x, y, w, h) in ground_truth_bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Display window", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()