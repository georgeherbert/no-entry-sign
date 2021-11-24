import numpy as np
import cv2

NUM = 4

class Image():
    def __init__(self, location):
        self.image = cv2.imread(location, cv2.IMREAD_COLOR)
        self.face_cpp_bounding_boxes = self.get_bounding_boxes("face_cpp_bounding_boxes")
        self.ground_truth_bounding_boxes = self.get_bounding_boxes("ground_truth_bounding_boxes")
        
        self.add_bounding_boxes_to_image(self.face_cpp_bounding_boxes, (0, 255, 0))
        self.add_bounding_boxes_to_image(self.ground_truth_bounding_boxes, (0, 0, 255))

        self.write_image()
        self.display_image()

    def get_bounding_boxes(self, location):
        with open(f"faces_detected/{location}/{NUM}.txt") as f:
            lines = f.readlines()
        return [list(map(int, line.strip().split(" "))) for line in lines]

    def add_bounding_boxes_to_image(self, bounding_boxes, colour):
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), colour, 2)

    def write_image(self):
        cv2.imwrite(f"faces_detected/task_1_images/{NUM}.jpg", self.image)

    def display_image(self):
        cv2.imshow("Display window", self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Image(f"No_entry/NoEntry{NUM}.bmp")