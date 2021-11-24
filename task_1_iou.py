import numpy as np
import cv2

NUM = 5

class Image():
    def __init__(self, location):
        self.image = cv2.imread(location, cv2.IMREAD_COLOR)
        self.face_cpp_bounding_boxes = self.get_bounding_boxes("face_cpp_bounding_boxes")
        self.ground_truth_bounding_boxes = self.get_bounding_boxes("ground_truth_bounding_boxes")
        
        self.add_bounding_boxes_to_image(self.face_cpp_bounding_boxes, (0, 255, 0))
        self.add_bounding_boxes_to_image(self.ground_truth_bounding_boxes, (0, 0, 255))

        self.intersection_over_union()

        self.write_image()
        self.display_image()

    def get_bounding_boxes(self, location):
        with open(f"faces_detected/{location}/{NUM}.txt") as f:
            lines = f.readlines()
        return BoundingBoxes([BoundingBox(*list(map(int, line.strip().split(" ")))) for line in lines])

    def add_bounding_boxes_to_image(self, bounding_boxes, colour):
        for box in bounding_boxes.boxes:
            cv2.rectangle(self.image, (box.x, box.y), (box.x + box.w, box.y + box.h), colour, 2)

    def intersection_over_union(self):
        for face_cpp_bounding_box in self.face_cpp_bounding_boxes.boxes:
            for ground_truth_bounding_box in self.ground_truth_bounding_boxes.boxes:
                x_left = max(face_cpp_bounding_box.x, ground_truth_bounding_box.x)
                y_top = max(face_cpp_bounding_box.y, ground_truth_bounding_box.y)
                x_right = min(face_cpp_bounding_box.x + face_cpp_bounding_box.w, ground_truth_bounding_box.x + ground_truth_bounding_box.w)
                y_bottom = min(face_cpp_bounding_box.y + face_cpp_bounding_box.h, ground_truth_bounding_box.y + ground_truth_bounding_box.h)

                if x_right < x_left or y_bottom < y_top:
                    # print(0)
                    pass
                else:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    face_cpp_bounding_box_area = face_cpp_bounding_box.w * face_cpp_bounding_box.h
                    ground_truth_bounding_box_area = ground_truth_bounding_box.w * ground_truth_bounding_box.h
                    iou = intersection_area / (face_cpp_bounding_box_area + ground_truth_bounding_box_area - intersection_area)
                    print(iou)

    def write_image(self):
        cv2.imwrite(f"faces_detected/task_1_images/{NUM}.jpg", self.image)

    def display_image(self):
        cv2.imshow("Display window", self.image)
        cv2.waitKey()
        cv2.destroyAllWindows()

class BoundingBoxes():
    def __init__(self, boxes):
        self.boxes = boxes

class BoundingBox():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

if __name__ == "__main__":
    Image(f"No_entry/NoEntry{NUM}.bmp")