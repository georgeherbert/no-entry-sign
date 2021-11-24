import numpy as np
import cv2

NUM = 11
IOU_THRESHOLD = 0.5

class Image():
    def __init__(self, location):
        self.image = cv2.imread(location, cv2.IMREAD_COLOR)
        self.face_cpp_bounding_boxes = self.get_bounding_boxes("face_cpp_bounding_boxes")
        self.ground_truth_bounding_boxes = self.get_bounding_boxes("ground_truth_bounding_boxes")
        
        self.add_to_image(self.face_cpp_bounding_boxes, (0, 255, 0))
        self.add_to_image(self.ground_truth_bounding_boxes, (0, 0, 255))

        self.successful_intersections = self.calc_successful_intersections()
        # self.add_to_image(self.successful_intersections, (255, 0, 0))

        self.true_positive_rate = self.calc_true_positive_rate()
        print(f"TPR: {self.true_positive_rate}")

        self.write_image()

    def get_bounding_boxes(self, location):
        with open(f"faces_detected/{location}/{NUM}.txt") as f:
            lines = f.readlines()
        return [BoundingBox(*list(map(int, line.strip().split(" ")))) for line in lines]

    def add_to_image(self, bounding_boxes, colour):
        for bounding_box in bounding_boxes:
            bounding_box.add_to_image(self.image, colour)

    def calc_successful_intersections(self):
        successful_intersections = []
        for ground_truth_bounding_box in self.ground_truth_bounding_boxes:
            potential_intersections = []
            for face_cpp_bounding_box in self.face_cpp_bounding_boxes:
                x_left = max(face_cpp_bounding_box.x, ground_truth_bounding_box.x)
                y_top = max(face_cpp_bounding_box.y, ground_truth_bounding_box.y)
                x_right = min(face_cpp_bounding_box.x + face_cpp_bounding_box.w, ground_truth_bounding_box.x + ground_truth_bounding_box.w)
                y_bottom = min(face_cpp_bounding_box.y + face_cpp_bounding_box.h, ground_truth_bounding_box.y + ground_truth_bounding_box.h)
                if x_right >= x_left and y_bottom >= y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    face_cpp_bounding_box_area = face_cpp_bounding_box.w * face_cpp_bounding_box.h
                    ground_truth_bounding_box_area = ground_truth_bounding_box.w * ground_truth_bounding_box.h
                    iou = intersection_area / (face_cpp_bounding_box_area + ground_truth_bounding_box_area - intersection_area)
                    potential_intersections.append([face_cpp_bounding_box, iou])
            if potential_intersections:
                largest_intersection = max(potential_intersections, key = lambda x: x[1])
                if largest_intersection[1] > IOU_THRESHOLD:
                    successful_intersections.append(largest_intersection[0])
        return successful_intersections

    def calc_true_positive_rate(self):
        true_positives = len(self.successful_intersections)
        if true_positives:
            true_positive_rate = true_positives / len(self.ground_truth_bounding_boxes)
        else:
            true_positive_rate = None
        return true_positive_rate

    def write_image(self):
        cv2.imwrite(f"faces_detected/task_1_images/{NUM}.jpg", self.image)

class BoundingBox():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def add_to_image(self, image, colour):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), colour, 2)

if __name__ == "__main__":
    Image(f"No_entry/NoEntry{NUM}.bmp")