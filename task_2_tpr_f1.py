import cv2

IOU_THRESHOLD = 0.5

class Image():
    def __init__(self, num):
        self.num = num
        self.image = cv2.imread(f"No_entry/NoEntry{self.num}.bmp", cv2.IMREAD_COLOR)
        self.detected_bounding_boxes = self.get_detected_bounding_boxes()
        self.ground_truth_bounding_boxes = self.get_ground_truth_bounding_boxes()
        
        self.add_to_image(self.detected_bounding_boxes, (0, 255, 0))
        self.add_to_image(self.ground_truth_bounding_boxes, (0, 0, 255))

        self.successful_intersections = self.calc_successful_intersections()
        # self.add_to_image(self.successful_intersections, (255, 0, 0))
        
        self.true_positive_rate()
        self.f1_score()

        self.write_image()

    def get_detected_bounding_boxes(self):
        cascade = cv2.CascadeClassifier("NoEntrycascade/cascade.xml")
        image_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # image_grey = cv2.equalizeHist(image_grey)
        objects_detected = cascade.detectMultiScale(
            image_grey,
            scaleFactor = 1.1,
            minNeighbors = 1,
            flags = cv2.CASCADE_SCALE_IMAGE,
            minSize = (10, 10),
            maxSize = (300, 300)
        )
        return [BoundingBox(*object_detected) for object_detected in objects_detected]

    def get_ground_truth_bounding_boxes(self):
        with open(f"no_entry_signs_detected/ground_truth_bounding_boxes/{self.num}.txt") as f:
            lines = f.readlines()
        return [BoundingBox(*list(map(int, line.strip().split(" ")))) for line in lines]

    def add_to_image(self, bounding_boxes, colour):
        for bounding_box in bounding_boxes:
            bounding_box.add_to_image(self.image, colour)

    def calc_successful_intersections(self):
        successful_intersections = []
        for ground_truth_bounding_box in self.ground_truth_bounding_boxes:
            potential_intersections = []
            for detected_bounding_box in self.detected_bounding_boxes:
                x_left = max(detected_bounding_box.x, ground_truth_bounding_box.x)
                y_top = max(detected_bounding_box.y, ground_truth_bounding_box.y)
                x_right = min(detected_bounding_box.x + detected_bounding_box.w, ground_truth_bounding_box.x + ground_truth_bounding_box.w)
                y_bottom = min(detected_bounding_box.y + detected_bounding_box.h, ground_truth_bounding_box.y + ground_truth_bounding_box.h)
                if x_right >= x_left and y_bottom >= y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    face_cpp_bounding_box_area = detected_bounding_box.w * detected_bounding_box.h
                    ground_truth_bounding_box_area = ground_truth_bounding_box.w * ground_truth_bounding_box.h
                    iou = intersection_area / (face_cpp_bounding_box_area + ground_truth_bounding_box_area - intersection_area)
                    potential_intersections.append([detected_bounding_box, iou])
            if potential_intersections:
                largest_intersection = max(potential_intersections, key = lambda x: x[1])
                if largest_intersection[1] > IOU_THRESHOLD:
                    successful_intersections.append(largest_intersection[0])
        return successful_intersections

    def true_positive_rate(self):
        true_positives = len(self.successful_intersections)
        actual_positives = len(self.ground_truth_bounding_boxes)
        if actual_positives:
            true_positive_rate = true_positives / actual_positives
        else:
            true_positive_rate = None
        print(f"TPR: {true_positive_rate}")

    def f1_score(self):
        true_positives = len(self.successful_intersections)
        false_positives = len(self.detected_bounding_boxes) - true_positives
        false_negatives = len(self.ground_truth_bounding_boxes) - true_positives
        if true_positives or false_positives or false_negatives:
            score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
        else:
            score = None
        print(f"F1: {score}")

    def write_image(self):
        cv2.imwrite(f"no_entry_signs_detected/task_2_images/{self.num}.jpg", self.image)

class BoundingBox():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def add_to_image(self, image, colour):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), colour, 2)

if __name__ == "__main__":
    for i in range(0, 16):
        print(f"Image {i}")
        print("-" * 10)
        Image(i)
        print("")