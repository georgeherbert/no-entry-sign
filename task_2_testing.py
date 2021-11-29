import cv2

from task_2_viola_jones import ViolaJonesDetector

IOU_THRESHOLD = 0.5

class Testing():
    def __init__(self, num):
        self.num = num
        self.detector = ViolaJonesDetector(f"No_entry/NoEntry{self.num}.bmp")
        self.ground_truth_boxes = self.get_ground_truth_boxes()
        self.successful_intersections = self.calculate_successful_intersections()
        self.tpr = self.calculate_tpr()
        self.f1 = self.calculate_f1()
        
    def get_ground_truth_boxes(self):
        with open(f"no_entry_sign_ground_truths/{self.num}.txt") as f:
            lines = f.readlines()
        return [tuple(map(int, line.strip().split(" "))) for line in lines]

    def calculate_successful_intersections(self):
        successful_intersections = []
        for x1, y1, w1, h1 in self.ground_truth_boxes:
            potential_intersections = []
            for x2, y2, w2, h2 in self.detector.objects:
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                if x_right >= x_left and y_bottom >= y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    ground_truth_box_area = w1 * h1
                    detector_box_area = w2 * h2
                    iou = intersection_area / (ground_truth_box_area + detector_box_area - intersection_area)
                    potential_intersections.append([(x2, y2, w2, h2), iou])
            if potential_intersections:
                largest_intersection = max(potential_intersections, key = lambda x: x[1])
                if largest_intersection[1] > IOU_THRESHOLD:
                    successful_intersections.append(largest_intersection[0])
        return successful_intersections

    def calculate_tpr(self):
        true_positives = len(self.successful_intersections)
        actual_positives = len(self.ground_truth_boxes)
        if actual_positives:
            true_positive_rate = true_positives / actual_positives
        else:
            true_positive_rate = None
        return true_positive_rate

    def calculate_f1(self):
        true_positives = len(self.successful_intersections)
        false_positives = len(self.detector.objects) - true_positives
        false_negatives = len(self.ground_truth_boxes) - true_positives
        if true_positives or false_positives or false_negatives:
            f1_score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
        else:
            f1_score = None
        return f1_score

    def draw_boxes(self):
        self.detector.draw_boxes()
        for x, y, w, h in self.ground_truth_boxes:
            cv2.rectangle(self.detector.image.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # for x, y, w, h in self.successful_intersections:
        #     cv2.rectangle(self.detector.image.image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def save_image(self):
        cv2.imwrite(f"task_2_testing_output/{self.num}.jpg", self.detector.image.image)

if __name__ == "__main__":
    for i in range(0, 16):
        test = Testing(i)
        test.draw_boxes()
        test.save_image()
        print(f"Image {i}")
        print("-" * 10)
        print(f"TPR: {test.tpr}")
        print(f"F1: {test.f1}")
        print("")