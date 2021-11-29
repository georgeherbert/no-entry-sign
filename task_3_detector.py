import sys
import numpy as np
import cv2

MINIMUM_RADIUS = 10
MAXIMUM_RADIUS = 110
MIN_DISTANCE = 50

T_S = 250
# T_H = 100

IOU_THRESHOLD = 0.5

class ErrorSignDetector():
    def __init__(self, file):
        self.image = Image(file)
        self.hough_circles = HoughCirclesDetector(self.image)
        self.viola_jones = ViolaJonesDetector(self.image)
        self.objects = self.calculate_objects()

    def calculate_objects(self):
        successful_intersections = []
        for x1, y1, w1, h1 in self.hough_circles.boxes:
            potential_intersections = []
            for x2, y2, w2, h2 in self.viola_jones.objects:
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                if x_right >= x_left and y_bottom >= y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    hough_box_area = w1 * h1
                    viola_jones_area = w2 * h2
                    iou = intersection_area / (hough_box_area + viola_jones_area - intersection_area)
                    potential_intersections.append([(x1, y1, w1, h1), iou])
            if potential_intersections:
                largest_intersection = max(potential_intersections, key = lambda x: x[1])
                if largest_intersection[1] > IOU_THRESHOLD:
                    successful_intersections.append(largest_intersection[0])
        return successful_intersections

    def draw_boxes(self):
        for x, y, w, h in self.objects:
            cv2.rectangle(self.image.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # for x, y, w, h in self.viola_jones.objects:
        #     cv2.rectangle(self.image.image, (x, y), (x + w, y + h), (255, 255, 0), 1)
        # for x, y, r, _ in self.hough_circles.circles:
        #     cv2.circle(self.image.image, (int(x), int(y)), int(r + MINIMUM_RADIUS), (0, 255, 255), 1)

    def normalise(self, image):
        image_normalised = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        return image_normalised.astype(np.uint8)

    def save_images(self):
        cv2.imwrite("task_3_detector_output/1_image_grey.jpg", self.image.grey)
        cv2.imwrite("task_3_detector_output/2_dx_display.jpg", self.normalise(self.hough_circles.dx))
        cv2.imwrite("task_3_detector_output/3_dy_display.jpg", self.normalise(self.hough_circles.dy))
        cv2.imwrite("task_3_detector_output/4_gradient_direction.jpg", self.normalise(self.hough_circles.gradient_direction))
        cv2.imwrite("task_3_detector_output/5_gradient_magnitude.jpg", self.normalise(self.hough_circles.gradient_magnitude))
        cv2.imwrite("task_3_detector_output/6_gradient_magnitude_threshold.jpg", self.hough_circles.gradient_magnitude_threshold)
        cv2.imwrite("task_3_detector_output/7_summed_hough_space.jpg", self.normalise(np.sum(self.hough_circles.hough_space, axis = 2)))
        cv2.imwrite("task_3_detector_output/8_output_image.jpg", self.image.image)
    
class Image():
    def __init__(self, file):
        self.image = cv2.imread(file, cv2.IMREAD_COLOR)
        self.grey = cv2.cvtColor(src = self.image, code = cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.grey.shape

class HoughCirclesDetector():
    def __init__(self, image):
        self.image = image
        self.dx = cv2.Sobel(self.image.grey, cv2.CV_64F, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        self.dy = cv2.Sobel(self.image.grey, cv2.CV_64F, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        self.gradient_magnitude = np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.gradient_direction = np.arctan2(self.dy, self.dx)
        self.gradient_magnitude_threshold = self.calculate_gradient_magnitude_threshold()
        self.hough_space = self.calculate_hough_space()
        self.circles = self.calculate_circles()
        self.boxes = [(int(x - r - MINIMUM_RADIUS), int(y - r - MINIMUM_RADIUS), int((r + MINIMUM_RADIUS) * 2), int((r + MINIMUM_RADIUS) * 2)) for x, y, r, _ in self.circles]

    def calculate_gradient_magnitude_threshold(self):
        gradient_magnitude_threshold = self.gradient_magnitude.copy()
        gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
        gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255
        return gradient_magnitude_threshold

    def calculate_hough_space(self):
        radii = MAXIMUM_RADIUS - MINIMUM_RADIUS
        hough_space = np.zeros((self.image.height, self.image.width, radii))
        for x in range(self.image.height):
            for y in range(self.image.width):
                if self.gradient_magnitude_threshold[x][y] == 255:
                    for r in range(radii):
                        x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(self.gradient_direction[x][y]))
                        y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(self.gradient_direction[x][y]))
                        if x_0 > 0 and x_0 < self.image.height and y_0 > 0 and y_0 < self.image.width:
                            hough_space[x_0][y_0][r] += 1
                        x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(self.gradient_direction[x][y] + np.pi))
                        y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(self.gradient_direction[x][y] + np.pi))
                        if x_0 > 0 and x_0 < self.image.height and y_0 > 0 and y_0 < self.image.width:
                            hough_space[x_0][y_0][r] += 1
        return hough_space

    def calculate_possible_circles(self):
        t_h = int(np.max(self.hough_space) * 0.5)
        circles = []
        for y in range(self.image.height):
            for x in range(self.image.width):
                for r in range(MAXIMUM_RADIUS - MINIMUM_RADIUS):
                    if self.hough_space[y][x][r] >= t_h:
                        circles.append([x, y, r, self.hough_space[y][x][r]])
        return circles

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_circles(self):
        circles = []
        for (x1, y1, r1, w1) in self.calculate_possible_circles():
            exists_in_circles = False
            for i, (x2, y2, r2, w2) in enumerate(circles):
                if self.distance(x1, y1, x2, y2) < MIN_DISTANCE:
                    new_w2 = w1 + w2
                    new_x2 = (w2 * x2 + w1 * x1) / (new_w2)
                    new_y2 = (w2 * y2 + w1 * y1) / (new_w2)
                    new_r2 = (w2 * r2 + w1 * r1) / (new_w2)
                    circles[i] = [new_x2, new_y2, new_r2, new_w2]
                    exists_in_circles = True
                    break
            if not exists_in_circles:
                circles.append([x1, y1, r1, w1])
        return circles

class ViolaJonesDetector():
    def __init__(self, image):
        self.image = image
        self.cascade = cv2.CascadeClassifier("training/NoEntrycascade/cascade.xml")
        self.objects = self.cascade.detectMultiScale(
            cv2.equalizeHist(self.image.grey),
            scaleFactor = 1.01,
            minNeighbors = 1,
            flags = cv2.CASCADE_SCALE_IMAGE,
            minSize = (10, 10),
            maxSize = (300, 300)
        )   

if __name__ == "__main__":
    detector = ErrorSignDetector(sys.argv[1])
    detector.draw_boxes()
    detector.save_images()