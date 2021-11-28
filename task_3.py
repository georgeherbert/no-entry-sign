import sys
import numpy as np
import cv2

MINIMUM_RADIUS = 10
MAXIMUM_RADIUS = 110
MIN_DISTANCE = 50

T_S = 250
# T_H = 100

IOU_THRESHOLD = 0.5

class Image():
    def __init__(self, file):
        self.image = cv2.imread(file, cv2.IMREAD_COLOR)
        self.image_grey = cv2.cvtColor(src = self.image, code = cv2.COLOR_BGR2GRAY)
        self.dx = cv2.Sobel(self.image_grey, cv2.CV_64F, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        self.dy = cv2.Sobel(self.image_grey, cv2.CV_64F, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        self.gradient_magnitude = np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.gradient_direction = np.arctan2(self.dy, self.dx)
        self.hough_circles = HoughSpaceCircles(self.image, self.gradient_magnitude, self.gradient_direction)
        self.viola_jones_objects = ViolaJonesObjects(self.image, self.image_grey)
        self.error_sign_detector = ErrorSignDetector(self.image, self.hough_circles.boxes, self.viola_jones_objects.objects)

        # self.hough_circles.draw_circles()
        self.viola_jones_objects.draw_boxes()
        self.save_images()

    def normalise(self, image):
        image_normalised = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        return image_normalised.astype(np.uint8)

    def save_images(self):
        cv2.imwrite("task_3/1_image_grey.jpg", self.image_grey)
        cv2.imwrite("task_3/2_dx_display.jpg", self.normalise(self.dx))
        cv2.imwrite("task_3/3_dy_display.jpg", self.normalise(self.dy))
        cv2.imwrite("task_3/4_gradient_direction.jpg", self.normalise(self.gradient_direction))
        cv2.imwrite("task_3/5_gradient_magnitude.jpg", self.normalise(self.gradient_magnitude))
        cv2.imwrite("task_3/6_gradient_magnitude_threshold.jpg", self.hough_circles.gradient_magnitude_threshold)
        cv2.imwrite("task_3/7_summed_hough_space.jpg", self.normalise(np.sum(self.hough_circles.hough_space, axis = 2)))
        cv2.imwrite("task_3/8_output_image.jpg", self.image)
        
class HoughSpaceCircles():
    def __init__(self, image, gradient_magnitude, gradient_direction):
        self.image = image
        self.gradient_magnitude = gradient_magnitude
        self.gradient_direction = gradient_direction
        self.gradient_magnitude_threshold = self.get_gradient_magnitude_threshold()
        self.height = self.gradient_magnitude.shape[0]
        self.width = self.gradient_magnitude.shape[1]
        self.hough_space = self.get_hough_space()
        self.possible_circles = self.get_possible_circles()
        self.circles = self.get_circles()
        self.boxes = [(int(x - r - MINIMUM_RADIUS), int(y - r - MINIMUM_RADIUS), int((r + MINIMUM_RADIUS) * 2), int((r + MINIMUM_RADIUS) * 2)) for x, y, r, _ in self.circles]

    def get_gradient_magnitude_threshold(self):
        gradient_magnitude_threshold = self.gradient_magnitude.copy()
        gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
        gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255
        return gradient_magnitude_threshold

    def get_hough_space(self):
        radii = MAXIMUM_RADIUS - MINIMUM_RADIUS
        hough_space = np.zeros((self.height, self.width, radii))
        for x in range(self.height):
            for y in range(self.width):
                if self.gradient_magnitude_threshold[x][y] == 255:
                    for r in range(radii):
                        x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(self.gradient_direction[x][y]))
                        y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(self.gradient_direction[x][y]))
                        if x_0 > 0 and x_0 < self.height and y_0 > 0 and y_0 < self.width:
                            hough_space[x_0][y_0][r] += 1
                        x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(self.gradient_direction[x][y] + np.pi))
                        y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(self.gradient_direction[x][y] + np.pi))
                        if x_0 > 0 and x_0 < self.height and y_0 > 0 and y_0 < self.width:
                            hough_space[x_0][y_0][r] += 1
        return hough_space

    def get_possible_circles(self):
        t_h = int(np.max(self.hough_space) * 0.5)
        circles = []
        for y in range(self.height):
            for x in range(self.width):
                for r in range(MAXIMUM_RADIUS - MINIMUM_RADIUS):
                    if self.hough_space[y][x][r] >= t_h:
                        circles.append([x, y, r, self.hough_space[y][x][r]])
        return circles

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_circles(self):
        circles = []
        for (x1, y1, r1, w1) in self.possible_circles:
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

    def draw_circles(self):
        for x, y, r, _ in self.circles:
            cv2.circle(self.image, (int(x), int(y)), int(r) + MINIMUM_RADIUS, (255, 0, 0), 2)
        # for x, y, w, h in self.boxes:
        #     cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2) 

class ViolaJonesObjects():
    def __init__(self, image, image_grey):
        self.image = image
        self.cascade = cv2.CascadeClassifier("NoEntrycascade/cascade.xml")
        self.objects = self.cascade.detectMultiScale(
            image_grey,
            scaleFactor = 1.1,
            minNeighbors = 1,
            flags = cv2.CASCADE_SCALE_IMAGE,
            minSize = (10, 10),
            maxSize = (300, 300)
        )
    
    def draw_boxes(self):
        for x, y, w, h in self.objects:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2) 

class ErrorSignDetector():
    def __init__(self, image, hough_boxes, viola_jones_boxes):
        self.image = image
        self.hough_boxes = hough_boxes
        self.viola_jones_boxes = viola_jones_boxes
        self.custom_boxes = self.calc_successful_intersections()

        for x, y, w, h in self.custom_boxes:
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    def calc_successful_intersections(self):
        successful_intersections = []
        for x1, y1, w1, h1 in self.hough_boxes:
            potential_intersections = []
            for x2, y2, w2, h2 in self.viola_jones_boxes:
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)
                if x_right >= x_left and y_bottom >= y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    hough_box_area = w1 * h1
                    viola_jones_area = w2 * h2
                    iou = intersection_area / (hough_box_area + viola_jones_area - intersection_area)
                    potential_intersections.append([(x2, y2, w2, h2), iou])
            if potential_intersections:
                largest_intersection = max(potential_intersections, key = lambda x: x[1])
                if largest_intersection[1] > IOU_THRESHOLD:
                    successful_intersections.append(largest_intersection[0])
        return successful_intersections

if __name__ == "__main__":
    Image(sys.argv[1])