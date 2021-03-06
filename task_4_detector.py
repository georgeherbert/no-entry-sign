import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans

MINIMUM_RADIUS = 10
MAXIMUM_RADIUS = 110
MIN_DISTANCE = 50

T_S = 250
# T_H = 100

IOU_THRESHOLD = 0.5

class ErrorSignDetector():
    def __init__(self, file):
        self.image = Image(file)
        self.hough_details = HoughDetails(self.image)
        self.hough_circles = HoughCirclesDetector(self.hough_details)
        self.viola_jones = ViolaJonesDetector(self.image)
        self.objects, self.unsuccessful_circles = self.calculate_circles()
        
        if self.unsuccessful_circles:
            self.colour_line = ColourLineDetector(self.image, self.unsuccessful_circles, self.hough_details)
            self.objects += self.colour_line.colour_circles

    def calculate_circles(self):
        successful_intersections = []
        unsuccessful_intersections = []
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
            if (x1, y1, w1, h1) not in successful_intersections:
                unsuccessful_intersections.append((x1, y1, w1, h1))
        return successful_intersections, unsuccessful_intersections

    def draw_boxes(self):
        for x, y, w, h in self.objects:
            cv2.rectangle(self.image.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # for x, y, r, _ in self.hough_circles.circles:
        #     cv2.circle(self.image.image, (int(x), int(y)), int(r + MINIMUM_RADIUS), (0, 255, 255), 1)
        # for x, y, w, h in self.viola_jones.objects:
        #     cv2.rectangle(self.image.image, (x, y), (x + w, y + h), (255, 255, 0), 1)

    def normalise(self, image):
        image_normalised = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        return image_normalised.astype(np.uint8)

    def save_images(self):
        cv2.imwrite("task_4_detector_output/1_image_grey.jpg", self.image.grey)
        cv2.imwrite("task_4_detector_output/2_dx_display.jpg", self.normalise(self.hough_circles.hough_details.dx))
        cv2.imwrite("task_4_detector_output/3_dy_display.jpg", self.normalise(self.hough_circles.hough_details.dy))
        cv2.imwrite("task_4_detector_output/4_gradient_direction.jpg", self.normalise(self.hough_circles.hough_details.gradient_direction))
        cv2.imwrite("task_4_detector_output/5_gradient_magnitude.jpg", self.normalise(self.hough_circles.hough_details.gradient_magnitude))
        cv2.imwrite("task_4_detector_output/6_gradient_magnitude_threshold.jpg", self.hough_circles.hough_details.gradient_magnitude_threshold)
        cv2.imwrite("task_4_detector_output/7_summed_hough_space.jpg", self.normalise(np.sum(self.hough_circles.hough_space, axis = 2)))
        cv2.imwrite("task_4_detector_output/8_output_image.jpg", self.image.image)
        cv2.imwrite("detected.jpg", self.image.image)
    
class Image():
    def __init__(self, file):
        self.image = cv2.imread(file, cv2.IMREAD_COLOR)
        self.grey = cv2.cvtColor(src = self.image, code = cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.grey.shape

class DistanceDetector():
    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class HoughDetails():
    def __init__(self, image):
        self.image = image
        self.dx = cv2.Sobel(self.image.grey, cv2.CV_64F, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        self.dy = cv2.Sobel(self.image.grey, cv2.CV_64F, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
        self.gradient_magnitude = np.sqrt(self.dx ** 2 + self.dy ** 2)
        self.gradient_direction = np.arctan2(self.dy, self.dx)
        self.gradient_magnitude_threshold = self.calculate_gradient_magnitude_threshold()

    def calculate_gradient_magnitude_threshold(self):
        gradient_magnitude_threshold = self.gradient_magnitude.copy()
        gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
        gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255
        return gradient_magnitude_threshold

class HoughCirclesDetector(DistanceDetector):
    def __init__(self, hough_details):
        self.hough_details = hough_details
        self.hough_space = self.calculate_hough_space()
        self.circles = self.calculate_circles()
        self.boxes = [(int(x - r - MINIMUM_RADIUS), int(y - r - MINIMUM_RADIUS), int((r + MINIMUM_RADIUS) * 2), int((r + MINIMUM_RADIUS) * 2)) for x, y, r, _ in self.circles]

    def calculate_hough_space(self):
        radii = MAXIMUM_RADIUS - MINIMUM_RADIUS
        hough_space = np.zeros((self.hough_details.image.height, self.hough_details.image.width, radii))
        for x in range(self.hough_details.image.height):
            for y in range(self.hough_details.image.width):
                if self.hough_details.gradient_magnitude_threshold[x][y] == 255:
                    for r in range(radii):
                        x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(self.hough_details.gradient_direction[x][y]))
                        y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(self.hough_details.gradient_direction[x][y]))
                        if x_0 > 0 and x_0 < self.hough_details.image.height and y_0 > 0 and y_0 < self.hough_details.image.width:
                            hough_space[x_0][y_0][r] += 1
                        x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(self.hough_details.gradient_direction[x][y] + np.pi))
                        y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(self.hough_details.gradient_direction[x][y] + np.pi))
                        if x_0 > 0 and x_0 < self.hough_details.image.height and y_0 > 0 and y_0 < self.hough_details.image.width:
                            hough_space[x_0][y_0][r] += 1
        return hough_space

    def calculate_possible_circles(self):
        t_h = int(np.max(self.hough_space) * 0.5)
        circles = []
        for y in range(self.hough_details.image.height):
            for x in range(self.hough_details.image.width):
                for r in range(MAXIMUM_RADIUS - MINIMUM_RADIUS):
                    if self.hough_space[y][x][r] >= t_h:
                        circles.append([x, y, r, self.hough_space[y][x][r]])
        return circles

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
            self.image.grey,
            scaleFactor = 1.001,
            minNeighbors = 1,
            flags = cv2.CASCADE_SCALE_IMAGE,
            minSize = (10, 10),
            maxSize = (300, 300)
        )   

class ColourLineDetector(DistanceDetector):
    def __init__(self, image, circles, hough_details):
        self.image = image
        self.hough_details = hough_details
        self.circles = circles
        self.colour_circles = self.calculate_colour_circles()
        
    def fit_range(self, x, y, w, h):
        return max(x, 0), max(y, 0), min(w, self.image.width - x), min(h, self.image.height - y)

    def calculate_circle(self, w, h_fitted, w_fitted, new_image):
        new_image_lab = cv2.cvtColor(new_image, cv2.COLOR_BGR2LAB)
        radius = int(w / 2)
        centre = (radius, radius)
        points_in_circle = []
        for y in range(h_fitted):
            for x in range(w_fitted):
                if self.distance(x, y, *centre) <= radius - int(w / 20):
                    points_in_circle.append(list(new_image_lab[y][x]))
        return np.array(points_in_circle)[:, 1:]

    def calculate_red_white(self, ab):
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(ab)
        red_index = np.argmax(kmeans.cluster_centers_.sum(axis = 1))
        red = kmeans.cluster_centers_[red_index]
        white = kmeans.cluster_centers_[np.abs(red_index - 1)]
        ab_clustered = kmeans.predict(ab)
        red_count = np.count_nonzero(ab_clustered == red_index)
        white_count = np.count_nonzero(ab_clustered == np.abs(red_index - 1))
        return red, white, red_count, white_count

    def calculate_colour_circles(self):
        new_objects = []
        for x, y, w, h, in self.circles:
            x_fitted, y_fitted, w_fitted, h_fitted = self.fit_range(x, y, w, h)
            new_image = self.image.image[y_fitted:y_fitted + h_fitted, x_fitted:x_fitted + w_fitted]
            points_in_circle_ab = self.calculate_circle(w, h_fitted, w_fitted, new_image)
            red, white, red_count, white_count = self.calculate_red_white(points_in_circle_ab)
            if self.distance(*red, 208, 193) < 65 and self.distance(*white, 128, 128) < 15 and red_count > white_count:
                lines = HoughLinesDetector(self.hough_details, x, y, w, h).lines
                if len(lines) == 2:
                    theta_0_abs = np.abs(lines[0][1])
                    theta_1_abs = np.abs(lines[1][1])
                    if theta_0_abs > 1.25 and theta_0_abs < 1.9 and theta_1_abs > 1.25 and theta_1_abs < 1.9 and np.abs(theta_0_abs - theta_1_abs) < 0.2:
                        new_objects.append((x, y, w, h))
        return new_objects

class HoughLinesDetector():
    def __init__(self, hough_details, x, y, w, h):
        self.hough_details = hough_details
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = self.hough_details.image.image[y:y + h, x: x + w]
        self.gradient_direction = self.hough_details.gradient_direction[y:y + h, x: x + w]
        self.gradient_magnitude_threshold = self.hough_details.gradient_magnitude_threshold[y:y + h, x:x + w]
        self.rho_max = int(np.round(np.sqrt(self.w ** 2 + self.h ** 2)))
        self.rhos = np.linspace(-self.rho_max, self.rho_max, self.rho_max * 2)
        self.thetas = np.deg2rad(np.arange(-180, 180))
        self.hough_space = self.calculate_hough_space()
        self.lines = self.calculate_lines()

    def calculate_hough_space(self):
        hough_space = np.zeros((2 * self.rho_max, len(self.thetas)))
        for y in range(self.h):
            for x in range(self.w):
                if self.gradient_magnitude_threshold[y][x]:
                    for i, theta in enumerate(self.thetas):
                        if theta > self.gradient_direction[y][x] - 0.005 and theta < self.gradient_direction[y][x] + 0.005:
                            rho = x * np.cos(theta) + y * np.sin(theta)
                            hough_space[int(rho) + self.rho_max][i] += 1
        return hough_space

    def calculate_possible_lines(self):
        t_h = np.max(self.hough_space) * 0.5
        lines = []
        for i in range(self.rho_max * 2):
            for j in range(len(self.thetas)):
                if self.hough_space[i][j] > t_h:
                    rho = self.rhos[i]
                    theta = self.thetas[j]
                    lines.append((rho, theta))
        return lines

    def calculate_lines(self):
        possible_lines = self.calculate_possible_lines()
        correct_lines = []
        rho_threshold = self.rho_max / 10
        theta_threshold = 0.1
        for rho1, theta1 in possible_lines:
            in_correct_lines = False
            for rho2, theta2 in correct_lines:
                if np.abs(rho1 - rho2) < rho_threshold and np.abs(theta1 - theta2) < theta_threshold:
                    in_correct_lines = True
            if not in_correct_lines:
                correct_lines.append((rho1, theta1))
        return correct_lines

if __name__ == "__main__":
    detector = ErrorSignDetector(sys.argv[1])
    detector.draw_boxes()
    detector.save_images()