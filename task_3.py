import sys
import numpy as np
import cv2

MINIMUM_RADIUS = 10
MAXIMUM_RADIUS = 110
MIN_DISTANCE = 20

T_S = 250
T_H = 100

def normalise(image):
    image_normalised = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    image_normalised_uint8 = image_normalised.astype(np.uint8)
    return image_normalised_uint8

def sobel(image):
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)

    dx_display = normalise(dx)
    dy_display = normalise(dy)

    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    gradient_magnitude_display = normalise(gradient_magnitude)

    gradient_direction = np.arctan2(dy, dx)
    gradient_direction_display = normalise(gradient_direction)

    cv2.imwrite("task_3/2_dx_display.jpg", dx_display)
    cv2.imwrite("task_3/3_dy_display.jpg", dy_display)
    cv2.imwrite("task_3/5_gradient_magnitude.jpg", gradient_magnitude_display)
    cv2.imwrite("task_3/4_gradient_direction.jpg", gradient_direction_display)

    return gradient_magnitude, gradient_direction

def hough_circles(gradient_magnitude, gradient_direction):
    image_height = gradient_magnitude.shape[0]
    image_width = gradient_magnitude.shape[1]

    radii = MAXIMUM_RADIUS - MINIMUM_RADIUS

    hough_space = np.zeros((image_height, image_width, radii))

    gradient_magnitude_threshold = gradient_magnitude.copy()
    gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
    gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255
    cv2.imwrite("task_3/6_gradient_magnitude_threshold.jpg", gradient_magnitude_threshold)

    for x in range(image_height):
        for y in range(image_width):
            if gradient_magnitude_threshold[x][y] == 255:
                for r in range(radii):
                    x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(gradient_direction[x][y]))
                    y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(gradient_direction[x][y]))
                    if x_0 > 0 and x_0 < image_height and y_0 > 0 and y_0 < image_width:
                        hough_space[x_0][y_0][r] += 1
                    x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(gradient_direction[x][y] + np.pi))
                    y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(gradient_direction[x][y] + np.pi))
                    if x_0 > 0 and x_0 < image_height and y_0 > 0 and y_0 < image_width:
                        hough_space[x_0][y_0][r] += 1

    return hough_space

def display_hough_space(hough_space):
    summed_hough_space = np.sum(hough_space, axis = 2)
    summed_hough_space_display = normalise(summed_hough_space)
    return summed_hough_space_display

def viola_jones_detect(image_grey):
    cascade = cv2.CascadeClassifier("NoEntryCascade/cascade.xml")
    objects_detected = cascade.detectMultiScale(
        image_grey,
        scaleFactor = 1.1,
        minNeighbors = 1,
        flags = cv2.CASCADE_SCALE_IMAGE,
        minSize = (10, 10),
        maxSize = (300, 300)
    )
    return objects_detected

def get_possible_circles(image_height, image_width, hough_space_circles):
    t_h = int(np.max(hough_space_circles) * 0.5)
    circles = []
    for x in range(image_height):
        for y in range(image_width):
            for r in range(MAXIMUM_RADIUS - MINIMUM_RADIUS):
                if hough_space_circles[x][y][r] >= t_h:
                    circles.append([x, y, r, hough_space_circles[x][y][r]])
                    # cv2.circle(image, (y, x), r + MINIMUM_RADIUS, (255, 0, 0), 2)
    return circles

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_distinct_circles(circles):
    distinct_circles = []
    for (x1, y1, r1, w1) in circles:
        exists_in_distinct = False
        for i, (x2, y2, r2, w2) in enumerate(distinct_circles):
            if distance(x1, y1, x2, y2) < MIN_DISTANCE:
                new_w2 = w1 + w2
                new_x2 = (w2 * x2 + w1 * x1) / (new_w2)
                new_y2 = (w2 * y2 + w1 * y1) / (new_w2)
                new_r2 = (w2 * r2 + w1 * r1) / (new_w2)
                distinct_circles[i] = [new_x2, new_y2, new_r2, new_w2]
                exists_in_distinct = True
                break
        if not exists_in_distinct:
            distinct_circles.append([x1, y1, r1, w1])
    return distinct_circles

def main():
    image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    image_grey = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
    cv2.imwrite("task_3/1_image.jpg", image)

    gradient_magnitude, gradient_direction = sobel(image_grey)

    hough_space_circles = hough_circles(gradient_magnitude, gradient_direction)
    hough_space_circles_display = display_hough_space(hough_space_circles)
    cv2.imwrite("task_3/7_summed_hough_space.jpg", hough_space_circles_display)

    # t_h = 0.5 * hough_space_circles.max()
    # hough_space_circles_threshold = hough_space_circles_display.copy()
    # hough_space_circles_threshold[hough_space_circles_threshold < t_h] = 0
    # hough_space_circles_threshold[hough_space_circles_threshold >= t_h] = 255
    # cv2.imwrite("task_3/8_summed_hough_space_threshold.jpg", hough_space_circles_threshold)

    objects_detected = viola_jones_detect(image_grey)
    for (x, y, w, h) in objects_detected:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    image_height = gradient_magnitude.shape[0]
    image_width = gradient_magnitude.shape[1]

    circles = get_possible_circles(image_height, image_width, hough_space_circles)
    distinct_circles = get_distinct_circles(circles)

    for (x, y, r, h) in distinct_circles:
        print(x, y)
        cv2.circle(image, (int(y), int(x)), int(r) + MINIMUM_RADIUS, (255, 0, 0), 2)

    cv2.imwrite("task_3/8_output_image.jpg", image)
    # cv2.imshow("Display window", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()