import sys
import numpy as np
import cv2

KERNEL_DERIVATIVE_X = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])
KERNEL_DERIVATIVE_Y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])
MINIMUM_RADIUS = 20
MAXIMUM_RADIUS = 90

T_S = 250
T_H = 100

def convolution(image, kernel):
    kernel_flipped = np.flip(kernel)
    output = np.zeros(image.shape)
    image = np.pad(image, pad_width = 1)
    for i in range(len(output)):
        for j in range(len(output[i])):
            slice = image[i: i + 3, j: j + 3]
            multiplied = kernel_flipped * slice
            total = multiplied.sum()
            output[i][j] = total
    return output

def normalise(image):
    image_normalised = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))
    image_normalised_uint8 = image_normalised.astype(np.uint8)
    return image_normalised_uint8

def sobel(image):
    dx = convolution(image, KERNEL_DERIVATIVE_X)
    dy = convolution(image, KERNEL_DERIVATIVE_Y)
    dx_display = normalise(dx)
    dy_display = normalise(dy)

    gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
    gradient_magnitude_display = normalise(gradient_magnitude)

    gradient_direction = np.arctan2(dy, dx)
    gradient_direction_display = normalise(gradient_direction)

    cv2.imwrite("task_3/dx_display.jpg", dx_display)
    cv2.imwrite("task_3/dy_display.jpg", dy_display)
    cv2.imwrite("task_3/gradient_magnitude.jpg", gradient_magnitude_display)
    cv2.imwrite("task_3/gradient_direction.jpg", gradient_direction_display)

    return gradient_magnitude, gradient_direction

def hough(gradient_magnitude, gradient_direction):
    image_height = gradient_magnitude.shape[0]
    image_width = gradient_magnitude.shape[1]

    radii = MAXIMUM_RADIUS - MINIMUM_RADIUS

    hough_space = np.zeros((image_height, image_width, radii))

    gradient_magnitude_threshold = gradient_magnitude.copy()
    gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
    gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255
    cv2.imwrite("task_3/gradient_magnitude_threshold.jpg", gradient_magnitude_threshold)

    for x in range(image_height):
        for y in range(image_width):
            if gradient_magnitude_threshold[x][y] == 255:
                for r in range(radii):
                    x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(gradient_direction[x][y]))
                    y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(gradient_direction[x][y]))
                    if x_0 > 0 and x_0 < image_height and y_0 > 0 and y_0 < image_width:
                        hough_space[x_0][y_0][r] += 1
                    # x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(gradient_direction[x][y] + np.pi))
                    # y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(gradient_direction[x][y] + np.pi))
                    # if x_0 > 0 and x_0 < image_height and y_0 > 0 and y_0 < image_width:
                    #     hough_space[x_0][y_0][r] += 1

    return hough_space

def display_hough_space(hough_space):
    summed_hough_space = np.sum(hough_space, axis = 2)
    summed_hough_space_display = normalise(summed_hough_space)
    return summed_hough_space_display

def main():
    image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    image_grey = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)

    gradient_magnitude, gradient_direction = sobel(image_grey)

    hough_space = hough(gradient_magnitude, gradient_direction,)
    hough_space_display = display_hough_space(hough_space)
    cv2.imwrite("task_3/summed_hough_space.jpg", hough_space_display)

    hough_space_threshold = hough_space_display.copy()
    hough_space_threshold[hough_space_threshold < T_H] = 0
    hough_space_threshold[hough_space_threshold >= T_H] = 255
    cv2.imwrite("task_3/summed_hough_space_threshold.jpg", hough_space_threshold)

    image_height = gradient_magnitude.shape[0]
    image_width = gradient_magnitude.shape[1]

    t_h = int(np.max(hough_space) * 0.7)

    for x in range(image_height):
        for y in range(image_width):
            for r in range(MAXIMUM_RADIUS - MINIMUM_RADIUS):
                if hough_space[x][y][r] >= t_h:
                    print(hough_space[x][y][r])
                    cv2.circle(image, (y, x), r + MINIMUM_RADIUS, (0, 0, 255), 2)

    cv2.imshow("Display window", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()