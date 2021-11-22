import numpy as np
import cv2 as cv

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
MINIMUM_RADIUS = 10
MAXIMUM_RADIUS = 90

T_S = 250

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

    cv.imwrite("dx_display.jpg", dx_display)
    cv.imwrite("dy_display.jpg", dy_display)
    cv.imwrite("gradient_magnitude.jpg", gradient_magnitude_display)
    cv.imwrite("gradient_direction.jpg", gradient_direction_display)

    return gradient_magnitude, gradient_direction

def hough(gradient_magnitude, gradient_direction):
    image_height = gradient_magnitude.shape[0]
    image_width = gradient_magnitude.shape[1]

    radii = MAXIMUM_RADIUS - MINIMUM_RADIUS

    hough_space = np.zeros((image_height, image_width, radii))

    gradient_magnitude_threshold = gradient_magnitude.copy()
    gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
    gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255

    for x in range(image_height):
        for y in range(image_width):
            if gradient_magnitude_threshold[x][y] == 255:
                for r in range(radii):
                    x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(gradient_direction[x][y]))
                    y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(gradient_direction[x][y]))
                    if x_0 > 0 and x_0 < image_height and y_0 > 0 and y_0 < image_width:
                        hough_space[x_0][y_0][r] += 1
                    # Add pi to the gradient angle so the angle also points in the other direction
                    # We don't know if the gradient is pointing towards or away from the center of the circle.
                    # https://stackoverflow.com/questions/48279460/unable-to-properly-calculate-a-b-space-in-hough-transformation-for-circle-det
                    x_0 = int(x - (r + MINIMUM_RADIUS) * np.sin(gradient_direction[x][y] + np.pi))
                    y_0 = int(y - (r + MINIMUM_RADIUS) * np.cos(gradient_direction[x][y] + np.pi))
                    if x_0 > 0 and x_0 < image_height and y_0 > 0 and y_0 < image_width:
                        hough_space[x_0][y_0][r] += 1

    return hough_space

def display_hough_space(hough_space):
    summed_hough_space = np.sum(hough_space, axis = 2)
    summed_hough_space_display = normalise(summed_hough_space)
    return summed_hough_space_display
    

def main():
    image = cv.imread("coins2.png")
    image_grey = cv.cvtColor(src = image, code = cv.COLOR_BGR2GRAY)

    gradient_magnitude, gradient_direction = sobel(image_grey)

    hough_space = hough(gradient_magnitude, gradient_direction,)
    hough_space_display = display_hough_space(hough_space)
    cv.imwrite("summed_hough_space.jpg", hough_space_display)

    image_height = gradient_magnitude.shape[0]
    image_width = gradient_magnitude.shape[1]

    t_h = int(np.max(hough_space) * 0.7)

    for x in range(image_height):
        for y in range(image_width):
            for r in range(MAXIMUM_RADIUS - MINIMUM_RADIUS):
                if hough_space[x][y][r] >= t_h:
                    print(hough_space[x][y][r])
                    cv.circle(image, (y, x), r + MINIMUM_RADIUS, (0, 0, 255), 2)

    cv.imshow("Display window", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()