import sys
import cv2
import numpy as np

T_S = 250

def normalise(to_normalise):
    image_normalised = 255 * (to_normalise - np.min(to_normalise)) / (np.max(to_normalise) - np.min(to_normalise))
    return image_normalised.astype(np.uint8)

image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
grey = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)

dx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
dy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize = 3, scale = 1, delta = 0, borderType = cv2.BORDER_DEFAULT)
gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
gradient_direction = np.arctan2(dy, dx)

gradient_magnitude_threshold = gradient_magnitude.copy()
gradient_magnitude_threshold[gradient_magnitude_threshold < T_S] = 0
gradient_magnitude_threshold[gradient_magnitude_threshold >= T_S] = 255

y_len, x_len = grey.shape
rho_max = int(np.round(np.sqrt(y_len ** 2 + x_len ** 2)))
rhos = np.linspace(-rho_max, rho_max, rho_max * 2)

thetas = np.deg2rad(np.arange(-180, 180))

hough_space = np.zeros((2 * rho_max, len(thetas)))

# print(np.max(gradient_direction * 57.2958))

for y in range(y_len):
    for x in range(x_len):
        if gradient_magnitude_threshold[y][x]:
            for i, theta in enumerate(thetas):
                if theta > gradient_direction[y][x] - 0.005 and theta < gradient_direction[y][x] + 0.005 and (np.abs(theta) > np.deg2rad(70) and np.abs(theta) < np.deg2rad(110)):
                    rho = x * np.cos(theta) + y * np.sin(theta)
                    hough_space[int(rho) + rho_max][i] += 1

print(np.max(hough_space))
t_h = np.max(hough_space) * 0.5

lines = []
for i in range(rho_max * 2):
    for j in range(len(thetas)):
        if hough_space[i][j] > t_h:
            rho = rhos[i]
            theta = thetas[j]
            lines.append((rho, theta))
            # x1 = 0
            # y1 = int(rho / np.sin(theta))
            # x2 = x_len
            # y2 = int((rho / np.sin(theta)) - (x2 / np.tan(theta)))
            # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

correct_lines = []
rho_threshold = rho_max / 100
theta_threshold = 0.05
for rho1, theta1 in lines:
    in_correct_lines = False
    for rho2, theta2 in correct_lines:
        if np.abs(rho1 - rho2) < rho_threshold and np.abs(theta1 - theta2) < theta_threshold:
            in_correct_lines = True
    if not in_correct_lines:
        correct_lines.append((rho1, theta1))

for rho, theta in correct_lines:
    print(rho, theta)
    x1 = 0
    y1 = int(rho / np.sin(theta))
    x2 = x_len
    y2 = int((rho / np.sin(theta)) - (x2 / np.tan(theta)))
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

normalised = normalise(hough_space)
cv2.imshow("", normalised)
cv2.waitKey()
cv2.imshow("", image)
cv2.waitKey()

