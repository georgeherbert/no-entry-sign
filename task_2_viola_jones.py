import sys
import cv2

class ViolaJonesDetector():
    def __init__(self, file):
        self.image = Image(file)
        self.cascade = cv2.CascadeClassifier("training/NoEntrycascade/cascade.xml")
        self.objects = self.cascade.detectMultiScale(
            cv2.equalizeHist(self.image.grey),
            scaleFactor = 1.1,
            minNeighbors = 1,
            flags = cv2.CASCADE_SCALE_IMAGE,
            minSize = (10, 10),
            maxSize = (300, 300)
        )

    def draw_boxes(self):
        for x, y, w, h in self.objects:
            cv2.rectangle(self.image.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def save_images(self):
        cv2.imwrite("task_2_viola_jones_output/1_image_grey.jpg", self.image.grey)
        cv2.imwrite("task_2_viola_jones_output/2_output_image.jpg", self.image.image)

class Image():
    def __init__(self, file):
        self.image = cv2.imread(file, cv2.IMREAD_COLOR)
        self.grey = cv2.cvtColor(src = self.image, code = cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.grey.shape

if __name__ == "__main__":
    detector = ViolaJonesDetector(sys.argv[1])
    detector.draw_boxes()
    detector.save_images()