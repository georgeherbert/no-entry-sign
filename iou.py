import numpy as np
import cv2

def main():
    with open("faces_detected/coordinates/0.txt") as f:
        lines = f.readlines()
        lines_cleaned = [line.strip() for line in lines]
        print(lines_cleaned)

if __name__ == "__main__":
    main()