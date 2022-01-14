# No Entry Sign

'No Entry Sign' is a piece of coursework I produced for the Image Processing and Computer Vision (COMS20011) module at the University of Bristol.

## Description

The coursework instructions are in the file [instructions.pdf](instructions.pdf).

The program takes the location of a bitmap image file (BMP) as input to the program. The program then identifies no entry signs in the image file, draws bounding boxes, and saves the image with bounding boxes to detected.jpg.

The coursework report is in the file [report.pdf](report/report.pdf).

## Getting Started

To begin with, clone the repository:

```bash
git clone https://github.com/georgeherbert/no-entry-sign.git
```

Install the dependencies:
```bash
pip3 install -r requirements.txt
```

To run the program on the file [No_entry/NoEntry10.bmp](No_entry/NoEntry10.bmp):
```bash
python3 task_4_detector.py No_entry/NoEntry10.bmp
```