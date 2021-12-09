Setup to run the detectors on the lab machines:

1. Create a virtual environment in the same directory as the detectors:
python3 -m venv venv

2. Start the virtual environment:
source venv/bin/activate

3. Upgrade pip:
pip3 install --upgrade pip

4. Install the required libraries:
pip3 install -r requirements.txt

--------------------------------------------------

To run each detector:

- To run the task 2 detector:
python3 task_2_viola_jones.py <image>
e.g. python3 task_2_viola_jones.py No_entry/NoEntry15.bmp

- To run the task 3 detector:
python3 task_3_detector.py <image>
e.g. python3 task_3_detector.py No_entry/NoEntry15.bmp

- To run the task 4 detector:
python3 task_4_detector.py <image>
e.g. python3 task_4_detector.py No_entry/NoEntry15.bmp