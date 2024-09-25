
# Armor Tracker

This project implements an armor detection and tracking system using Python and OpenCV. The system detects armor plates based on light detection and performs image processing to track the armor in video frames.

## Features

- **Light Detection**: Detects red or blue lights on armor plates using color thresholding and contour analysis.
- **Armor Detection**: Uses bounding boxes around detected light bars to identify the position of the armor plate.
- **Perspective Transformation**: Extracts numbers from the armor plates by applying perspective transformation to the detected armor region.
- **Real-Time Video Processing**: Processes frames from a video stream to detect and track armor in real-time.
- **Custom Algorithm**: Implements a custom algorithm to locate the armor based on spatial relationships between detected light bars.

## Technologies Used

- **Python**: Main programming language used for implementing the detection system.
- **OpenCV**: Used for image processing, contour detection, and video frame analysis.
- **NumPy**: Utilized for handling arrays and performing mathematical operations.
- **Matplotlib**: Used for visualizing images during the development and testing process.

## Setup and Installation

1. **Install Python and Dependencies**:
   Ensure that Python 3.x is installed on your machine, then install the required libraries using pip:
   ```bash
   pip install numpy opencv-python matplotlib
   ```

2. **Download the Code**:
   Download or clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-repo/armor-tracker.git
   ```

3. **Run the Code**:
   Place your video file (e.g., `1234567.mp4`) in the project directory and run the script to start detecting and tracking armor in the video:
   ```bash
   python armortracker.py
   ```

4. **Control**:
   - Press **'q'** to stop the video and exit the program.

## Functions

- `detect_light(src, color)`: Detects blue or red light bars from the armor plate in the image and returns the positions of the detected light bars.
- `extractNumbers(src, armor)`: Performs perspective transformation to extract the number region from the detected armor and returns a binarized image of the numbers.
- `img_show(name, src)`: Displays an image using OpenCV's `imshow` function.

## How It Works

- The system takes in video frames and detects light bars based on the specified color (red or blue).
- It then calculates the bounding box of the detected light bars and identifies whether it corresponds to armor.
- A perspective transformation is applied to extract and binarize the number region of the armor.
- The system continuously processes each frame of the video in real-time, drawing bounding boxes around detected armor plates.

## Future Improvements

- Improve light detection algorithm for better accuracy in noisy environments.
- Add support for more advanced tracking techniques like Kalman filters.
- Extend the detection to handle more complex and dynamic environments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
