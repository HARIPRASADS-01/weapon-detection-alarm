# Real-Time  Weapon Detection Alert System

This project implements a real-time object detection system using a custom-trained YOLOv5 model (`weapon.pt`) to detect weapons in video streams. The system triggers an alarm when a weapon is detected in the video feed. The project is written in Python and uses PyTorch, OpenCV, and Pygame for real-time detection and alerting.

## Features

- **Real-time Weapon Detection**: Detects weapon in a live video stream using the YOLOv5 model.
- **Weapon Detection**: Triggers an alarm when a weapon is detected in the video feed.
- **High FPS**: Optimized for fast processing to maintain high frames per second (FPS).
- **Custom Detection Model**: You can customize the model by training your own YOLOv5 model or use the provided `weapon.pt`.

## Prerequisites

- **Python 3.6+**
- **CUDA-compatible GPU** (optional but recommended for faster inference)
  
## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/object-detection-with-alarm.git
    cd object-detection-with-alarm
    ```

2. **Install dependencies**:

    Make sure to install all required Python packages by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download YOLOv5 weights**:
   
   You need the custom-trained model `weapon.pt`. Place this file in the root of the project directory.
   
4. **Run the application**:

    ```bash
    python weapon_detection.py
    ```

    The system will start capturing video from your webcam and perform real-time object detection.

## Usage

- Press `q` to exit the application.
- Make sure to have `alarm.wav` in the root folder for the alarm sound to work when a weapon is detected.

## Customizing the Model

If you want to use your own YOLOv5 model for different object detection tasks:

1. Train a YOLOv5 model with your custom dataset.
2. Replace the `weapon.pt` file in the project directory with your new model's weights.
3. Update the code as needed (e.g., change the confidence threshold, classes, etc.).

## Dependencies

- **torch**: The core deep learning library for loading and using the YOLOv5 model.
- **opencv-python**: Used for real-time video processing and display.
- **pygame**: Handles the alarm sound playback.
- **numpy**: For numerical operations on frames and calculations.

## Requirements

The `requirements.txt` file contains all the necessary dependencies to run the project. Install them via `pip install -r requirements.txt`.

