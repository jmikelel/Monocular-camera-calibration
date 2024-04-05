import cv2
import numpy as np
import json
import argparse
import os

"""
Vision-based measurement system for calculating distances and perimeter based on selected points in an image.

Usage:
python get-measurements.py --cam_index 0 --Z 56 --cal_file calibration_data.json

Arguments:
--cam_index: Index of the camera to use (default: 0)
--Z: Distance from camera to object (in cm)
--cal_file: Path to the camera calibration file
"""

selected_points = []  # Define selected_points as a global variable

def undistort_image(distorted_image, calibration_data):
    """
    Undistorts an image using camera calibration parameters.

    Args:
        distorted_image: The distorted image as a NumPy array.
        calibration_data: A dictionary containing camera calibration parameters.

    Returns:
        The undistorted image as a NumPy array.
    """
    # Load camera matrix and distortion coefficients
    camera_matrix = np.asarray(calibration_data["camera_matrix"])
    distortion_coefficients = np.asarray(calibration_data["distortion_coefficients"])

    # Undistort the image
    undistorted_image = cv2.undistort(distorted_image, camera_matrix, distortion_coefficients)
    return undistorted_image

def compute_line_segments(selected_points):
    """
    Computes the length of each line segment formed by consecutive points.

    Args:
        selected_points: A list of tuples representing selected points (x, y).

    Returns:
        A list of distances between consecutive points.
    """
    distances = []
    for i in range(len(selected_points)):
        point1 = selected_points[i]
        point2 = selected_points[(i + 1) % len(selected_points)]
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        distances.append(distance)
    return distances

def compute_perimeter(distances):
    """
    Computes the perimeter by summing up the distances between consecutive points.

    Args:
        distances: A list of distances between consecutive points.

    Returns:
        The total perimeter.
    """
    return sum(distances)

def shutdown_on_enter(cap):
    global selected_points
    print("Press Enter to shutdown the program...")
    input()
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

def main():
    global selected_points

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vision-based measurement system")
    parser.add_argument("--cam_index", type=int, default=0, help="Camera index")
    parser.add_argument("--Z", type=float, required=True, help="Distance from camera to object (cm)")
    parser.add_argument("--cal_file", type=str, required=True, help="Camera calibration file")
    args = parser.parse_args()

    # Load camera calibration data
    with open(args.cal_file, "r") as f:
        calibration_data = json.load(f)

    # Initialize camera capture
    cap = cv2.VideoCapture(args.cam_index)

    # Define measurement flag
    measuring = False

    def on_mouse_click(event, x, y, flags, param):
        nonlocal measuring
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point on left click
            selected_points.append((x, y))
            print(f"Point selected: ({x}, {y})")
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Stop selecting on middle click
            measuring = False
            print("Selection stopped.")
            if len(selected_points) >= 2:
                # Perform measurements
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    return

                # Undistort the frame before processing
                undistorted_image = undistort_image(frame.copy(), calibration_data)

                distances = compute_line_segments(selected_points)
                perimeter = compute_perimeter(distances)

                print("Distances between consecutive points:")
                for i in range(len(distances)):
                    print(f"- Line segment {i+1}: {distances[i]:.2f} pixels")  # Format with 2 decimal places

                print(f"Perimeter: {perimeter:.2f}")

    # Set mouse callback function
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse_click)

    shutdown_on_enter(cap)

if __name__ == "__main__":
    main()
