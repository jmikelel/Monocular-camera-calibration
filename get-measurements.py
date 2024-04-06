import cv2
import numpy as np
import json
import argparse

selected_points = []

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

def compute_distance(selected_points, calibration_data, real_distance_cm):
    """
    Computes the distance in centimeters between two selected points on the image.

    Args:
        selected_points: A list of tuples representing selected points (x, y).
        calibration_data: A dictionary containing camera calibration parameters.
        real_distance_cm: The real distance between the selected points in centimeters.

    Returns:
        The distance between the selected points in centimeters.
    """
    # Calculate distance in pixels
    distance_pixels = np.linalg.norm(np.array(selected_points[0]) - np.array(selected_points[1]))

    # Retrieve focal length from calibration data
    focal_length = calibration_data["camera_matrix"][0][0]  # Assuming focal length is in pixels

    # Convert distance from pixels to centimeters
    distance_cm = (real_distance_cm * focal_length) / distance_pixels

    return distance_cm

def on_mouse_click(event, x, y, flags, param):
    """
    Callback function for mouse events.
    """
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point on left click
        selected_points.append((x, y))
        print(f"Point selected: ({x}, {y})")
        if len(selected_points) == 2:
            # Calculate distance when two points are selected
            distance_cm = compute_distance(selected_points, calibration_data, real_distance_cm=10.0)
            print(f"Distance between points: {distance_cm:.2f} cm")
    elif event == cv2.EVENT_MBUTTONDOWN:
        # Stop selecting on middle click
        print("Selection stopped.")

def draw_lines_between_points(image, points):
    """
    Draws lines between consecutive points on the image.

    Args:
        image: The image to draw on.
        points: A list of tuples representing selected points (x, y).

    Returns:
        The image with lines drawn between points.
    """
    line_thickness = 5  # Adjust thickness as desired
    for i in range(len(points)):
        point1 = points[i]
        point2 = points[(i + 1) % len(points)]
        cv2.line(image, point1, point2, (0, 255, 0), line_thickness)  # Green color
    return image

def draw_selected_points(image, points):
    """
    Draws selected points on the image.

    Args:
        image: The image to draw on.
        points: A list of tuples representing selected points (x, y).

    Returns:
        The image with selected points drawn.
    """
    circle_radius = 5  # Adjust radius as desired
    circle_thickness = 2  # Adjust thickness as desired
    for point in points:
        cv2.circle(image, point, circle_radius, (0, 255, 255), circle_thickness)  # Yellow circle
    return image

def shutdown_on_right_click(event, x, y, flags, param):
    """
    Callback function to close the window on right click.
    """
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()

def main():
    global selected_points, calibration_data

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Vision-based measurement system")
    parser.add_argument("--cam_index", type=int, default=0, help="Camera index")
    parser.add_argument("--cal_file", type=str, required=True, help="Camera calibration file")
    args = parser.parse_args()

    # Load camera calibration data
    with open(args.cal_file, "r") as f:
        calibration_data = json.load(f)

    # Initialize camera capture
    cap = cv2.VideoCapture(args.cam_index)

    # Set mouse callback function
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse_click)
    cv2.setMouseCallback("Frame", shutdown_on_right_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Draw lines between selected points
        frame_with_lines = draw_lines_between_points(frame.copy(), selected_points)

        # Draw selected points
        frame_with_points = draw_selected_points(frame_with_lines, selected_points)

        # Show the frame with lines and selected points
        cv2.imshow("Frame", frame_with_points)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
