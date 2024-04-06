import cv2
import numpy as np
import json
import argparse

"""
Vision-based measurement system for calculating distances and perimeter based on selected points in an image.

Usage:
python get-measurements.py --cam_index 0 --Z 56 --cal_file calibration_data.json

Arguments:
--cam_index: Index of the camera to use (default: 0)
--Z: Distance from camera to object (in cm)
--cal_file: Path to the camera calibration file
"""

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

def compute_distance(point1, point2, Z):
    """
    Computes the distance between two points in centimeters.

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).
        Z: Distance from camera to object (in cm).

    Returns:
        The distance between the two points in centimeters.
    """
    # Euclidean distance in pixels
    distance_pixels = np.linalg.norm(np.array(point1) - np.array(point2))
    # Convert distance to centimeters using Z
    distance_cm = (distance_pixels * Z) / focal_length
    return distance_cm

def compute_line_segments(selected_points, Z):
    """
    Computes the length of each line segment formed by consecutive points.

    Args:
        selected_points: A list of tuples representing selected points (x, y).
        Z: Distance from camera to object (in cm).

    Returns:
        A list of distances between consecutive points in centimeters.
    """
    distances = []
    for i in range(len(selected_points)):
        point1 = selected_points[i]
        point2 = selected_points[(i + 1) % len(selected_points)]
        distance = compute_distance(point1, point2, Z)
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
            # Calculate and print distance if there are only two points
            distance = compute_distance(selected_points[0], selected_points[1], args.Z)
            print(f"Distance: {distance:.2f} cm")
        elif len(selected_points) > 2:
            # Calculate and print perimeter if there are more than two points
            distances = compute_line_segments(selected_points, args.Z)
            perimeter = compute_perimeter(distances)
            print(f"Perimeter: {perimeter:.2f} cm")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Close the window on right click
        cv2.destroyAllWindows()

def draw_lines_between_points(image, points):
  """
  Draws lines between consecutive points on the image.

  Args:
      image: The image to draw on.
      points: A list of tuples representing selected points (x, y).

  Returns:
      The image with lines drawn between points.
  """
  for i in range(len(points)):
    point1 = points[i]
    point2 = points[(i + 1) % len(points)]
    cv2.line(image, point1, point2, (0, 255, 0), 2)  # Green color
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

def main():
    global args
    global focal_length

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

    # Set mouse callback function
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", on_mouse_click)

    # Get the focal length from camera matrix
    focal_length = calibration_data["camera_matrix"][0][0]

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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

##ya deberia jalar esta cosa. Parece que la calibracion esta afectando 
##demasiado al codigo ya que en un rectangulo de 10x5 me da medidas muy MUY bajas
##lo bueno es que se arreglo la camara, los puntos y las lineas y aunque parece que
##no calcula bien tomando en cuenta la profundidad al menos hace las sumas y distancias
##usando los pixeles