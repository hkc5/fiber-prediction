import cv2
import math

def calculate_angle(point1, point2):
    # Calculate the angle in degrees between two points
    x1, y1 = point1
    x2, y2 = point2

    delta_x = x2 - x1
    delta_y = y2 - y1

    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    cv2.imshow('Image', image)

    # Wait for the user to click two points
    print("Click on two points in the image to calculate the angle.")
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) == 2:
                angle = calculate_angle(points[0], points[1])
                print(f"Angle between points: {angle:.2f} degrees")
                cv2.destroyAllWindows()

    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)

if __name__ == "__main__":
    image_path = "./diffusion/diffusion_voxels/curated"  # Replace with your image file path
    main(image_path)
