import math


def calculate_angle(p1, p2):
    """Calculate the angle in degrees between two points."""
    x1, y1 = p1
    x2, y2 = p2
    delta_x = x2 - x1
    delta_y = y2 - y1

    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = -math.degrees(angle_radians)

    if angle_degrees < 0:  # Normalize the angle between 0-180
        angle_degrees += 180

    if p1 == p2:  # Two same points
        angle_degrees = -1

    return int(angle_degrees)
