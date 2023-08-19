import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def calculate_angle(point1, point2):
    # Calculate the angle in degrees between two points
    x1, y1 = point1
    x2, y2 = point2
    delta_x = x2 - x1
    delta_y = y2 - y1

    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = -math.degrees(angle_radians)

    return int(angle_degrees)

def main(image_path):
    # Load the image
    img = mpimg.imread(image_path)

    # Display the image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    plt.title("Click on two points in the image to calculate the angle.")

    # Initialize variables for clicked points
    points = []

    def onclick(event):
        nonlocal points

        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))

            # Display a cursor at the clicked point
            ax.plot(event.xdata, event.ydata, 'rs', markersize=5)
            plt.draw()

            if len(points) == 2:
                angle = calculate_angle(points[0], points[1])
                print(f"Angle between points: {angle:.2f} degrees")
                plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == "__main__":
    image_path = "./diffusion/diffusion_voxels/curated/0fa_200_450.png"  # Replace with your image file path
    main(image_path)
