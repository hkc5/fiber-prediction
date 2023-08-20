import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os

def calculate_angle(p1, p2):
    # Calculate the angle in degrees between two points
    x1, y1 = p1
    x2, y2 = p2
    delta_x = x2 - x1
    delta_y = y2 - y1

    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = -math.degrees(angle_radians)

    if angle_degrees < 0: # Normalize the angle between 0-180
        angle_degrees += 180

    if p1 == p2: # Two same points 
        angle_degrees= -1

    return int(angle_degrees)

def get_angle(image_path):
    # Load the image
    img = mpimg.imread(image_path)

    # Display the image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap= "gray")
    plt.title("Click on two points in the image to calculate the angle.")

    # Initialize variables for clicked points
    points= []
    angle= 0
    def onclick(event):
        nonlocal points, angle

        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))

            # Display a cursor at the clicked point
            ax.plot(event.xdata, event.ydata, 'rs', markersize=5)
            plt.draw()

            if len(points) == 2:
                angle = calculate_angle(points[0], points[1])
                # print(f"Angle between points: {angle:.2f} degrees")
                plt.close()
                
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return angle

def get_angle_list(csv_file):
    angles= []

    try:
        with open(csv_file, 'r') as file:
            for line in file:
                angle = int(line.strip())  # Remove leading/trailing whitespace
                angles.append(angle)

    except FileNotFoundError:
        with open(csv_file, 'w') as file:
            print("Angle CSV file created.")

    return angles


if __name__ == "__main__":
    curated_dir= "./diffusion/diffusion_voxels/curated/"  # Replace with your image file path
    csv_dir= "./diffusion/diffusion_voxels/diffusion.csv"
    file_list= [file for file in os.listdir(curated_dir) if file.endswith(".png")]
    angle_list= get_angle_list()

    for file in file_list[:2]:
        image_path= curated_dir + file
        angle= get_angle(image_path)
        print(f"Angle between points: {angle} degrees")

