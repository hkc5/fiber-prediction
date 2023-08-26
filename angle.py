import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def rename_files(curated_dir, first_no):
    file_list= [file for file in os.listdir(curated_dir) if file.endswith(".png")]

    first_no= 2001
    for file in file_list:
        os.rename(curated_dir+file, f"{curated_dir}{first_no}.png")
        first_no += 1

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
    plt.title(f"Click on two points in the image to calculate the angle. \n {image_path}")

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
    no_pics= len(file_list)
    try:
        angles= np.loadtxt(csv_file)
        print("Existing CSV file read!")
    except:
        angles= -1*np.ones(no_pics)
        np.savetxt(csv_file, angles)
        print("Angle CSV file created!")
    return angles

def update_angle_list(angle_list, angle, file):
    index= int(file.replace(".png", "")) - 2000 -1
    angle_list[index]= angle
    return angle_list

def write_angle_list(csv_dir, angle_list):
    np.savetxt(csv_dir, angle_list, fmt= "%.0f")

def check_labeled(file, angle_list):
    index= int(file.replace(".png", "")) - 2000 -1
    if angle_list[index] == -1:
        return False
    else:
        return True

if __name__ == "__main__":
    curated_dir= "./diffusion/diffusion_voxels/curated/"
    # rename_files(curated_dir, 2001)
    csv_dir= "./diffusion/diffusion_voxels/diffusion.csv"
    file_list= [file for file in os.listdir(curated_dir) if file.endswith(".png")]
    angle_list= get_angle_list(csv_dir)
    labeled_count= 0
    total_count= angle_list.shape[0]
    
    for file in file_list:
        if not check_labeled(file, angle_list):
            image_path= curated_dir + file
            angle= get_angle(image_path)       
            if angle == -1:
                print("Quitting labeling!")
                break
            else:
                angle_list= update_angle_list(angle_list, angle, file)
                write_angle_list(csv_dir, angle_list)
                labeled_count += 1
                print(f"{file} | Angle between points: {angle} degrees! | {angle_list[angle_list != -1].shape[0]}/{total_count} | CSV Updated!")            

    print(f"{labeled_count} new images were labeled!")
    