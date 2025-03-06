import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from fiber_predictor.utils.angle_utils import calculate_angle


def load_or_create_angle_data(csv_path, image_files):
    """Load or create the CSV file as a DataFrame with filename: angle pairs."""
    if not image_files:
        print("No image files found in the specified directory.")
        return pd.DataFrame(columns=["angle"])  # Return an empty DataFrame if no images

    if os.path.exists(csv_path):
        angle_data = pd.read_csv(csv_path, index_col="filename")
        print("Existing CSV file loaded!")
    else:
        # Initialize with -1 for all images to mark them as unlabeled
        angle_data = pd.DataFrame(
            index=pd.Index(image_files, name="filename"), columns=["angle"]
        )
        angle_data["angle"] = -1
        print("Angle CSV file created with initial -1 values for unlabeled images.")

    # Add any new images to angle_data with -1 (unlabeled)
    for image_file in image_files:
        if image_file not in angle_data.index:
            angle_data.loc[image_file] = -1

    # Remove entries for files that no longer exist in the directory
    existing_files_set = set(image_files)
    csv_files_set = set(angle_data.index)
    missing_files = csv_files_set - existing_files_set

    if missing_files:
        angle_data.drop(index=missing_files, inplace=True)
        print(
            f"Removed {len(missing_files)} entries from CSV for files no longer in directory."
        )
        # Save immediately after removal to confirm
        angle_data.to_csv(csv_path)
        print("CSV file updated and saved after removing missing entries.")

    return angle_data


def save_angle_data(angle_data, csv_path):
    """Save the angle data DataFrame to a CSV file."""
    angle_data.to_csv(csv_path)


def get_angle(image_path):
    """Display an image and allow the user to click two points to calculate the angle."""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, cmap="gray")
    plt.title(
        f"Click on two points in the image to calculate the angle.\n"
        f"Right click to mark as unwanted.\n {image_path}"
    )

    points = []
    angle = -1

    def onclick(event):
        nonlocal points, angle
        if event.button == 1:  # Left click to select points
            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
                ax.plot(event.xdata, event.ydata, "rs", markersize=5)
                plt.draw()

                if len(points) == 2:
                    angle = calculate_angle(points[0], points[1])
                    plt.close()
        elif event.button == 3:  # Right click to mark as unwanted
            angle = -2  # Set to -2 to indicate unwanted
            plt.close()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    return angle


def main(image_dir, csv_name):
    """Main function for labeling angles in images."""
    csv_path = os.path.join(image_dir, csv_name)
    image_files = sorted(
        [file for file in os.listdir(image_dir) if file.endswith(".png")]
    )

    # Check if there are any image files in the directory
    if not image_files:
        print(f"No .png files found in directory: {image_dir}")
        return

    angle_data = load_or_create_angle_data(csv_path, image_files)

    # Check for images that are still marked as unlabeled (-1)
    unlabeled_files = [
        file for file in image_files if angle_data.at[file, "angle"] == -1
    ]

    if not unlabeled_files:
        print("No new images to label.")
    else:
        print(f"Found {len(unlabeled_files)} images to label.")
        print("=" * 50, end="\n\n")

        labeled_count = 0
        for image_file in unlabeled_files:
            image_path = os.path.join(image_dir, image_file)
            angle = get_angle(image_path)

            if angle == -1:
                print("Quitting labeling!")
                break

            if angle == -2:
                angle_data.at[image_file, "angle"] = -2
                save_angle_data(angle_data, csv_path)
                print(f"{image_file} | Image marked as unwanted! | CSV Updated!")

            else:
                angle_data.at[image_file, "angle"] = angle
                save_angle_data(angle_data, csv_path)
                labeled_count += 1
                print(
                    f"{image_file} | Angle between points: {angle} degrees! | CSV Updated!"
                )

        print(f"{labeled_count} new images were labeled!")

        # Save the updated CSV after labeling all images
        save_angle_data(angle_data, csv_path)

    # Print summary information
    total_entries = len(angle_data)
    labeled_entries = (
        angle_data["angle"] >= 0
    ).sum()  # Count labeled entries (non -1, non -2)
    unwanted_entries = (angle_data["angle"] == -2).sum()
    unlabeled_entries = (angle_data["angle"] == -1).sum()

    print("=" * 50)
    print(f"Total entries in CSV: {total_entries}")
    print(f"Total labeled entries: {labeled_entries}")
    print(f"Total unwanted entries (-2): {unwanted_entries}")
    print(f"Total unlabeled entries (-1): {unlabeled_entries}")
    print("=" * 50, end="\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label angles in images by selecting points."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./images/diffusion/diffusion_voxels/",
        help="Directory containing images to label (default: './images/diffusion/diffusion_voxels/')",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="labels.csv",
        help="Name of the CSV file for storing angles (default: 'labels.csv')",
    )

    args = parser.parse_args()
    main(args.image_dir, args.csv_name)
