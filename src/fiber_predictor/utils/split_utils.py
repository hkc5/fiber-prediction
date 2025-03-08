import os
import shutil
import pandas as pd
import tempfile
from sklearn.model_selection import train_test_split

def split_dataset_temp(
    orig_folder,
    labels_csv_name="labels.csv",
    test_size=0.2,
    random_seed=42
):
    """
    Creates a temporary directory containing separate 'train' and 'test' folders
    for the folder datasets. Randomly splits data into an 80/20 ratio by default.
    
    Returns:
    - temp_dir  : TemporaryDirectory object (stay in scope as long as needed)
    - train_csv : The path to the new 'labels_train.csv'
    - test_csv  : The path to the new 'labels_test.csv'
    """
    # Read the original labels file
    csv_path = os.path.join(orig_folder, labels_csv_name)
    df = pd.read_csv(csv_path)  # Must contain at least ['filename', 'angle'] columns

    # Train/Test split (80/20 default)
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=True
    )

    # Create a temporary directory
    temp = tempfile.TemporaryDirectory()  # Will auto-delete when no longer referenced

    # Inside that temp directory, create 'train' and 'test' subfolders
    train_folder = os.path.join(temp.name, "train")
    test_folder  = os.path.join(temp.name, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Copy images to train/test
    for idx, row in df_train.iterrows():
        src_img = os.path.join(orig_folder, row["filename"])
        dst_img = os.path.join(train_folder, row["filename"])
        shutil.copy(src_img, dst_img)

    for idx, row in df_test.iterrows():
        src_img = os.path.join(orig_folder, row["filename"])
        dst_img = os.path.join(test_folder, row["filename"])
        shutil.copy(src_img, dst_img)

    # Create the two new label CSVs
    train_csv = os.path.join(train_folder, "labels_train.csv")
    test_csv  = os.path.join(test_folder, "labels_test.csv")
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"Dataset split into temp folder: {temp.name}")
    print(f" Train folder: {train_folder}  ({len(df_train)} images)")
    print(f" Test folder : {test_folder}   ({len(df_test)} images)\n")
    
    return temp, (train_folder, train_csv), (test_folder, test_csv)

if __name__ == "__main__":
    bio_orig_folder = "/Users/hakancan/Documents/Biofluids Lab/fiber-orientation/images/bio"
    temp_dir, train_csv, test_csv = split_dataset_temp(bio_orig_folder)
    # Print the directories, csv paths, sizes, etc.
    print(f"Temp dir: {temp_dir.name}")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV : {test_csv}")
    print(f"Train size: {len(pd.read_csv(train_csv))}")
    print(f"Test size : {len(pd.read_csv(test_csv))}")

    # When done, the temp_dir will be auto-deleted
    temp_dir.cleanup()
    print("Temp dir deleted.")
