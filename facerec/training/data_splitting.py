import os
import shutil
from sklearn.model_selection import train_test_split

# Define the path to the preprocessed data directory
preprocessed_data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "facerec", "preprocessed_data")

# Define the path to the output directory for the train/test split
split_data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "facerec", "split_data")

# Define the train/test split ratio
split_ratio = 0.2

# Loop over all subdirectories in the preprocessed data directory
for subdir in os.listdir(preprocessed_data_dir):
    subdir_path = os.path.join(preprocessed_data_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # Split the images in the current subdirectory into train and test sets
    images = os.listdir(subdir_path)
    train_images, test_images = train_test_split(images, test_size=split_ratio)

    # Create subdirectories for train and test sets in the split data directory
    train_subdir = os.path.join(split_data_dir, "train", subdir)
    if not os.path.exists(train_subdir):
        os.makedirs(train_subdir)
    test_subdir = os.path.join(split_data_dir, "test", subdir)
    if not os.path.exists(test_subdir):
        os.makedirs(test_subdir)

    # Copy the train images to the train subdirectory
    for img in train_images:
        src = os.path.join(subdir_path, img)
        dst = os.path.join(train_subdir, img)
        shutil.copyfile(src, dst)

    # Copy the test images to the test subdirectory
    for img in test_images:
        src = os.path.join(subdir_path, img)
        dst = os.path.join(test_subdir, img)
        shutil.copyfile(src, dst)
