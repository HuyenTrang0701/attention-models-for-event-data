'''
YOLOv8 expects label files with 5 columns (class, x_center, y_center, width, height). 
However, the label files generated have 6 columns (class, x_center, y_center, width, height, probability).
Solution:

    Modify the Label Generation: Ensure the label files contain only the required 5 columns.
    Verify and Clean Up Existing Labels: Remove the extra column from any existing label files.

'''
import glob
import os
import time

# Define the base path once
base_path = "/home/neuromorph/htn/dataset/DSEC/YOLOWorld"

def clean_label_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 6:
            # Remove the last part (probability)
            cleaned_lines.append(" ".join(parts[:5]))

    with open(file_path, 'w') as f:
        f.write("\n".join(cleaned_lines))

# Start the timer
start_time = time.time()

# Paths for train, val, and test labels
label_paths = {
    'train': os.path.join(base_path, "train/labels/*.txt"),
    'val': os.path.join(base_path, "val/labels/*.txt"),
    'test': os.path.join(base_path, "test/labels/*.txt")
}

# Clean up labels for each dataset split
for split, path in label_paths.items():
    print(f"Cleaning labels for {split} set...")
    label_files = glob.glob(path)
    for label_file in label_files:
        clean_label_file(label_file)
    print(f"Finished cleaning labels for {split} set.")

# Stop the timer
end_time = time.time()

# Calculate and print the total time taken
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")


