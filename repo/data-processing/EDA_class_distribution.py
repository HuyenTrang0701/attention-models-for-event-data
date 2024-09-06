import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Function to parse annotations
def parse_annotations(annotations_dir):
    class_counts = {}
    for file_name in os.listdir(annotations_dir):
        if file_name.endswith('.txt'):
            with open(os.path.join(annotations_dir, file_name), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1
    return class_counts

# Function to plot class distribution
def plot_class_distribution(train_counts, val_counts, test_counts, class_names):
    classes = list(class_names.values())
    train_values = [train_counts.get(i, 0) for i in range(len(class_names))]
    val_values = [val_counts.get(i, 0) for i in range(len(class_names))]
    test_values = [test_counts.get(i, 0) for i in range(len(class_names))]
    
    df = pd.DataFrame({
        'Class': classes,
        'Train': train_values,
        'Validation': val_values,
        'Test': test_values
    })
    
    ax = df.set_index('Class').plot(kind='bar', figsize=(10, 6))
    plt.title('Class Distribution in Datasets')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)

    # Add text annotations on the bars
    for container in ax.containers:
        ax.bar_label(container, label_type='edge')
    
    plt.show()

if __name__ == "__main__":
    start_time = time.time()  # Start time measurement

    # Paths to annotation directories
    base_dir = Path('/home/neuromorph/htn/dataset/DSEC/YOLOWorld')
    train_annotations_dir = base_dir / 'train/labels'
    val_annotations_dir = base_dir / 'val/labels'
    test_annotations_dir = base_dir / 'test/labels'

    # Class names mapping (adjust as per classes)
    class_names = {
        0: 'pedestrian',
        1: 'rider',
        2: 'car',
        3: 'bus',
        4: 'truck',
        5: 'bicycle',
        6: 'motorcycle'
    }
    
    # Parse annotations to get class counts
    train_counts = parse_annotations(train_annotations_dir)
    val_counts = parse_annotations(val_annotations_dir)
    test_counts = parse_annotations(test_annotations_dir)

    # Print the count of each class after cleaning for each dataset
    print("\nClass counts after cleaning:")
    print(f"Training set: {train_counts}")
    print(f"Validation set: {val_counts}")
    print(f"Test set: {test_counts}")

    # Plot class distribution
    plot_class_distribution(train_counts, val_counts, test_counts, class_names)

    end_time = time.time()  # End time measurement
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
