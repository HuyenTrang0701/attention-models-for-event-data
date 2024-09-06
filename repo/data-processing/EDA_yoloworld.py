import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Function to read labels
def read_labels(label_path):
    try:
        labels = pd.read_csv(label_path, delim_whitespace=True, header=None)
        labels.columns = ['class', 'x_center', 'y_center', 'width', 'height']
        return labels
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return None

# Function to validate labels
def validate_labels(labels):
    missing_values = labels.isnull().sum()
    negative_values = (labels < 0).sum()
    out_of_bounds = ((labels['x_center'] > 1) | (labels['y_center'] > 1) | 
                     (labels['width'] > 1) | (labels['height'] > 1)).sum()
    return missing_values, negative_values, out_of_bounds

# Function to clean labels
def clean_labels(labels):
    labels = labels[(labels >= 0).all(axis=1)]
    labels = labels[(labels['x_center'] <= 1) & (labels['y_center'] <= 1) & 
                    (labels['width'] <= 1) & (labels['height'] <= 1)]
    return labels

# Function to analyze and clean dataset
def analyze_and_clean_dataset(labels_dir, save_cleaned=False):
    all_labels = []
    overall_stats = {"missing_values": 0, "negative_values": 0, "out_of_bounds": 0}
    
    for label_file in labels_dir.glob("*.txt"):
        labels = read_labels(label_file)
        if labels is not None:
            missing_values, negative_values, out_of_bounds = validate_labels(labels)
            
            overall_stats["missing_values"] += missing_values.sum()
            overall_stats["negative_values"] += negative_values.sum()
            overall_stats["out_of_bounds"] += out_of_bounds

            cleaned_labels = clean_labels(labels)
            all_labels.append(cleaned_labels)
            
            if save_cleaned:
                cleaned_labels.to_csv(label_file, sep=' ', header=False, index=False)
    
    if all_labels:
        all_labels_df = pd.concat(all_labels, ignore_index=True)
    else:
        all_labels_df = pd.DataFrame(columns=['class', 'x_center', 'y_center', 'width', 'height'])
    
    print("\nOverall Dataset Statistics - Before Cleaning:")
    print(f"  Total Missing values: {overall_stats['missing_values']}")
    print(f"  Total Negative values: {overall_stats['negative_values']}")
    print(f"  Total Out of bounds values: {overall_stats['out_of_bounds']}")

    # Re-validate overall stats after cleaning
    overall_stats_after = validate_labels(all_labels_df)
    print("\nOverall Dataset Statistics - After Cleaning:")
    print(f"  Total Missing values: {overall_stats_after[0].sum()}")
    print(f"  Total Negative values: {overall_stats_after[1].sum()}")
    print(f"  Total Out of bounds values: {overall_stats_after[2]}")
    
    return all_labels_df

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
        6: 'motorcycle',
        7: 'train'
    }

    # Analyze and clean training dataset labels
    print("Analyzing and Cleaning Training Dataset Labels...")
    all_train_labels = analyze_and_clean_dataset(train_annotations_dir, save_cleaned=True)
    
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
