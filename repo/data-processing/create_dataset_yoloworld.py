# Set run by GPU
import torch
import shutil
from pathlib import Path
import numpy as np
import cv2
import sys
import argparse
import random
import os
import time

# Ensure argparse is imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dsec_det.dataset import DSECDet
from dsec_det.visualize import render_events_on_image
from dsec_det.label import COLORS, CLASSES

# Start timing
start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for the dataset
output_dir = Path("/home/neuromorph/htn/dataset/DSEC/YOLOWorld")

train_images_dir = output_dir / "train/images"
train_labels_dir = output_dir / "train/labels"
train_prompts_dir = output_dir / "train/prompts"

val_images_dir = output_dir / "val/images"
val_labels_dir = output_dir / "val/labels"
val_prompts_dir = output_dir / "val/prompts"

test_images_dir = output_dir / "test/images"
test_labels_dir = output_dir / "test/labels"
test_prompts_dir = output_dir / "test/prompts"

for directory in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, test_images_dir, test_labels_dir, train_prompts_dir, val_prompts_dir, test_prompts_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Function to save event images and annotations
def save_event_data_and_annotations(dataset, split, images_dir, labels_dir):
    for index in range(len(dataset)):
        try:
            output = dataset[index]
            event_image = np.zeros((480, 640, 3), np.uint8)
            event_image.fill(255)
            event_image = render_events_on_image(event_image, x=output['events']['x'], y=output['events']['y'], p=output['events']['p'])

            # Save the event image
            image_filename = f"{index:06d}.png"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), event_image)

            # Prepare annotation data
            tracks = output['tracks']
            annotations = []
            for i in range(len(tracks['x'])):
                cls_id = tracks['class_id'][i]
                x_center = (tracks['x'][i] + tracks['w'][i] / 2) / 640
                y_center = (tracks['y'][i] + tracks['h'][i] / 2) / 480
                width = tracks['w'][i] / 640
                height = tracks['h'][i] / 480
                probability = tracks['class_confidence'][i]
                annotations.append(f"{cls_id} {x_center} {y_center} {width} {height} {probability}")

            # Save the annotation
            label_filename = f"{index:06d}.txt"
            label_path = labels_dir / label_filename
            with open(label_path, 'w') as label_file:
                label_file.write("\n".join(annotations))

            print(f"Processed {index+1}/{len(dataset)} for {split}")

        except Exception as e:
            print(f"Error processing index {index}: {e}")

# Function to create textual prompts from labels
def create_text_prompts(labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                lines = f.readlines()

            # Initialize a dictionary to count occurrences of each class
            class_counts = {class_name: 0 for class_name in class_names}

            for line in lines:
                class_id = int(line.split()[0])
                class_name = class_names[class_id]
                class_counts[class_name] += 1

            # Create the prompt by mentioning the count for each class that appears
            prompt_parts = []
            for class_name, count in class_counts.items():
                if count > 0:
                    prompt_parts.append(f"{count} {class_name}{'s' if count > 1 else ''}")

            prompt = "This image contains " + ", ".join(prompt_parts) + "."

            # Save the prompt to a file
            with open(os.path.join(output_dir, label_file.replace('.txt', '.txt')), 'w') as out_f:
                out_f.write(prompt)


# Function to move a portion of training data to validation set
def move_to_validation(train_images, train_labels, val_images_dir, val_labels_dir, val_split=0.2):
    total_images = len(train_images)
    val_size = int(total_images * val_split)
    val_indices = random.sample(range(total_images), val_size)

    for idx in val_indices:
        image_src = train_images[idx]
        label_src = train_labels[idx]

        image_dst = val_images_dir / image_src.name
        label_dst = val_labels_dir / label_src.name

        shutil.move(image_src, image_dst)
        shutil.move(label_src, label_dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create YOLOv8 dataset from event data.")
    parser.add_argument("--dsec_merged", type=Path, required=True)
    args = parser.parse_args()

    assert args.dsec_merged.exists() and args.dsec_merged.is_dir()

    # Process training set
    train_dataset = DSECDet(args.dsec_merged, split="train", sync="back", debug=False)
    save_event_data_and_annotations(train_dataset, "train", train_images_dir, train_labels_dir)

    # Move 20% of training data to validation
    train_images = list(train_images_dir.glob("*.png"))
    train_labels = list(train_labels_dir.glob("*.txt"))
    move_to_validation(train_images, train_labels, val_images_dir, val_labels_dir)

    # Process test set
    test_dataset = DSECDet(args.dsec_merged, split="test", sync="back", debug=False)
    save_event_data_and_annotations(test_dataset, "test", test_images_dir, test_labels_dir)

    # Create text prompts for train, val, and test sets
    create_text_prompts(train_labels_dir, train_prompts_dir)
    create_text_prompts(val_labels_dir, val_prompts_dir)
    create_text_prompts(test_labels_dir, test_prompts_dir)

    print("Dataset creation complete.")

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total execution time: {elapsed_time:.2f} seconds")