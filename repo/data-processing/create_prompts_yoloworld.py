import os

# Define the class names
class_names = ['pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train']

# Path to your dataset
dataset_path = "/home/neuromorph/htn/dataset/DSEC/htn_trial/event_image"

# Create text prompts for train, val, and test sets
train_labels_dir = os.path.join(dataset_path, 'train/labels')
val_labels_dir = os.path.join(dataset_path, 'val/labels')
test_labels_dir = os.path.join(dataset_path, 'test/labels')

train_prompts_dir = os.path.join(dataset_path, 'train/prompts')
val_prompts_dir = os.path.join(dataset_path, 'val/prompts')
test_prompts_dir = os.path.join(dataset_path, 'test/prompts')

# Function to create textual prompts from labels
def create_text_prompts(labels_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                lines = f.readlines()

            prompt = "This image contains "
            for line in lines:
                class_id = int(line.split()[0])
                class_name = class_names[class_id]
                prompt += f"a {class_name}, "

            # Remove the last comma and add a period
            prompt = prompt.rstrip(", ") + "."
            with open(os.path.join(output_dir, label_file.replace('.txt', '.txt')), 'w') as out_f:
                out_f.write(prompt)

create_text_prompts(train_labels_dir, train_prompts_dir)
create_text_prompts(val_labels_dir, val_prompts_dir)
create_text_prompts(test_labels_dir, test_prompts_dir)

