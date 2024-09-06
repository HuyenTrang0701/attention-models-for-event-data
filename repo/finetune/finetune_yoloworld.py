import time
from ultralytics import YOLOWorld

# Start the timer
start_time = time.time()

# Load the pre-trained YOLO-World model
model = YOLOWorld('/home/neuromorph/htn/repo/dsec-det/htn/yolov8x-worldv2.pt')

# Dictionary to store time consumption
time_consumption = {}

# Train the model
train_start_time = time.time()
model.train(
    data='/home/neuromorph/htn/repo/dsec-det/htn/event_data_yoloworld.yaml',  # Path to your dataset yaml file
    epochs=7,
    imgsz=256,
    batch=32,
    name='yoloworld_finetune',  # Name of the training session
    save_period=1,  # Save model weights every 2 epochs
    project='/home/neuromorph/htn/repo/dsec-det/htn/aa_trial'  # Path to save the training results
)
print(f"Training time: {time.time() - train_start_time:.2f} seconds")

train_time = time.time() - train_start_time
print(f"Training time: {train_time:.2f} seconds")
time_consumption['Training time'] = train_time


# Validate the model on the validation set
val_start_time = time.time()
model.val(
    data='/home/neuromorph/htn/repo/dsec-det/htn/event_data_yoloworld.yaml',
    split='val',
    save_json=True
)
print(f"Validation time: {time.time() - val_start_time:.2f} seconds")

val_time = time.time() - val_start_time
print(f"Validation time: {val_time:.2f} seconds")
time_consumption['Validation time'] = val_time

# Test the model on the test set
test_start_time = time.time()
model.val(
    data='/home/neuromorph/htn/repo/dsec-det/htn/event_data_yoloworld.yaml',
    split='test',
    save_json=True
)
print(f"Testing time: {time.time() - test_start_time:.2f} seconds")

test_time = time.time() - test_start_time
print(f"Testing time: {test_time:.2f} seconds")
time_consumption['Testing time'] = test_time


# Save the final model
save_start_time = time.time()
model.save('/home/neuromorph/htn/repo/dsec-det/htn/aa_trial/custom_yoloworld_img512_40epoch_batch32.pt')
print(f"Model saving time: {time.time() - save_start_time:.2f} seconds")

save_time = time.time() - save_start_time
print(f"Model saving time: {save_time:.2f} seconds")
time_consumption['Model saving time'] = save_time

# Total time taken
total_time = time.time() - start_time
print(f"Total time taken: {total_time:.2f} seconds")
time_consumption['Total time taken'] = total_time

# Save time consumption to a text file
with open('/home/neuromorph/htn/repo/dsec-det/htn/aa_trial/time_consumption.txt', 'w') as f:
    for key, value in time_consumption.items():
        f.write(f"{key}: {value:.2f} seconds\n")