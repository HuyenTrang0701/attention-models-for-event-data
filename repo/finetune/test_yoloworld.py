import time
from ultralytics import YOLOWorld

# Start the timer
start_time = time.time()

# Load the finetuned YOLO-World model
model = YOLOWorld('/home/neuromorph/htn/repo/dsec-det/htn/results_yoloworld_img256_7epoch_batch32/custom_yoloworld_img256_7epoch_batch32.pt')

# Path to the new test dataset
new_test_data = '/home/neuromorph/htn/dataset/DSEC/YOLOWorld/test/images'

# Run inference/predictions on the new test dataset
pred_start_time = time.time()
results = model.predict(
    source=new_test_data,  # Path to the test dataset directory or images
    save=True,             # Save the predictions
    save_txt=True,         # Save predictions as .txt files
    save_json=True,        # Save predictions as .json files
    project='/home/neuromorph/htn/dsec-det/htn/aa_trial',  # Directory to save the prediction results
    name='new_test_predictions',          # Name of the prediction session
    imgsz=512,             # Image size for prediction
    batch=32,              # Batch size for prediction
    conf=0.25              # Confidence threshold for predictions
)
print(f"Prediction time: {time.time() - pred_start_time:.2f} seconds")

# End of the script
print(f"Total time taken: {time.time() - start_time:.2f} seconds")
