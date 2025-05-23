from ultralytics import YOLO
import os
import time

MODEL_VARIANT = 'yolov8s.pt'

# Dataset and output
DATASET_YAML_PATH = '/home/ascend/Desktop/cv_project/dataset/dataset.yaml'
DATASET_ROOT = '/home/ascend/Desktop/cv_project/dataset' 
IMG_SUBDIR = 'combined_color' 
DEVICE = '0'
PROJECT_NAME = 'AD_PROJECT'
RUN_NAME = f'AD_RUN_{int(time.time())}' # Creates unique name each run


# Tunable params
#LEARNING_RATE = 0.001          # Overwritten when using optimizer = auto
OPTIMIZER = 'auto'              # Method used to adjust the models learning (often AdamW)
EPOCHS = 170                    # Number of times the model goes through the training set
IMGSZ = 1024                    # Resize of input images
BATCH_SIZE = 8                  # Number of images the model processes at the same time
PREDICT_CONF = 0.15             # Min confidence score for detection to be valid
PATIENCE = 50                   # Early stopping
AUG_SCALE = 0.3                 # Randomly zoom images in/out
AUG_TRANSLATE = 0.1             # Randomly shift images left/right/up/down
AUG_COPY_PASTE = 0.1            # Probability of copying objects and pasting them onto other images 
CLOSE_MOSAIC_EPOCHS = 30        # Number of final epochs to turn off mosaic augmentation 


# 1. Training
best_model_path = None

print(f"\n 1. Model Training"); 
print(f"Loading base model: {MODEL_VARIANT}") 

model = YOLO(MODEL_VARIANT); 
print(f"Output project: {PROJECT_NAME}, Run name: {RUN_NAME}"); 
print("Starting training...")

results = model.train(
    data=DATASET_YAML_PATH,
    #lr0=LEARNING_RATE,         # auto overwrites learning rate
    optimizer=OPTIMIZER,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    scale=AUG_SCALE,
    translate=AUG_TRANSLATE,   
    copy_paste=AUG_COPY_PASTE, 
    close_mosaic=CLOSE_MOSAIC_EPOCHS, 
    device=DEVICE,
    project=PROJECT_NAME,
    name=RUN_NAME,
    exist_ok=False,
    cache=True,                 # Trying to use RAM since I lack disk space
    val=True
    )


print("Training completed.")

best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights', 'best.pt') 


# 2. Validation
print(f"\n 2. Model Validation")
print(f"Loading trained model for validation: {best_model_path}")
model = YOLO(best_model_path)

val_run_name = f'{RUN_NAME}_validation'

print("Starting validation...")
val_results = model.val(
    data=DATASET_YAML_PATH,
    imgsz=IMGSZ,
    batch=BATCH_SIZE, 
    split='val',                # Validation dataset split
    device=DEVICE,
    project=PROJECT_NAME, 
    name=val_run_name,
    exist_ok=True
)
print("Validation completed.")

# Key Metrics
metrics = val_results.results_dict
print("\nValidation Metrics:")
print(f"  - Precision(B): {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"  - Recall(B):    {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
print(f"  - mAP50(B):     {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"  - mAP50-95(B):  {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")


# 3. Prediction
print(f"\n 3. Generating Predictions on Test Set")
test_img_path = os.path.join(DATASET_ROOT, IMG_SUBDIR, 'test', 'images')

print(f"Loading trained model for prediction: {best_model_path}")
model = YOLO(best_model_path)
predict_run_name = f'{RUN_NAME}_predictions'
predict_output_base = os.path.join(PROJECT_NAME, predict_run_name)
pred_save_dir = os.path.join(predict_output_base, 'labels')

print(f"Running prediction on images in: {test_img_path}")

predict_results = model.predict(
    source=test_img_path,
    imgsz=IMGSZ,
    conf=PREDICT_CONF,
    save_txt=True,
    save_conf=True,
    project=PROJECT_NAME,
    name=predict_run_name,
    device=DEVICE,
    exist_ok=True
    )

print(f"Prediction .txt files saved to: {pred_save_dir}")

print("\n Script Finished")