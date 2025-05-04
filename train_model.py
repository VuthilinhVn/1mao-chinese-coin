from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8s.pt")  # or yolov8m.pt / yolov8x.pt

# Train the model
model.train(
    data=r"E:/coin_count/training_data/data/data.yaml",
    epochs=70, # YOLO will look at the entire training dataset 70 times
    imgsz=640, # Resizes all images to 640x640 pixels before feeding them into the model.
    batch=16,  # Number of images processed at once in training.
    name="coin_count"
)


'''
Output:
best.pt - the best version of the model (based on validation)
results.png - graphs of training progress
confusion_matrix.png - how well the model predicts each class

this is what you are doing with this code Summary:
Hey YOLO, train for 70 rounds on my coin dataset (resized to 640x640), 
processing 16 images at a time, and save everything under a folder called euro_coin_detector.
'''