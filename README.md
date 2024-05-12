# Brain Tumor Detection

In this project, we use yolov3, yolov7, and yolov8 to test how augmenting the dataset using copy-and-paste during the training process will enhance the model performance.

## Main Commands
Please set your dataset path and modify the data.yaml file before running each model.

### YOLOv3
There are two ways to run this code:
- Method 1: Be sure to have the yolov3_terminal package in the google drive and keep the folder as the same structure. You can set the parameters and mode you want to use in the "Training the YOLOv3 model" and "Testing the YOLOv3 model" sections in the YOLOv3_BTD.ipynb file. Then, make sure to run every cells from the start.
- Method 2: Download the yolov3_terminal package. Open the the terminal and set the directory to source_code folder. Then, run the following codes to test and train.
  ```
  python3 run.py --mode train --epoch 5
  ```
  ```
  python3 run.py --mode test
  ```

### YOLOv7
To train the model:
```
! python ./yolov7/train.py --weights yolov7.pt --cfg ./yolov7/cfg/training/yolov7.yaml --data ./yolov7/data/data.yaml \
    --batch-size 16 --epochs 30 --img-size 416 416 --adam --single-cls --hyp ./yolov7/data/hyp.scratch.custom.yaml --name run
```
To test the model:
```
!python ./yolov7/test.py --weights /content/runs/train/run/weights/best.pt --data ./yolov7/data/data.yaml --task test --name yolo_test
```

### YOLOv8
To train and validate the model:
```
!yolo task=detect mode=train model=yolov8m.pt data={Your Dataset Path}/data.yaml epochs=30 imgsz=416 optimizer='Adam' single_cls name=run
```

### Data augmentation

## Major changes in the adapted code

### YOLOv3
- Incorporated code to compute test loss and accuracy metrics
- Incorporated code to produce loss graph for training and testing
- Converted the original code package into an IPython Notebook (.ipynb)

### YOLOv7
- Incorporated code to test the model besides training
- Applied Adam optimizer
- Adjusted hyperparameters for fine-tuning
  - image size (image dimention) = 416
  - batch size = 16
  - single_cls = TRUE

### YOLOv8
- This code is original, written after reading the tutorials and the official guideline.

### Data augmentation

## Contribution
### Zhichen Zhou
- YOLOv3 folder
  - yolov3_BTD.ipynb: code to run YOLOv3 model for 5 epochs
  - yolov3_terminal.zip: a package to be downloaded or uploaded to Google Drive in the correct directory structure
  - yolov3_results.zip: a package of the evaluation results and graphs for the control and augmented datasets respectively
- YOLOv7 folder
  - yolov7_BTD.ipynb: code to run YOLOv7 model for 30 epochs
  - yolov7_results.zip: a package of the evaluation results and graphs for the control and augmented datasets respectively
- YOLOv8 folder
  - yolov8_BTD.ipynb: code to run YOLOv8 model for 30 epochs
  - yolov8_results.zip: a package of the evaluation results and graphs for the control and augmented datasets respectively
- Experimental trials folder
  - Try_DETR.ipynb: tried to apply DETR to our project, but failed due to the requirement of a COCO dataset

### Kexuan Li

## Dataset
### Baseline dataset
https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets

### Healthy brain dataset
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

## Citation
### YOLOv3
Adapted from: https://github.com/mr-ravin/Brain-Tumor-Detection-MRI-using-YOLO-v3-Pytorch

### YOLOv7
Adapted from: https://gist.github.com/melek227/80db855e32a7908fa8ba15957d146b28

### YOLOv8
Learning resourses: 
- https://learnopencv.com/train-yolov8-on-custom-dataset/
- https://docs.ultralytics.com/usage/cfg/

### Data augmentation (copy-and-paste)



