# Brain Tumor Detection

In this project, we use yolov3, yolov7, and yolov8 to test how augmenting the dataset using copy-and-paste during the training process will enhance the model performance.

## Main Commands
Please remember to set your dataset path before running each model.

### YOLOv3
There are two ways to run this code:
- Method 1: Set the parameters and mode you want to use in the "Training the YOLOv3 model" and "Testing the YOLOv3 model" in the YOLOv3_BTD.ipynb file. Then, make sure to run every cells from the start.
- Method 2: download the yolov3_terminal package, in the terminal, run the following codes to test and train.
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


