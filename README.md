# Eyglasses classification task

This repository contains code of eyglasses classifier trained on the subset of CelebA dataset. To run the code proceed to the following instructions.

### Deployment

The solution is wrapped into a Docker container. To build it run the command from the project root.

```
./build.sh
```

Then you need to run the container by replacing {YOUR_DATA_FOLDER} with the path to the test dataset and executing

```
docker run --name eyglasses_waytobehigh -v {YOUR_DATA_FOLDER}:/data/ -dit --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all eyglasses_waytobehigh
docker attach eyglasses_waytobehigh
```

Finally, run

```
python src/predict.py --data-path /data/
```

It will output the images classifed as "people with glasses".

```
predict.py [-h] [--data-path DATA_PATH] [--model-path MODEL_PATH]
                  [--threshold THRESHOLD]

This script loads a classification model and applies it to all files in the
data-path directory, outputing the path to every positively classified image

optional arguments:
  --data-path DATA_PATH
                        Path to the data to be classified
  --model-path MODEL_PATH
                        Path to the model
  --threshold THRESHOLD
                        Probability threshold of an image to be positive
```

### Training

```
usage: train.py [-h] [--data-path DATA_PATH] [--epochs EPOCHS]
                [--weight-decay WEIGHT_DECAY] [--lr LR]
                [--width-mult WIDTH_MULT] [--gamma GAMMA]
                [--batch-size BATCH_SIZE]
                
                  -h, --help            show this help message and exit
  --data-path DATA_PATH
                        Path to a dataset in the following format: {data-
                        path}/class1, {data-path}/class2, ...
  --epochs EPOCHS       Number of epochs
  --weight-decay WEIGHT_DECAY
                        L2 regularization coefficient
  --lr LR               Learning rate
  --width-mult WIDTH_MULT
                        Width multiplier of MobileNetV2
  --gamma GAMMA         Decay coefficient of exponential scheduler
  --batch-size BATCH_SIZE
                        Size of a batch

```
