import argparse
import torch
import torchvision

from pathlib import Path
from PIL import Image
from train import SHARED_TRANSFORMS

def parse_args():
    parser = argparse.ArgumentParser(description="This script loads a classification model and applies it to all "
                                                 "files in the data-path directory, outputing the path to every "
                                                 "positively classified image")
    parser.add_argument('--data-path', type=str, default='/data/', help='Path to the data to be classified')
    parser.add_argument('--model-path', type=str, default='models/best.pth', help='Path to the model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold of an image to be positive')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = vars(parse_args())
    threshold = args['threshold']
    transforms = torchvision.transforms.Compose(SHARED_TRANSFORMS)
    model = torch.jit.load(args['model_path'])
    for image_path in Path(args['data_path']).glob('*'):
        image = Image.open(image_path)
        image = transforms(image).cuda()
        pred = torch.sigmoid(model(image.unsqueeze(0)))
        if pred[0, 0] > threshold:
            print(image_path)
