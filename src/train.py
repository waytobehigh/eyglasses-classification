import argparse
import torch
import torchvision
import numpy as np

from collections import defaultdict
from copy import deepcopy
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from utils import save_model, evaluate_model


TEST_SIZE = 2500
INPUT_SHAPE = (128, 128)
TRANSFORMS = [
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomResizedCrop(INPUT_SHAPE, scale=(0.7, 1.0)),
    torchvision.transforms.ColorJitter(0.3, 0.3, 0.3),
]
SHARED_TRANSFORMS = [
    torchvision.transforms.Resize(INPUT_SHAPE),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4998056, 0.42343128, 0.3848843],
                                     [0.30866688, 0.28865363, 0.28857686])
]


def parse_args():
    parser = argparse.ArgumentParser(description="This script trains the model and saves the best checkpoint "
                                                 "as well as loss-metric plot to the current directory")
    parser.add_argument('--data-path', type=str, default='/data/',
                        help='Path to a dataset in the following format: {data-path}/class1, {data-path}/class2, ...')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization coefficient')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--width-mult', type=float, default=0.5, help='Width multiplier of MobileNetV2')
    parser.add_argument('--gamma', type=float, default=1, help='Decay coefficient of exponential scheduler'),
    parser.add_argument('--batch-size', type=int, default=64, help='Size of a batch')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = vars(parse_args())
    print(args, flush=True)
    dataset = torchvision.datasets.DatasetFolder(
        args['data_path'],
        loader=Image.open,
        is_valid_file=lambda x: True,
        target_transform=lambda x: 0 if x else 1)

    datasets = {}
    torch.manual_seed(0)
    datasets['train'], datasets['test'], datasets['val'] = torch.utils.data.dataset.random_split(
        dataset, [len(dataset) - 2 * TEST_SIZE, TEST_SIZE, TEST_SIZE])
    torch.manual_seed(torch.initial_seed())

    for key, value in datasets.items():
        datasets[key].dataset = deepcopy(datasets[key].dataset)

    datasets['train'].dataset.transform = torchvision.transforms.Compose([
        *TRANSFORMS,
        *SHARED_TRANSFORMS
    ])
    datasets['test'].dataset.transform = torchvision.transforms.Compose([
        *SHARED_TRANSFORMS
    ])
    datasets['val'].dataset.transform = torchvision.transforms.Compose([
        *SHARED_TRANSFORMS
    ])

    loaders = {}

    for key, value in datasets.items():
        loaders[key] = torch.utils.data.DataLoader(value, batch_size=args['batch_size'], shuffle=True)

    experiment_name = f'WEIGHT_DECAY={args["weight_decay"]},LR={args["lr"]},' \
                      f'WIDTH_MULT={args["width_mult"]},GAMMA={args["gamma"]}'
    model = torchvision.models.MobileNetV2(num_classes=1, width_mult=args['width_mult']).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, args['gamma'])
    compute_loss = torch.nn.BCEWithLogitsLoss()

    metrics = defaultdict(list)
    for i in range(args['epochs']):
        print(f'[*] Epoch {i}...')
        losses_tmp = []
        model.train()
        for X_batch, y_batch in tqdm(loaders['train']):
            preds = model(X_batch.cuda()).cpu()
            loss = compute_loss(preds[:, 0], y_batch.to(torch.float))
            losses_tmp.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
        scheduler.step()
        metrics['train_loss'].append(np.mean(losses_tmp))
        for name, value in zip(['val_loss', 'val_rocauc'], evaluate_model(model, loaders['val'], compute_loss)):
            metrics[name].append(value)

        if metrics['val_rocauc'][-1] > max(metrics['val_rocauc'][:-1] + [0]):
            save_model(model, experiment_name + '.pth', INPUT_SHAPE)
        print(f"[*] Train loss: {metrics['train_loss'][-1]}, Val loss: {metrics['val_loss'][-1]}, "
              f"Val rocauc: {metrics['val_rocauc'][-1]}, Best rocauc: {max(metrics['val_rocauc'])}")
        plt.figure(figsize=(7, 14))
        plt.subplot(2, 1, 1)
        plt.plot(metrics['val_loss'], label='val_loss')
        plt.plot(metrics['train_loss'], label='train_loss')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(metrics['val_rocauc'])
        plt.hlines(max(metrics['val_rocauc']), *plt.xlim(), color='red',
                   linestyle='--', label=f"{max(metrics['val_rocauc']):.6}")
        plt.legend()
        plt.savefig(experiment_name + '.png')
        plt.close()

    print(f'[!] Training has completed.')
    model = torch.jit.load(experiment_name + '.pth')
    test_loss, test_rocauc = evaluate_model(model, loaders['test'], compute_loss)
    print(f"[*] Test loss: {test_loss}, Test rocauc: {test_rocauc}")
