# -*- coding: utf-8 -*-
"""
Clean baseline classifier training script.

Trains a ResNet-50 clean model for a given dataset and saves a checkpoint
compatible with all UAP attack scripts (checkpoint key: 'netC').

Usage (run from repo root):
  PYTHONPATH=. python models/train_baseline.py --dataset CXRAY
  PYTHONPATH=. python models/train_baseline.py --dataset ISIC2019
  PYTHONPATH=. python models/train_baseline.py --dataset ISIC2019 --num_classes 3 \
      --csv_train data/splits/isic2019/ISIC2019_train_012.csv \
      --csv_test  data/splits/isic2019/ISIC2019_test_012.csv
  PYTHONPATH=. python models/train_baseline.py --dataset OCT

Checkpoint is saved to: checkpoints/<dataset_lower>/<dataset_lower>_clean_baseline.pth.tar
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader

from shared.dataloader import ISICDataset, DatasetSeprateByClass


# Default paths per dataset — override any of these with CLI args
DATASET_DEFAULTS = {
    'CXRAY': {
        'num_classes': 2,
        'train_data': 'data/cxray/train',
        'test_data':  'data/cxray/test',
        'csv_train':  'data/splits/cxray/CXRAY-train.csv',
        'csv_test':   'data/splits/cxray/CXRAY-test.csv',
    },
    'ISIC2019': {
        'num_classes': 8,
        'train_data': 'ISIC2019',
        'test_data':  'ISIC2019',
        'csv_train':  'data/splits/isic2019/ISIC2019_train.csv',
        'csv_test':   'data/splits/isic2019/ISIC2019_test.csv',
    },
    'OCT': {
        'num_classes': 4,
        'train_data': 'data/oct2017/train',
        'test_data':  'data/oct2017/test',
        'csv_train':  'data/splits/oct2017/OCT2017-train.csv',
        'csv_test':   'data/splits/oct2017/OCT2017-test.csv',
    },
}


def build_dataloader(dataset, data_path, csv_file, split, transform, batch_size, num_classes, num_workers):
    """
    Build a DataLoader using the same dataset classes as the UAP scripts,
    ensuring label indices are consistent between training and evaluation.
    """
    if dataset == 'ISIC2019':
        ds = ISICDataset(
            datapath=data_path,
            csv_file=csv_file,
            data_type=split,
            transform=transform,
            one_hot_encode=False,
            num_classes=num_classes,
        )
    else:
        ds = DatasetSeprateByClass(
            root_dir=data_path,
            csv_file=csv_file,
            data_type=split,
            transform=transform,
            one_hot_encode=False,
            num_classes=num_classes,
        )
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train_data'), num_workers=num_workers)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train clean baseline classifier')
    parser.add_argument('--dataset',     type=str, required=True, choices=['CXRAY', 'ISIC2019', 'OCT'],
                        help='Dataset to train on')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes — overrides dataset default. '
                             'Use 3 for ISIC2019 3-class variant.')
    parser.add_argument('--train_data',  type=str, default=None,
                        help='Root directory of training images (overrides default)')
    parser.add_argument('--test_data',   type=str, default=None,
                        help='Root directory of test images (overrides default)')
    parser.add_argument('--csv_train',   type=str, default=None,
                        help='Training split CSV path (overrides default)')
    parser.add_argument('--csv_test',    type=str, default=None,
                        help='Test split CSV path (overrides default)')
    parser.add_argument('--n_epochs',    type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--output_dir',  type=str,   default='checkpoints',
                        help='Root directory for saved checkpoints')
    parser.add_argument('--device',      type=str,   default='cuda')
    parser.add_argument('--num_workers', type=int,   default=2)
    opt = parser.parse_args()

    cfg = DATASET_DEFAULTS[opt.dataset]
    num_classes = opt.num_classes or cfg['num_classes']
    train_data  = opt.train_data  or cfg['train_data']
    test_data   = opt.test_data   or cfg['test_data']
    csv_train   = opt.csv_train   or cfg['csv_train']
    csv_test    = opt.csv_test    or cfg['csv_test']

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_loader = build_dataloader(opt.dataset, train_data, csv_train, 'train_data',
                                    train_transform, opt.batch_size, num_classes, opt.num_workers)
    test_loader  = build_dataloader(opt.dataset, test_data,  csv_test,  'test_data',
                                    test_transform,  opt.batch_size, num_classes, opt.num_workers)

    print(f"Dataset : {opt.dataset} | Classes: {num_classes}")
    print(f"Train   : {len(train_loader.dataset)} images | Test: {len(test_loader.dataset)} images")

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    print(f"Device  : {device}")

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    # Decay LR by 10x at epochs 40, 70, 90
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70, 90], gamma=0.1)

    ckpt_dir  = os.path.join(opt.output_dir, opt.dataset.lower())
    ckpt_path = os.path.join(ckpt_dir, f'{opt.dataset.lower()}_clean_baseline.pth.tar')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_acc = 0.0
    start = time.time()

    for epoch in range(opt.n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss,  test_acc  = eval_epoch(model,  test_loader,  criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1:3d}/{opt.n_epochs}]  "
              f"Train — Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%  |  "
              f"Test  — Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'netC':           model.state_dict(),
                'optimizerC':     optimizer.state_dict(),
                'epoch':          epoch,
                'best_clean_acc': best_acc,
                'dataset':        opt.dataset,
                'num_classes':    num_classes,
            }, ckpt_path)
            print(f"  → Checkpoint saved  (best acc: {best_acc:.2f}%)")

    elapsed = time.time() - start
    print(f"\nTraining complete — Best test acc: {best_acc:.2f}%  |  Time: {elapsed:.0f}s")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == '__main__':
    main()
