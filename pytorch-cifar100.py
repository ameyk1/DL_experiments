import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
from nn_architectures import alexnet
import argparse

def parse_args():
    # Parse input arguments
    usage_command = 'python3 mskaggle.py '
    parser = argparse.ArgumentParser(prog='mskaggle', usage=usage_command)
    
    parser.add_argument('--num_epochs', dest='num_epochs',
                        help='Number of Epochs for training',
                        default=200, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='How many images in one batch',
                        default=256, type=int)
    parser.add_argument('--in_channels', dest='in_channels',
                        help='Sie of input images',
                        default=3, type=int)
    parser.add_argument('--num_classes', dest='n_classes',
                        help='Number of Classes/Labels in dataset',
                        default=2, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='Number of Classes/Labels in dataset',
                        default=12, type=int)
    args = parser.parse_args()
    return args


def main ():
    args = parse_args()
    print('Called with args:')
    print(args)
    ### best_loss = 0.0
    NUM_EPOCHS = args.num_epochs
    BEST_MODEL_PATH = 'best_mskaggle'
    n_classes = args.n_classes
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    best_accuracy = 0.0
    
    # Select Device to train on
    device_id_t = torch.cuda.current_device()
    device_name_t = torch.cuda.get_device_name(device_id_t)
    print(device_name_t) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # CIFAR Data Download and data augmentation and transform
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    val_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR100(root='./CIFA100data',train=True, transform=train_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataset = datasets.CIFAR100(root='./CIFAR100data',train=False, transform=val_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Get Model
    model = alexnet.alexnet(in_channels=args.in_channels, num_classes=n_classes)
    print(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-8 )
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 300, 750], gamma=0.5)
    test_loss = 0.0
    test_correct = 0.0
    total = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        scheduler.step(epoch)
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch%10 == 0:
            model.eval()
            test_error_count = 0.0
            for images, labels in iter(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()
                prediction = torch.max(outputs, 1)
                total += labels.size(0)
                test_correct += float(torch.sum(labels == outputs.argmax(1)))

            test_accuracy = (test_correct/total)*100
            print('Epoch = %d: Test Accuracy = %f %%' % (epoch, test_accuracy))
        
            if test_accuracy > best_accuracy:
                torch.save(model.state_dict(), "{}.pth".format(BEST_MODEL_PATH))
                best_accuracy = test_accuracy


if __name__ == "__main__":
    main()