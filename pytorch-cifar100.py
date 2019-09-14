import torch
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from apex.parallel import LARC

import numpy as np
import matplotlib.pyplot as plt
from nn_architectures import alexnet, overfeat, mobilenetv2
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
                        default=10, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='Number of Classes/Labels in dataset',
                        default=12, type=int)
    args = parser.parse_args()
    return args

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(32, padding=4), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_dataset = datasets.CIFAR10(root='./CIFA10data',train=True, transform=train_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_dataset = datasets.CIFAR10(root='./CIFAR10data',train=False, transform=val_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    num_channels_stage = [16,24,32,64,96,160,160]
    num_channels_1x1 = [32,320,1280]
    num_stages=2
    t=6
    repeat=2
    class_num=10
    # Get Model
    #model = alexnet.alexnet(in_channels=args.in_channels, num_classes=n_classes)
    #model = overfeat.overfeat(in_channels=args.in_channels, num_classes=n_classes)
    model = mobilenetv2.mobilenetv2(class_num=class_num, num_stages=num_stages, num_channels_stage=num_channels_stage, num_channels_1x1=num_channels_1x1, repeat=repeat, t=t)
    print(model)
    model = model.to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4 )
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4 )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300,700,1000],gamma=0.1)
    #optimizer = LARC.LARC(optimizer, trust_coefficient=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25)
    test_correct = 0.0
    total = 0.0
    train_loss_values = []
    test_loss_values = []
    accuracy_val =[]

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_running_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            train_running_loss += loss.item()
            optimizer.step()
        scheduler.step(epoch)
        train_loss_values.append(train_running_loss / len(train_dataset))
            
        if epoch%50 == 0:
            model.eval()
            test_running_loss = 0.0
            top1 = AverageMeter()
            top5 = AverageMeter()      
            for images, labels in iter(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                test_running_loss += loss.item()
                #prediction = torch.max(outputs, 1)
                #total += labels.size(0)
                #test_correct += float(torch.sum(labels == outputs.argmax(1)))
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))   
                top1.update(to_python_float(prec1), images.size(0))
                top5.update(to_python_float(prec5), images.size(0))

            test_loss_values.append(test_running_loss / len(test_dataset))
            
            print(' *Epoch{epoch:d}\t * Prec@1 {top1.avg:.3f}\t * Prec@5 {top5.avg:.3f}\t'.format(epoch=epoch, top1=top1, top5=top5))
            #test_accuracy = prec1_running
            #print('Epoch = %d: Test Accuracy = %f %%' % (epoch, test_accuracy))
            #accuracy_val.append(test_accuracy)
    
            # if test_accuracy > best_accuracy:
            #    torch.save(model.state_dict(), "{}.pth".format(BEST_MODEL_PATH))
            #    best_accuracy = test_accuracy

    with open('train_loss_values.txt', 'w') as f:
        for item in train_loss_values:
            f.write("%s\n" % item)
    with open('test_loss_values.txt', 'w') as f:
        for item in test_loss_values:
            f.write("%s\n" % item)
    with open('accuracy_val.txt', 'w') as f:
        for item in accuracy_val:
            f.write("%s\n" % item)
              
    # plt.plot(train_loss_values)
    # plt.plot(test_loss_values)
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper left')
    # plt.show()
if __name__ == "__main__":
    main()
