import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR


#### Create logger
def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('SpReLU')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

def print_args(args, logger):
    logger.info('Arguments:')
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')


# Define the network
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(nn.MaxPool2d(2)(self.conv1(x)))
        x = torch.relu(nn.MaxPool2d(2)(self.conv2(x)))
        x = x.view(-1, 256)  # reshape before feeding to fc layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(log_interval, model, device, train_loader, optimizer, epoch, logger):
    model.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            accuracy = 100. * correct / ((batch_idx+1) * data.size(0))
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), accuracy))
    return 100. * correct / len(train_loader.dataset)

def test(model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)


    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy

def load_and_test(model_path, model, device, test_loader, logger):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info('Loaded Model Test set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='LeNet', choices = ["MLP", "LeNet"],
                        help='Which model to be used in the training')
    parser.add_argument('--path', type=str, default='./logging/',
                        help='For specifying the model saving and output logging directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Use which GPU for training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--evaluate', type=str, default='',
                        help='Model path to conduct evaluation without training')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")


    if not os.path.exists(args.path):
        os.makedirs(args.path)
        print(f"The folder '{args.path}' was created.")
    else:
        print(f"The folder '{args.path}' already exists.")

    # Logging setup
    logger = get_logger(os.path.join(args.path, "mnist_train.log"))

    print_args(args, logger)

    kwargs = {'num_workers': 1, 'pin_memory': True} 

    # Normalize with MNIST dataset mean and std
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, 
                                  transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = eval(args.model + "()")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)


    if args.evaluate:
        load_and_test(args.evaluate, model, device, test_loader, logger)
        exit()


    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args.log_interval, model, device, train_loader, optimizer, epoch, logger)
        test_acc = test(model, device, test_loader, logger)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            logger.info("Save model with best accuracy: {:.4f}%".format(best_acc))
            torch.save(model.state_dict(), os.path.join(args.path, "mnist_model.pt"))
        logger.info("Current best test accuracy: {:.4f}%".format(best_acc))
    logger.info('Final best Test Accuracy: {:.4f}%'.format(best_acc))

if __name__ == '__main__':
    
    main()