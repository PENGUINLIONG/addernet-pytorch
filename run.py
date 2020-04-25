import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import onnx
from ladder import Ladder2D

dev = torch.device('cuda:0')

train_data = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.2,))]))
test_data = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.2,))]))
train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=256, num_workers=2)



class LadderLeNet(nn.Module):
    def __init__(self):
        super(LadderLeNet, self).__init__()
        self.conv1 = Ladder2D(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = Ladder2D(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(self.bn1(self.conv1(x)), (2,2))
        x = F.max_pool2d(self.bn2(self.conv2(x)), (2,2))
        x = x.reshape(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(-1, self.num_flat_features(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LadderLeNet()
print (net)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
NEPOCHE = 50
NITER = NEPOCHE * len(train_loader)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NITER)

cur_batch_win = None

def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(dev)
        labels = labels.to(dev)
        optimizer.zero_grad()

        output = net.to(dev)(images)

        loss = loss_fn(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()
        scheduler.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(dev)
        labels = labels.to(dev)
        output = net.to(dev)(images)
        avg_loss += loss_fn(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(test_data)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(test_data)))


def train_and_test(epoch):
    train(epoch)
    test()

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    torch.save(net.to(torch.device('cpu')).state_dict(), 'lenet.params')


def main():
    for e in range(1, NEPOCHE):
        train_and_test(e)


if __name__ == '__main__':
    main()
