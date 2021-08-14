import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from densenet import DenseNet
from tqdm import tqdm
import matplotlib.pyplot as plt


def DenseNetBC_100_12():
    return DenseNet(growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10)

def save_graph(file_path="./results"):
    train_accuracy_list = []
    test_accuracy_list = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            pass

def unstructured_prune(path='./cifar_net.pth', prune_rate=0, accuracy_test=True, results_path='./prune_results.txt'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DenseNetBC_100_12()
    model = model.to(device)
    model.load_state_dict(torch.load(path))

    correct_train = 0
    total_train = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            alpha = float(prune_rate / 100)  # 삭제할 파라미터 비율
            max_idx = int(len(list(param.view(-1))) * alpha)
            mask = torch.argsort(torch.abs(param.view(-1)))
            mask = mask.ge(max_idx)
            new_param = torch.reshape(param.view(-1) * mask, tuple(param.shape))

            param.copy_(new_param)

    if accuracy_test:
        with torch.no_grad():
            for data in tqdm(train_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).to(device)

                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            for data in tqdm(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).to(device)

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

    print(f"\n\nPrune rate : {prune_rate}")
    print('Accuracy of the network on the 50000 train images : %.2f %%' % (100 * correct_train / total_train))
    print('Accuracy of the network on the 10000 test images : %.2f %%' % (100 * correct_test / total_test))

    with open(results_path, 'a', encoding='utf-8') as f:
        f.write(f"{(prune_rate)}% pruned Train Accuracy : {(100 * correct_train / total_train):.2f}\n")
        f.write(f"{(prune_rate)}% pruned test Accuracy : {(100 * correct_test / total_test):.2f}\n\n")






parser = argparse.ArgumentParser(description='unstructured Pruning')
parser.add_argument('--prune_rate', type=str, default='0-1-2-3-5-15-25-50-75-95', help='rate of weights to be pruned')
parser.add_argument('--results_path', type=str, default='./results', help='path results to be saved')
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    rate_list = args.prune_rate.split('-')

    for rate in rate_list:
        rate = int(rate)
        unstructured_prune('./cifar_net.pth', rate, results_path=args.results_path)

    save_graph()
