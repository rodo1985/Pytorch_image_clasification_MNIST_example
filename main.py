import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid
from torchvision.models import ResNet18_Weights


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet18 = torchvision.models.resnet18(weights = ResNet18_Weights)
        self.resnet18.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.resnet18(x)
        return x


def train(net, trainloader, valloader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        running_train_correct = 0
        running_train_total = 0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_train_total += labels.size(0)
            running_train_correct += (predicted == labels).sum().item()
        epoch_train_loss = running_train_loss / len(trainloader)
        epoch_train_accuracy = 100 * running_train_correct / running_train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(
            f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}%")

        running_val_loss = 0.0
        running_val_correct = 0
        running_val_total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(valloader), total=len(valloader)):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                running_val_total += labels.size(0)
                running_val_correct += (predicted == labels).sum().item()
            epoch_val_loss = running_val_loss / len(valloader)
            epoch_val_accuracy = 100 * running_val_correct / running_val_total
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_accuracy)
            print(
                f"Epoch {epoch + 1}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.2f}%")
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(net.state_dict(), 'best_model.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies


def test(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader, total=len(testloader)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def imshow(img, label):
    # convert tensor to numpy array and change dimension order
    img = img.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.1307])
    std = np.array([0.3081])
    img = std * img + mean  # reverse normalization
    img = np.clip(img, 0, 1)  # clip pixel values to valid range
    plt.imshow(img)
    plt.title(f'{label}')
    plt.axis('off')


def main():

    IsTrainMode = True
    batch_size = 512
    num_epochs = 10
    learning_rate = 0.001
    val_split = 0.2  # Percentage of the training set to be used for validation

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST dataset and split it into training, validation, and testing sets
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(val_split * num_train)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=val_sampler, num_workers=2)
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    if IsTrainMode:

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        train_losses, val_losses, train_accuracies, val_accuracies = train(
            net, trainloader, valloader, criterion, optimizer, num_epochs, device)

        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title("Loss over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title("Accuracy over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    else:
        net.load_state_dict(torch.load('best_model.pth'))
        net.eval()

        y_true = []
        y_pred = []
        misclassified_images = []
        misclassified_labels = []
        misclassified_preds = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true += labels.tolist()
                y_pred += predicted.tolist()
                for i in range(len(predicted)):
                    if predicted[i] != labels[i]:
                        misclassified_images.append(images[i])
                        misclassified_labels.append(labels[i])
                        misclassified_preds.append(predicted[i])

        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        fig = plt.figure(figsize=(4, 4))
        columns = 4
        rows = 4
        num_images = min(len(images), columns * rows)

        for i in range(1, num_images + 1):
            img = images[i-1]
            label = f'{labels[i-1]}({predicted[i-1]})'
            fig.add_subplot(rows, columns, i)
            imshow(make_grid(img), label)

        plt.show()

        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 4
        num_images = min(len(images), columns * rows)

        for i in range(1, num_images + 1):
            img = misclassified_images[i-1]
            label = f'{misclassified_labels[i-1]}({misclassified_preds[i-1]})'
            fig.add_subplot(rows, columns, i)
            imshow(make_grid(img), label)

        plt.show()


if __name__ == "__main__":
    main()
