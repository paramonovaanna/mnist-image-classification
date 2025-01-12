import torch
import torchvision
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from models import FCModel

n_epochs = 10
batch_size = 128
lr = 3e-4

OUTPUT_SIZE = 10 # всего 10 цифр возможно => выведутся вероятности каждой из цифр
    

def train_one_epoch(model, dataloader, loss_calculator, optimizer, scheduler, device):
    model.train()
    average_loss = 0
    for batch_idx, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        image, label = image.to(device), label.to(device)
        output = model(image) # calls forward(image), where image = batch of images
        loss = loss_calculator(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        average_loss += loss.item()
    average_loss /= (batch_idx + 1)
    return average_loss


def evaluate(model, dataloader, loss_calculator, device):
    model.eval()
    average_loss, accuracy = 0, 0
    total_elements = 0
    output = 0
    for batch_idx, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        image, label = image.to(device), label.to(device)
        output = model(image)
        loss = loss_calculator(output, label)

        accuracy += (output.argmax(-1) == label).sum().item()
        total_elements += output.shape[0] # output в классификации обычно имеет форму <число элементов, число классов>

        average_loss += loss.item()
    average_loss /= (batch_idx + 1)
    accuracy = accuracy / total_elements * 100
    return average_loss, accuracy

transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def training(model, train, test, loss_calculator, optimizer, scheduler, device):
    train_avg_losses = []
    test_avg_losses, test_accuracy_list = [], []

    for _ in range(n_epochs):
        train_avg_loss = train_one_epoch(model, train, loss_calculator, optimizer, scheduler, device)
        test_avg_loss, test_accuracy = evaluate(model, test, loss_calculator, device)

        train_avg_losses.append(train_avg_loss)
        test_avg_losses.append(test_avg_loss)
        test_accuracy_list.append(test_accuracy)


    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    axes[0].plot(train_avg_losses, color="blue", label="train")
    axes[0].plot(test_avg_losses, color="red", label="test")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Min train loss: {round(min(train_avg_losses), 5)}, min val loss: {round(min(test_avg_losses), 5)}")

    axes[1].plot(test_accuracy_list, color="red", label="val")
    axes[1].set_xlabel("Epoch number")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"Best Accuracy (%): {max(test_accuracy_list)}")

    plt.legend()

    plt.tight_layout()
    plt.show()

train_dataset = torchvision.datasets.MNIST(root="./mnist", train=True, download=False, transform=transformer)
test_dataset = torchvision.datasets.MNIST(root="./mnist", train=False, download=False, transform=transformer)

train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


image = train_dataset[0][0]
fc_model = FCModel(image.flatten().shape[0], batch_size, OUTPUT_SIZE)
device = "cuda" if torch.cuda.is_available() else "cpu"
fc_model.to(device)

loss_calculator = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(fc_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs * len(train))

training(fc_model, train, test, loss_calculator, optimizer, scheduler, device)