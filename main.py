import torch
import torchvision
from tqdm.auto import tqdm
import wandb

from models import FCModel

wandb.login()

configs = {"learning_rate": 3e-4, "batch_size": 64, "epochs": 10, "minimum_lr": 1e-4}

OUTPUT_SIZE = 10 # всего 10 цифр возможно => выведутся вероятности каждой из цифр

def train_one_epoch(model, dataloader, loss_calculator, optimizer, scheduler, device, epoch):
    model.train()
    average_loss = 0
    step = epoch * len(dataloader)
    for batch_idx, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        image, label = image.to(device), label.to(device)
        output = model(image) # calls forward(image), where image = batch of images
        loss = loss_calculator(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        wandb.log({
            "training loss": loss.item(),
            "learning rate": scheduler.get_last_lr()[0],
        }, step=step + batch_idx)

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
    for epoch in tqdm(range(configs["epochs"]), leave=False):
        train_avg_loss = train_one_epoch(model, train, loss_calculator, optimizer, scheduler, device, epoch)
        test_avg_loss, test_accuracy = evaluate(model, test, loss_calculator, device)

        wandb.log({
            "train average loss": train_avg_loss,
            "test average loss": test_avg_loss,
            "accuracy": test_accuracy,
        }, step=(epoch + 1) * len(train))

train_dataset = torchvision.datasets.MNIST(root="./mnist", train=True, download=False, transform=transformer)
test_dataset = torchvision.datasets.MNIST(root="./mnist", train=False, download=False, transform=transformer)

train = torch.utils.data.DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size=configs["batch_size"], shuffle=False)


image = train_dataset[0][0]
fc_model = FCModel(image.flatten().shape[0], configs["batch_size"], OUTPUT_SIZE)
device = "cuda" if torch.cuda.is_available() else "cpu"
fc_model.to(device)

loss_calculator = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(fc_model.parameters(), lr=configs["learning_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=configs["epochs"] * len(train), eta_min=configs["learning_rate"])

with wandb.init(project="mnist-digits-classification", name="third_try", config=configs) as run:  
    training(fc_model, train, test, loss_calculator, optimizer, scheduler, device)