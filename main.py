import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from capsnet import CapsNet
from capsnet.config import MNIST


def train(capsule_net, optimizer, data_loader, epoch, device=torch.device("cpu")):
    capsule_net.to(device)
    capsule_net.train()

    n_batch = np.ceil(len(data_loader.dataset) / data_loader.batch_size)
    total_loss = 0
    for i, (batch_image, batch_label) in enumerate(tqdm(data_loader)):
        batch_image, batch_label = batch_image.to(device), batch_label.to(device)

        optimizer.zero_grad()
        batch_obj_vectors, batch_reconstruction, batch_masks = capsule_net(batch_image, batch_label)
        loss = capsule_net.loss(batch_obj_vectors, batch_label, batch_image, batch_reconstruction)
        loss.backward()
        optimizer.step()

        accuracy = np.mean(
            np.argmax(batch_masks.detach().cpu().numpy(), 1) == batch_label.cpu().numpy()
        )
        total_loss += loss.item()
        avg_loss = loss.item() / float(data_loader.batch_size)

        if i % 100 == 0:
            tqdm.write(
                f"Epoch: [{epoch}], Batch: [{i + 1}/{n_batch}], train accuracy: {accuracy:.6f}, "
                f"loss: {avg_loss:.6f}"
            )


def evaluate(capsule_net, data_loader, epoch, device=torch.device("cpu")):
    capsule_net.to(device)
    capsule_net.eval()

    n_batch = np.ceil(len(data_loader.dataset) / data_loader.batch_size)
    with torch.no_grad():
        total_loss = 0
        for i, (batch_image, batch_label) in enumerate(tqdm(data_loader)):
            batch_image, batch_label = batch_image.to(device), batch_label.to(device)

            batch_obj_vectors, batch_reconstruction, batch_masks = capsule_net(batch_image)
            loss = capsule_net.loss(batch_obj_vectors, batch_label, batch_image, batch_reconstruction)

            accuracy = np.mean(
                np.argmax(batch_masks.detach().cpu().numpy(), 1) == batch_label.cpu().numpy()
            )
            total_loss += loss.item()
            avg_loss = loss.item() / float(data_loader.batch_size)

            if i % 100 == 0:
                tqdm.write(
                    f"Epoch: [{epoch}], Batch: [{i + 1}/{n_batch}], test accuracy: {accuracy:.6f}, "
                    f"loss: {avg_loss:.6f}"
                )


if __name__ == '__main__':
    BATCH_SIZE = 100
    EPOCHS = 30
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    DATASET_CONFIG = MNIST

    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=dataset_transform)
    test_dataset = datasets.MNIST('./data/mnist', train=False, download=True, transform=dataset_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(1)

    capsule_net = CapsNet(**DATASET_CONFIG)
    capsule_net.to(DEVICE)

    optimizer = torch.optim.Adam(capsule_net.parameters())

    for e in range(1, 1 + EPOCHS):
        train(capsule_net, optimizer, train_loader, e, device=DEVICE)
        evaluate(capsule_net, test_loader, e, device=DEVICE)
