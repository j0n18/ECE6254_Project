from autoencoders import ConvolutionalVAE
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def visualize_data(dataset='mnist'):
    fashion_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"}
    vis_fig = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        idx = np.argwhere(np.array(training_data.targets) == i - 1)
        img, label = training_data[idx.flatten()[0]]
        vis_fig.add_subplot(rows, cols, i)
        if dataset == 'FashionMNIST':
            plt.title(fashion_map[label])
        elif dataset == 'CIFAR':
            plt.title(training_data.classes[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# download the datasets
data_dir = r"C:\Users\abdos\Downloads\GT Academics\Spring 2022\ECE 6254 Statistical ML\Project\Datasets"

training_data = datasets.CIFAR100(
    root=data_dir,
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.CIFAR100(
    root=data_dir,
    train=False,
    download=True,
    transform=ToTensor()
)

""" VISUALIZE DATA (COMMENT OUT IF NOT NEEDED) """
visualize_data('CIFAR')

""" CREATE THE DATA LOADERS """
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

""" BUILD THE AUTOENCODER """
input_shape = training_data[0][0].shape
hidden_layers = [64, 128, 256, 512, 1024, 2048]
latent_dim = 16
vae = ConvolutionalVAE(input_shape=input_shape,
                       hidden_layers=hidden_layers,
                       latent_dim=latent_dim)
adam_optim = torch.optim.Adam(vae.parameters, lr=0.001)
vae.set_optimizer(adam_optim)
vae.to(DEVICE)

""" TRAIN THE MODEL """
epochs = 100
tol = 0.005
train_losses = [np.inf]
for epoch in range(1, epochs + 1):
    epoch_loss = []
    for train_x, _ in train_loader:
        train_x = train_x.to(DEVICE)
        epoch_loss.append(vae.training_step(train_x))
    train_losses.append(torch.mean(torch.as_tensor(epoch_loss)))
    print(f"EPOCH NO. {epoch}: Loss = {train_losses[epoch]}")
    # break if difference between successive losses is too small
    if np.abs(train_losses[epoch] - train_losses[epoch-1]) < tol:
        break

plt.plot(np.array(train_losses[1:]))
plt.show()

""" SAVE THE MODEL """
flname = "VAE_Trained"
f = open(flname, 'wb')
vae.to('cpu')
torch.save(vae, f)

""" TEST THE MODEL """
# TODO
