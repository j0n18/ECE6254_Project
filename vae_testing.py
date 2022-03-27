from autoencoders import ConvolutionalVAE
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# download the datasets
data_dir = r"C:\Users\abdos\Downloads\GT Academics\Spring 2022\ECE 6254 Statistical ML\Project\Datasets"

training_data = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=ToTensor()
)

""" VISUALIZE DATA (COMMENT OUT IF NOT NEEDED) """
"""
vis_fig = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    idx = np.argwhere(training_data.targets == i-1)
    img, label = training_data[idx.flatten()[0]]
    vis_fig.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()
"""

""" CREATE THE DATA LOADERS """
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

""" BUILD THE AUTOENCODER """
input_shape = (1, 28, 28)
hidden_layers = [32, 64, 128, 256, 512]
latent_dim = 2
vae = ConvolutionalVAE(input_shape=input_shape,
                       hidden_layers=hidden_layers,
                       latent_dim=latent_dim)
adam_optim = torch.optim.Adam(vae.parameters, lr=0.001)
recons_loss = torch.nn.BCELoss(reduction='none')
vae.set_optimizer(adam_optim)
vae.to(DEVICE)

""" TRAIN THE MODEL """
epochs = 50
train_losses = []
for epoch in range(1, epochs + 1):
    epoch_loss = []
    for train_x, _ in train_loader:
        train_x = train_x.to(DEVICE)
        epoch_loss.append(vae.training_step(train_x))
    train_losses.append(torch.mean(torch.as_tensor(epoch_loss)))

    print(f"EPOCH NO. {epoch}: Loss = {train_losses[epoch - 1]}")

plt.plot(np.array(train_losses))
plt.show()

""" SAVE THE MODEL """
flname = "VAE_Trained"
f = open(flname, 'wb')
vae.to('cpu')
torch.save(vae, f)

""" TEST THE MODEL """
# TODO
