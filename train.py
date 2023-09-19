import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import dataset
from networks import Generator, Discriminator

# from torchsummary import summary
import matplotlib.pyplot as plt
import wandb
import io
from PIL import Image

wandb.login(key="796a636ca8878cd6c1494d1282f73496c43e6b31")

wandb.init(project="3dgan", entity="jacksonherberts")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Number of parameters
# print(summary(generator, (200, 1, 1, 1)))
# print(summary(discriminator, (1,64,64,64)))

# "We set the learning rate of G to 0.0025,D to 10−5,and use a batch size of 100"
learning_rate_G = 0.0025
learning_rate_D = 1e-5
batch_size = 100
epochs = 100
loss = nn.BCELoss()

# "We use ADAM for optimization, with β = 0.5"
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(
    discriminator.parameters(), lr=learning_rate_D, betas=(0.5, 0.999)
)

train_loader, _ = dataset.get_dataloaders(batch_size=batch_size)


# Training loop
def train():
    D_loss = []
    G_loss = []
    for epoch in range(1, epochs + 1):
        tqdm_batch = tqdm(
            train_loader, total=len(train_loader), leave=False, dynamic_ncols=True
        )
        for _, batch in enumerate(tqdm_batch):
            real_data = batch["voxel"]  # Batch of real 3D voxel samples
            real_data = real_data.unsqueeze(1).to(device)
            batch_size = real_data.size(0)

            # Discriminator training
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1, 1, 1, 1, dtype=torch.float32).to(
                device
            )
            fake_labels = torch.zeros(batch_size, 1, 1, 1, 1, dtype=torch.float32).to(
                device
            )

            # Forward pass for real data
            real_outputs = discriminator(real_data)

            # Generate fake data and forward pass for fake data
            z = torch.rand(batch_size, 200, 1, 1, 1).to(device)  # Sample random noise
            fake_data = generator(z)
            fake_outputs = discriminator(fake_data.detach())

            # Discriminator loss
            d_loss_real = loss(real_outputs, real_labels)
            d_loss_fake = loss(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            # optimizer_D.step()

            # Generator training
            optimizer_G.zero_grad()
            z = torch.rand(batch_size, 200, 1, 1, 1).to(device)  # Sample random noise
            fake_data = generator(z)
            fake_outputs = discriminator(fake_data)

            # Generator loss
            g_loss = loss(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Adaptive training strategy for the discriminator
            # Assuming accuracy is computed based on how well D discriminates real vs. fake
            accuracy = (
                (real_outputs > 0.5).float().mean()
                + (fake_outputs <= 0.5).float().mean()
            ) / 2
            if accuracy < 0.8:
                optimizer_D.step()

            # print(torch.count_nonzero(fake_data[0] > 0.5).item())

        D_loss.append(d_loss.item())
        G_loss.append(g_loss.item())

        # Print loss and accuracy for monitoring
        tqdm_batch.set_description(
            f"Epoch [{epoch + 1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, D Accuracy: {accuracy.item():.4f}"
        )
        print(
            f"Epoch [{epoch + 1}/{epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(D_loss[: epoch + 1], label="D Loss")
        ax.plot(G_loss[: epoch + 1], label="G Loss")
        ax.set_ylabel("Loss")
        ax.legend()

        # Log the metrics and the combined loss plot to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "loss D": d_loss.item(),
                "loss G": g_loss.item(),
                "accuracy": accuracy.item(),
                "loss plot": wandb.Image(fig),
            }
        )
        plt.close(fig)

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"generator_ckpt_{epoch}")
            torch.save(discriminator.state_dict(), f"discriminator_ckpt_{epoch}")

    # plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(D_loss, label="D Loss")
    plt.plot(G_loss, label="G Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    wandb.log({"loss plot": wandb.Image(plt)})


def generate_samples(number_samples):
    generator.load_state_dict(torch.load(f"generator_ckpt_100"))
    for i in range(number_samples):
        z = torch.rand(1, 200, 1, 1, 1).to(device)  # Sample random noise
        fake_data = generator(z)
        fake_data = (fake_data[0][0] > 0.5).detach().cpu().numpy()
        ax = plt.figure().add_subplot(projection="3d")
        ax.voxels(fake_data)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)

        wandb.log({"generated samples": wandb.Image(img)})
        buf.close()
        plt.close()


# train()
generate_samples(10)
