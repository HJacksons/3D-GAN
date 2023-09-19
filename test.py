import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from networks import Generator, Discriminator  # Import your Generator and Discriminator

# from torchsummary import summary
import matplotlib.pyplot as plt
import wandb

wandb.login(key="796a636ca8878cd6c1494d1282f73496c43e6b31")

wandb.init(project="3dgan", entity="jacksonherberts")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained discriminator
discriminator = Discriminator().to(device)
discriminator.load_state_dict(
    torch.load("discriminator_ckpt_20")
)  # Update the path to your pre-trained discriminator checkpoint


# Define a feature extractor that extracts features from layers 2, 3, and 4 and applies max pooling
class FeatureExtractor(nn.Module):
    def __init__(self, discriminator):
        super(FeatureExtractor, self).__init__()
        self.layer1 = list(discriminator.layers.children())[0]
        self.layer2 = list(discriminator.layers.children())[2]
        self.layer3 = list(discriminator.layers.children())[5]
        self.layer4 = list(discriminator.layers.children())[8]
        self.maxpool2 = nn.MaxPool3d(kernel_size=8)
        self.maxpool3 = nn.MaxPool3d(kernel_size=4)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        # Pass input through layers 1 to 4
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # Apply max pooling separately to layers 2, 3, and 4
        pooled2 = self.maxpool2(out2)
        pooled3 = self.maxpool3(out3)
        pooled4 = self.maxpool4(out4)

        # Concatenate the pooled results
        concatenated_features = torch.cat((pooled2, pooled3, pooled4), dim=1)
        # print(concatenated_features.shape)
        return concatenated_features


# Create a classifier on top of the feature extractor
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                7168, num_classes
            ),  # Concatenated feature length from layers 2, 3, and 4
            nn.Softmax(dim=1),  # Softmax for classification
        )

    def forward(self, features):
        out = self.classifier(features)
        return out


feature_extractor = FeatureExtractor(discriminator).to(device)
num_classes = 10
classifier = Classifier(num_classes).to(device)

# Print the total number of parameters of the classifier
total_params_classifier = sum(p.numel() for p in classifier.parameters())
print(f"Total number of parameters in the classifier: {total_params_classifier}")

# Create an optimizer and loss function for training
learning_rate_classifier = 0.001
optimizer_classifier = optim.Adam(classifier.parameters(), lr=learning_rate_classifier)
criterion = nn.CrossEntropyLoss()


# Training loop for the classifier (using your data loader)
def train_classifier(classifier, train_loader, num_epochs):
    classifier.train()
    train_loss_history = []
    train_accuracy_history = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in train_loader:
            inputs, labels = batch["voxel"].unsqueeze(1).to(device), batch["label"].to(
                device
            )  # Adjust according to your dataset
            optimizer_classifier.zero_grad()

            # Forward input through the modified feature extractor
            features = feature_extractor(inputs)

            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_classifier.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy})

    return train_loss_history, train_accuracy_history


# You can adjust the number of epochs as needed
num_epochs_classifier = 100

# Replace train_loader with your actual data loader
train_loader, _ = dataset.get_dataloaders(batch_size=100)
train_loss_history, train_accuracy_history = train_classifier(
    classifier, train_loader, num_epochs_classifier
)

# Plot training loss and accuracy history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history)
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

wandb.log({"train_loss_history": plt, "train_accuracy_history": plt})
