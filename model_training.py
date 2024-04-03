import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import logging
import certifi
# Configure logging
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def plot_learning_curve(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def dnn_model_training(data_dir, model_save_path):
    print('Starting model training.')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print('Loading dataset.')
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': train_size, 'val': val_size}
    class_names = full_dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize lists to track loss and accuracy for plotting
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    def train_model(model, criterion, optimizer, num_epochs=50):
        print('Begin training loop.')
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes['train']
            epoch_acc = running_corrects.double() / dataset_sizes['train']

            # Evaluate on the validation set
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
            val_loss = val_running_loss / dataset_sizes['val']
            val_acc = val_running_corrects.double() / dataset_sizes['val']

            # Append losses for plotting
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)
            train_accs.append(epoch_acc.item())
            val_accs.append(val_acc.item())

            print(
                f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        return model

    model_trained = train_model(model, criterion, optimizer, num_epochs=50)
    torch.save(model_trained.state_dict(), model_save_path)
    print('Model training completed and model saved.')
    import json

    # Add this at the end of your dnn_model_training function, right before saving the model
    class_names_path = model_save_path.replace('.pth', '_class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(full_dataset.class_to_idx, f)  # Saves class to index mapping

    print(f'Class names saved to {class_names_path}')

    # Plot learning curves
    plot_learning_curve(train_losses, val_losses, train_accs, val_accs)



