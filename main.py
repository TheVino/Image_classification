import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import multiprocessing
import matplotlib.pyplot as plt

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Is cuda available? {torch.cuda.is_available()}')            # Just uncomment this if you need to check if your GPU is avail


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the class names for CIFAR-10
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def download_data(root='./data', download=False):

    # Path to the CIFAR-10 dataset folder
    dataset_folder = os.path.join(root, 'cifar-10-batches-py')

    # Define the expected files
    expected_files = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5',
        'test_batch', 'batches.meta'
    ]
    
    # Check if all expected files are present in the dataset folder
    files_present = all(
        os.path.isfile(os.path.join(dataset_folder, file))
        for file in expected_files
    )

    if not files_present:
        print("Files not found. Downloading dataset...")
        download = True
    else:
        print("Files already downloaded and verified.")
        download = False

    """Download CIFAR-10 dataset and prepare data loaders."""
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=download)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=download)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
    return train_data, test_data, train_loader, test_loader

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Define the layers of the neural network
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(train_loader, net, optimizer, loss_function, device):
    """Train the neural network."""
    for epoch in range(50):
        running_loss = 0.0
        
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Loss: {running_loss / len(train_loader):.4f}')

def load_image(image_path):
    """Load and preprocess an image."""
    new_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to CIFAR-10 image size
        transforms.ToTensor(),        # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    image = new_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)  # Move to the device

def predict_image(image_path, model, class_names):
    """Predict the class of a given image and display the result."""
    image = load_image(image_path)
    
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        outputs = model(image)   # Forward pass
        probabilities = F.softmax(outputs, dim=1)  # Get probabilities for each class
        prob_list = probabilities.squeeze().cpu().numpy()  # Convert to numpy array
        
        # Sort the probabilities and class names by probability in descending order
        sorted_indices = prob_list.argsort()[::-1]
        sorted_probs = prob_list[sorted_indices]
        sorted_class_names = [class_names[i] for i in sorted_indices]

        # Get the predicted class index and name
        probs, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]

        # Set up the figure size and background color
        plt.figure(figsize=(8, 8))
        plt.gcf().patch.set_facecolor('gray')  # Set the background color to light gray
        plt.subplots_adjust(top=0.6)  # Adjust the top to provide more space for text

        # Load the image for visualization
        img = Image.open(image_path).convert('RGB')
        plt.imshow(img)
        
        # Display the sorted prediction probabilities with color coding
        start_y = 0.9  # Starting Y position for the text
        spacing = 0.08  # Vertical space between each label

        for i, prob in enumerate(sorted_probs):
            if prob < 0.03:
                color = 'red'
            elif 0.03 < prob < 0.6:
                color = 'orange'
            else:
                color = 'green'

            y_pos = start_y - i * spacing  # Adjust the y position based on index
            plt.text(1.02, y_pos, f"{sorted_class_names[i]}: {prob*100:.2f}%", color=color, fontsize=10, bbox=dict(facecolor='lightgray', edgecolor='black', alpha=0.7),
                     transform=plt.gca().transAxes, ha='left', va='top')

        plt.title(f'Prediction: {predicted_class}', fontsize=12, color='black')
        plt.axis('off')  # Hide the axes
        plt.show()

def load_trained_model(net, device, model_path='trained_net.pth'):
    """Load the trained model if available."""
    if os.path.isfile(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {model_path}")
    else:
        print(f"No trained model found at {model_path}. Starting from scratch.")
    return net

def evaluate_model(net, test_loader, device):
    """Evaluate the model on the test dataset."""
    correct = 0
    total = 0
    net.eval()
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # Move test data to GPU
            outputs = net(images)   # Forward pass
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

def get_image_paths(folder_path='./images/'):
    """Get the list of image paths from the specified folder."""
    return [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

def main():
    # Ensure freeze_support is called for Windows
    multiprocessing.freeze_support()

    # Download data and prepare the data loader
    train_data, test_data, train_loader, test_loader = download_data()

    # Initialize neural network, loss function, and optimizer
    net = NeuralNet().to(device)  # Move the network to the GPU
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # Train the model (Uncomment 2 lines below if you need to train)
    # train_model(train_loader, net, optimizer, loss_function, device)
    # torch.save(net.state_dict(), 'trained_net.pth')  # Save trained model

    # Load trained model if available
    net = load_trained_model(net, device)

    # Evaluate the model
    evaluate_model(net, test_loader, device)

    # Load and predict each image in the folder
    image_paths = get_image_paths()
    for image_path in image_paths:
        predict_image(image_path, net, class_names)

if __name__ == "__main__":
    main()
