


# Image Classification *(PyTorch and CIFAR-10)* 🏞️
Hi! This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the [**CIFAR-10** dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The model predicts categories like airplanes, cars, birds, and more using image data, including functionality for training, evaluation, and prediction on new images.
 
 If you have CUDA installed, this will mainly work on your GPU when you proper install torchvision resources.
 If needed please check their documentation: https://pytorch.org/get-started/locally/
 

## Table of Contents

-   [What libs do I need](#what-libs-do-i-need)
-   [Testing the program](#testing-the-program)
-   [Dataset](#dataset)
-   [Training](#training)
-   [Prediction](#prediction)
-   [Future Features](#future-features)
-   [Contributing](#contributing)
-   [License](#license)


## What libs do I need? 
<img src="https://imgur.com/oMCL4sP.png" alt="drawing" width="400"/>

### Requirements

-   [Python 3.6+](https://www.python.org/)
-   [PyTorch](https://pytorch.org/)
-   [Torchvision](https://pypi.org/project/torchvision/)
-   [PIL (Python Imaging Library)](https://pillow.readthedocs.io/en/stable/)
-   [Matplotlib](https://matplotlib.org/)

## Testing the program

### 1. Downloading the CIFAR-10 Dataset

The dataset is automatically downloaded when you run the script for the first time if it's not already present locally.

### 2. Running the Code

The code is organized to perform the following tasks:

-   **Train the model** on the CIFAR-10 dataset
-   **Evaluate** its accuracy on test data
-   **Predict** the class of custom images

To run the full training, evaluation, and prediction pipeline, execute:


       python main.py


## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The dataset will be downloaded to the `./data` directory. It contains:

-   Airplane
-   Car
-   Bird
-   Cat
-   Deer
-   Dog
-   Frog
-   Horse
-   Ship
-   Truck

## Training

The neural network used consists of:

-   Two convolutional layers with ReLU and MaxPooling
-   Three fully connected layers for classification

The network is optimized using Stochastic Gradient Descent (SGD) with momentum.

To train the model, uncomment the lines inside the `train_model()` function:
![train_model](https://i.imgur.com/Udm7FpW.png)


## Prediction

You can predict the class of any image (in PNG, JPG, or JPEG format) stored in the `./images/` folder. Each image is resized to 32x32 pixels to match the CIFAR-10 dimensions.
![predict](https://i.imgur.com/VTV3Hvo.png)
### Example using these images:

Place the image inside the `./images/` folder and run the code. The prediction results will be displayed alongside the image, with probabilities for each class.

<img src="https://i.imgur.com/Lndvxfq.jpeg"  alt="drawing" width="300" align="center"/>
<img src="https://i.imgur.com/iLncOzo.jpeg" alt="drawing" width="300" align="center"/>
<img src="https://i.imgur.com/aDtlROu.jpeg" alt="drawing" width="300" align="center"/>
<img src="https://i.imgur.com/YMvWm3O.jpeg" alt="drawing" width="300" align="center"/>
<img src="https://i.imgur.com/IPNaSPH.jpeg" alt="drawing" width="300" align="center"/>
<img src="https://i.imgur.com/WTO4eVu.jpeg" alt="drawing" width="300" align="center"/>

#### Results:
<img src="https://i.imgur.com/8Y8cUwl.png" alt="drawing" width="400" align="center"/>
<img src="https://i.imgur.com/KzimmWA.png" alt="drawing" width="400" align="center"/>
<img src="https://i.imgur.com/FUFxHuV.png" alt="drawing" width="400" align="center"/>
<img src="https://i.imgur.com/8XfppPn.png" alt="drawing" width="400" align="center"/>
<img src="https://i.imgur.com/TVLWoTN.png" alt="drawing" width="400" align="center"/>
<img src="https://i.imgur.com/wZLajLc.png" alt="drawing" width="400" align="center"/>

### Future Features:
- Integrate with CIFAR-100 option to choose from
- Maybe some UI integration and Text-to-speech capability


## Contributing

Feel free to submit issues and enhancement requests, or fork the repository and send pull requests.

## License

This project is licensed under the MIT License.
