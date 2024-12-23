# Trash Classification

This project involves training a Convolutional Neural Network (CNN) model to classify various types of trash using the TrashNet dataset. The dataset is obtained from [Hugging Face](https://huggingface.co/datasets/garythung/trashnet).

<div align="center">
    <img src="test-result.png" alt="test-result">
</div>

## 🛠️ Requirements

To run this project, you will need:

- Python
- Jupyter Notebook
- Huggingface for dataset
- WandB AI for experiment tracking
- PyTorch
- Scikit-learn
- Matplotlib
- Seaborn

## ⚙️ Setting Up
This project uses Huggingface and WandB. To run the project, you need to set up your Huggingface token and WandB API key.
### Huggingface
1. Create an account at https://huggingface.co/.
2. Get your token from Huggingface tokens page.
### WandB
1. Create a WandB account at https://wandb.ai/.
2. Get your API key by visiting https://wandb.ai/authorize.

## 🚀 How to Run

To run the model locally, you can follow these steps:

1. Clone this repository:
```bash
git clone https://github.com/your-username/trash-classification.git
cd trash-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login Huggingface and input your token
```bash
huggingface-cli login
```

4. Login WandB and input your API key
```bash
wandb.login()
```

5. Execute the Jupyter notebook:
```bash
jupyter notebook notebook/trash_classification.ipynb
```

## 🗃️ Dataset

The dataset used in this project is the TrashNet dataset, available on Hugging Face at https://huggingface.co/datasets/garythung/trashnet. This dataset contains labeled images of trash categorized into six classes: Cardboard, Glass, Metal, Paper, Plastic, and Trash, which are used for training and testing the CNN model.

## 🤖 Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) designed for image classification. It consists of the following layers:

1. Convolutional Layers: 
   The model uses three convolutional layers with increasing output channels (16, 32, 64). These layers help in extracting hierarchical features from the input images.

2. Activation Function: 
   After each convolutional operation, the ReLU (Rectified Linear Unit) activation function is applied to introduce non-linearity.

3. Max Pooling:
   After each convolutional layer, a 2x2 max pooling operation is applied to reduce the spatial dimensions of the feature maps, thereby reducing computational complexity and controlling overfitting.

4. Fully Connected Layers:
   After flattening the output from the last convolutional layer, the model passes it through two fully connected layers. The first fully connected layer has 512 neurons, and the second outputs the final class probabilities.
   
5. Output Layer:
   The output layer consists of 6 neurons corresponding to the 6 classes in the TrashNet dataset.