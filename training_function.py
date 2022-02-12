# import libraries
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
import time

EXCEL_PATH = "/content/drive/MyDrive/CE Project/database.xlsx"
ROOT_DIR = "/content/drive/MyDrive/CE Project/image"
CSV_PATH = "/content/drive/MyDrive/CE Project/small.csv"


# ==== Split the Sample Dataset ====
def splitDataframe(total_df, dev_percent, test_percent, random_state=1):
    """
    create train, dev and test dataframe
    :param total_df: total dataframe
    :param dev_percent: percentage of dev sample size
    :param test_percent: percentage of test sample size
    :param random_state: random seed (default = 1)
    :return: train dataframe, dev dataframe, test dataframe
    """

    # Random indexing for total dataframe
    total_df = total_df.sample(frac=1, random_state=random_state)

    # Create split size for indexing
    total_sample = len(total_df)
    dev_sample = int(dev_percent * total_sample)
    test_sample = int(test_percent * total_sample)
    train_sample = total_sample - dev_sample - test_sample

    # Spliting the dataframe
    train_df = total_df[0:train_sample]
    dev_df = total_df[train_sample:train_sample + dev_sample]
    test_df = total_df[train_sample + dev_sample:]

    return train_df, dev_df, test_df


# ==== Create Data Frame Function ====
def createDataframe(paramreq="Final\nTotal", root_dir=ROOT_DIR, excel_path=EXCEL_PATH):
    """
    Crete Dataframe from total dataframe and check avaliable photo in directories
    Args:
      paramreq: Required parameter for analysis
      root_dir: Directory of photo file
      excel_path: Path of the excel whichs contain all parameters and photo path
    """

    # Read the whole data frame
    big_dataframe = pd.read_excel(excel_path)

    # Convert to small dataframe which contain filenames and required parameters
    small_dataframe = big_dataframe[["Filename", paramreq]]
    small_dataframe["Filename"] = small_dataframe["Filename"] + ".jpg"

    # List all available images in directory
    dirs = os.listdir(root_dir)

    # return the dataframe
    return small_dataframe[small_dataframe["Filename"].isin(dirs)]


# ==== Show the example image from tensors and label ====
def showImg(tensors, label):
    """
    Show the images and label on the titles
    Args:
      tensors: tensor of image (expected to be torch tensor with RGB channel)
      label: label of target image
    """
    plt.figure(figsize=(6, 8))
    plt.axis("off")
    np_img = np.transpose(tensors, (1, 2, 0))
    plt.imshow(np_img)
    plt.title(f"RMR Total Score {label:.1f}")


# ==== Function to train model ====
def trainModel(train_loader, test_loader, model, lossfun, optimizer, epochs, device=None, path = "Checkpoint.pt"):
    """
    Train the model with given loss function and optimizer
    Args:
      :param train_loader: Training dataloader
      :param test_loader: Test dataloader
      :param model: CNN model
      :param epochs: Number of training loops
      :param optimizer: optimizer (typically Adam)
      :param lossfun: loss function
      :param device: GPU (optional) for faster computation
    """

    # History
    history = {'lastAcc': 0.0}

    # Initialize the loss and accuracy
    trainLoss = np.zeros(epochs)
    valLoss = np.zeros(epochs)
    trainAcc = np.zeros(epochs)
    valAcc = np.zeros(epochs)
    trainingTime = np.zeros(epochs)

    # Sent model to GPU for faster computing
    if device:
        model.to(device)

    # Training Loops
    for epochi in range(epochs):

        # Initialize the time at the start of epoch
        start_time = time.time()

        # Switch model to train mode
        model.train()

        # Initializing batching loss and accuracy
        batchLoss = []
        batchAcc = []

        # Batching Loops
        for X, y in train_loader:

            if device:
                # Sent data to GPU
                X = X.to(device)
                y = y.to(device)

            # Forward propagation and compute loss
            yHat = model(X)
            loss = lossfun(yHat, y)

            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store batch accuracy and loss
            batchLoss.append(loss.item())
            batchAcc.append(torch.mean((torch.argmax(yHat, axis=1) == y).float()).item())

        # Store training accuracy and loss
        trainLoss[epochi] = np.mean(batchLoss)
        trainAcc[epochi] = np.mean(batchAcc) * 100

        # Evaulate the model on validation set
        model.eval()

        # Initializing batching loss and accuracy
        batchLoss = []
        batchAcc = []

        # Batching loops
        for X, y in test_loader:

            if device:
                # Sent data to GPU
                X = X.to(device)
                y = y.to(device)

            # Forward propagation and compute loss
            with torch.no_grad():
                yHat = model(X)
                loss = lossfun(yHat, y)

            # Store batch accuracy and loss
            batchLoss.append(loss.item())
            batchAcc.append(torch.mean((torch.argmax(yHat, axis=1) == y).float()).item())

        # Store validation accuracy and loss
        valLoss[epochi] = np.mean(batchLoss)
        valAcc[epochi] = np.mean(batchAcc) * 100

        # Stop the time
        stop_time = time.time()

        # Save the best model based on validation Accuracy
        if history["lastAcc"] < (np.mean(batchAcc) * 100):
            history["lastAcc"] = np.mean(batchAcc) * 100
            torch.save(model.state_dict(), path)
            history["epoch"] = epochi
        
        # Recorded training time
        training_time = -(start_time - stop_time)
        trainingTime[epochi] = training_time

        print(f"==== Epochs: {epochi+1} / {epochs} ====")
        print(f"Train Loss: {trainLoss[epochi]:.6f}, Train Acc: {trainAcc[epochi]:.2f}")
        print(f"Test Loss: {valLoss[epochi]:.6f}, Test Acc: {valAcc[epochi]:.2f}")
        print(f"Time: {training_time:.1f}s")

    return trainLoss, trainAcc, valLoss, valAcc, trainingTime, history, model


# ==== Make predictions on dataloader ====
def makePreds(dataloader, model, device=None):
    """
    Make prediction on dataloader in case, test data is too large
    :param dataloader: Dataloader
    :param model: Model
    :param device: Using GPU if needed
    :return:
    """
    # initialize prediction
    y_preds = np.array([])

    # Send model to GPU
    if device:
        model.to(device)

    # Evaluate the model on dataloader
    model.eval()
    for X, _ in dataloader:
        # Send data to GPU
        if device:
            X = X.to(device)

        # Make prediction
        with torch.no_grad():
            yHat = model(X)

        # Store yHat on y_preds
        y_preds = np.concatenate((y_preds, yHat.items()), axis=0)

    return y_preds


# ==== Create Custom Dataset Class ====
class RockFaceDataset(Dataset):
    """ Geological face mapping custom dataset """

    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_path (string): Path to csv_file with annotations.
            root_dir (string): Directory with all images
            transform : Optional transformation to be applied on image sample
        """

        self.rock_dataframe = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.rock_dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.rock_dataframe.iloc[idx, 0])
        image = mpimg.imread(img_name)
        label = torch.tensor(self.rock_dataframe.iloc[idx, 1], dtype = torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
