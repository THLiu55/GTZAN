import os
import time
import json
import argparse
from utils.params import Params
from utils.preprocess import load_audio, load_data
from sklearn.model_selection import train_test_split
import torch.optim as optim
from models.CNN import CNN
import torch.nn as nn
import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    parser.add_argument(
        "--mode", 
        type=str, 
        )
    args = parser.parse_args()

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    params = Params("hparams.yaml", args.model_name)
    if args.mode == "preprocess":
        load_audio(params)
    elif args.mode == "train":
        # Load & split data
        inputs,targets = load_data(params)
        inputs_train,inputs_test,targets_train,targets_test = train_test_split(inputs,targets,test_size=0.3)
        inputs_train = torch.Tensor(inputs_train).to(device)
        targets_train = torch.LongTensor(targets_train).to(device)
        inputs_test = torch.Tensor(inputs_test).to(device)
        targets_test = torch.LongTensor(targets_test).to(device)

        # Train model
        model = CNN(inputs_train.shape, 10).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        # Set up training parameters
        num_epochs = 100

        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            for i in tqdm(range(0, len(inputs_train))):
                
                # Forward pass
                outputs = model(inputs_train)
                loss = criterion(outputs, targets_train)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.item()


            # calculate validation loss, accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(inputs_test)
                loss = criterion(outputs, targets_test)
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == targets_test).sum().item()
                accuracy = correct / len(targets_test)
                print(f"EPOCH {epoch + 1}, train_loss: {running_loss / len(inputs_train)}, val_loss: {loss.item()}, val_acc: {accuracy}")
        print('Finished Training')



if __name__ == "__main__":
    main()