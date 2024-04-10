import os
import time
import json
import argparse
from utils.params import Params
from utils.preprocess import load_audio, prepare_data
from sklearn.model_selection import train_test_split
import torch.optim as optim
from models.CNN import CNN
from models.MLP import MLP
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from models.model_manager import getModel
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    args = parser.parse_args()

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # Load hyperparameters
    params = Params("hparams.yaml", args.model_name)

    # Load & split data
    inputs,targets = prepare_data(params)
    inputs_train,inputs_test,targets_train,targets_test = train_test_split(inputs,targets,test_size=0.3)
    inputs_train = torch.Tensor(inputs_train).to(device)
    targets_train = torch.LongTensor(targets_train).to(device)
    inputs_test = torch.Tensor(inputs_test).to(device)
    targets_test = torch.LongTensor(targets_test).to(device)

    # Train model
    model = getModel(params, inputs_train.shape, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    criterion = nn.CrossEntropyLoss()

    # Set up training parameters
    num_epochs = params.num_epochs
    batch_size = params.batch_size
    patience = params.patience  # Number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0  # Counter to keep track of the number of epochs without improvement
    log = {}

    # Training loop
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        for i in tqdm(range(0, len(inputs_train), batch_size)):
            # Get batch of inputs and targets
            inputs_batch = inputs_train[i:i+batch_size]
            targets_batch = targets_train[i:i+batch_size]
            
            # Forward pass
            outputs = model(inputs_batch)
            loss = criterion(outputs, targets_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
        
        # calculate training loss, validation loss, accuracy
        train_loss = running_loss / (len(inputs_train) // batch_size)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs_test)
            val_loss = criterion(outputs, targets_test)                    
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets_test).sum().item()
            accuracy = correct / len(targets_test)

        # save log
        log[epoch] = {
            'train_loss': train_loss,
            'val_loss': val_loss.item(),
        }
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, params.checkpoint_path)

        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping")
                break
        print(f"EPOCH {epoch + 1}, train_loss: {train_loss}, val_loss: {val_loss.item()}, val_acc: {accuracy}")            
    print('Finished Training')

    print("Performance on test set:")
    print(classification_report(targets_test.detach().cpu().numpy(), torch.argmax(model(inputs_test), dim=1).detach().cpu().numpy()))

    # Save log
    log_path = params.log_path
    with open(log_path, 'w') as f:
        json.dump(log, f)
    print(f"Log saved at {log_path}")


if __name__ == "__main__":
    main()