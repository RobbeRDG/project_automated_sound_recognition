import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from util import config

def train_classification_model(
        model,
        optimizer,
        loss_fn,
        scaler,
        dataloader,
    ):
    total_loss = 0.0
    
    loop = tqdm(dataloader)
    for idx, (inputs, labels) in enumerate(loop):
        # Send the inputs and labels to device
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

        # Zero the parameter gradient
        optimizer.zero_grad()

        # Runs the forward pass with autocasting.
        with torch.autocast(device_type=config.DEVICE.type):
            outputs = model(inputs)           
            loss = loss_fn(outputs, labels)

        # Update the total loss
        total_loss += loss.item()

        # Backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Return the total loss
    return total_loss
