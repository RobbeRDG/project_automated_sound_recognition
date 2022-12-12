import torch
from tqdm import tqdm

from util import config
from sklearn.metrics import f1_score


def validate_classification_model(
        model,
        loss_fn,
        dataloader,
    ):
    # Set the evaluation data structures
    total_loss = 0.0
    pred_classes = []
    true_classes = []

    # No gradients needed during validation
    with torch.no_grad():
        # First calculate the validation loss
        loop = tqdm(dataloader, leave=True)
        for idx, (input, label) in enumerate(loop):
            # Send the input and label to device
            input, label = input.to(config.DEVICE), label.to(config.DEVICE)

            # Runs the forward pass.
            output = model(input)
            loss = loss_fn(output, label)

            # Add the loss to the total
            total_loss += loss

            # Get the predicted class
            pred_class = torch.argmax(output[0])

            # update the evaluation data
            pred_classes.append(pred_class.item())
            true_classes.append(label.item())  

    # Get the average f1 score
    f1 = f1_score(true_classes, pred_classes, average=None)

    # Get the average f1 score
    f1_avg = f1_score(true_classes, pred_classes, average='macro')

    return total_loss, f1, f1_avg

    


