import torch
from os.path import join
import wandb
import datetime
from util import config

def save_model(model, checkpoint_path, save_as_artifact, artifact_name):
    # Set the best model
    best_model_state = model.state_dict()

    # Save the best model
    torch.save(best_model_state, checkpoint_path)

    # Also save the final model as an artifact
    if save_as_artifact:
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)


    return best_model_state

def get_artifact_model_weights(artifact_path, checkpoint_file):
    # Download the artifact
    artifact = wandb.use_artifact(artifact_path, type='model')

    model_weights = artifact.get_path(checkpoint_file)

    return model_weights.download()
