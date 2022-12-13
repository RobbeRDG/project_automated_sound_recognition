import datetime
from util import config
import torch

def extract_onehot_label_from_filename(filename):
    # Extract label from filename
    filename_label = filename.split('-')[0]

    # One-hot encode
    onehot_vector = []
    for label in config.LABELS:
        if filename_label == label: onehot_vector.append(1)
        else: onehot_vector.append(0)

    return torch.Tensor(onehot_vector)

def idx_to_onehot(idx):
    return torch.nn.functional.one_hot(torch.Tensor(idx), num_classes= len(config.LABELS))

def generate_run_name_from_config(augmentations):
    # Build up a string that contains the identifier of each used augmentation method
    augmentations_id_string = ''
    if augmentations['noise']['enabled']: augmentations_id_string += 'n'
    if augmentations['pitch_shift']['enabled']: augmentations_id_string += 'p'
    if augmentations['mixup']['enabled']: augmentations_id_string += 'm'
    if augmentations['freq_mask']['enabled']: augmentations_id_string += 'f'
    if augmentations['time_mask']['enabled']: augmentations_id_string += 't'

    # Define other information for the run name
    general_info_string = f'{config.BATCH_SIZE}_{config.LR}'

    # Define timestamp
    now = datetime.datetime.now()
    timestamp_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    return f'{general_info_string}_{augmentations_id_string}_{timestamp_string}'

