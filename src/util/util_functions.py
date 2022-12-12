from util import config

def extract_onehot_label_from_filename(filename):
    # Extract label from filename
    filename_label = filename.split('-')[0]

    # One-hot encode
    onehot_vector = []
    for label in config.LABELS:
        if filename_label == label: onehot_vector.append(1)
        else: onehot_vector.append(0)

    return onehot_vector