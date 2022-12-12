import random
import numpy as np
from tqdm import tqdm
import soundfile as sf
import os
from os.path import join
from util import config, feat_extract
from keras_cv.layers.preprocessing import MixUp
import tensorflow as tf

# Create a mixup instance 
mixup = MixUp(alpha = 0.2, seed  = 10)

# Initialize the feature extractor class
feature_extractor = feat_extract.FE(config.featconf)

def spec_augment_offline(spectrogram, configuration):
    '''
        mel_spectrogram: 2d mel spectrogram of size n*v, with n being the number of time frames and v being the number of mel frequency bins.
        configuration: dictionary with different parameters for spec augment.  
    '''
    # Get spectrogram shape
    width, height = spectrogram.shape

    # For the configured amount of frequency and time bins, already configure the size of these bins
    h_percentage = np.random.uniform(low=configuration['mask_fraction_min'], high=configuration['mask_fraction_max'],size=configuration['num_freq_lines'])
    w_percentage = np.random.uniform(low=configuration['mask_fraction_min_time'], high=configuration['mask_fraction_max_time'], size=configuration['num_time_lines'])

    # Take copy of the spectrogram
    new_input = spectrogram.copy()

    # Apply masking a set amount of random frequency bins
    for i in range(configuration['num_freq_lines']):
        if random.random() < configuration['p_spec_augment']:
            h_mask = int(np.ceil(h_percentage[i] * height))
            h = int(np.ceil(np.random.uniform(0.0, height - h_mask)))
            new_input[:, h:h + h_mask] = configuration['fill_val']

    # Apply masking a set amount of random time bins
    for j in range(configuration['num_time_lines']):
        if random.random() < configuration['p_spec_augment']:
            w_mask = int(np.ceil(w_percentage[j] * width))
            w = int(np.ceil(np.random.uniform(0.0, width - w_mask)))
            new_input[w:w + w_mask,:] = configuration['fill_val']

    return new_input


def generate_and_store_augmented_mel_features(
        files_list,
        result_export_path,
        n_copies, 
        augment,
        audiomentations_config=None,
        specaugment_config=None
    ):
    '''
        This function (1) takes the raw WAV features from a file, (2) applies the specified augmentations and (3) converts the raw features to the log mel representation
        The final result is stored in 
    '''

    if augment[0]:
        # Unpack the audiomentations config
        audiomentations_augmentations = audiomentations_config["augmentations"]

    if augment[1]:
        # Unpack the specaugment config
        specaugment_configurations = specaugment_config["configuration"]

    if augment[2]:
        # Unpack the mixup config
        mixup_alpha = specaugment_config["alpha"]
        mixup_probability = specaugment_config["prob"]


    # Go through each audio file in the files list and generate augmented copies
    for file in tqdm(files_list, position=0, leave=True):
        # Get the isolated file name
        isolated_file_name = file.split('/')[-1]

        for copy in range(0, n_copies):
            # Read the audio data
            in_data, samplerate = sf.read(file)

            # TODO, find out why this is needed
            #if in_data.shape[1] >=2:
            #    in_data = in_data[:,1]
            
            # Give error if samplerate is different from featconf['samFreq']
            assert samplerate == config.featconf['samFreq'], "Samplerate of the .wav file differs from the feature configuration!"

            if augment[0] or augment[1] or augment[2]:
                # Augment the audio data using audiomentations
                out_data = audiomentations_augmentations(samples=in_data, sample_rate=samplerate) if augment[0] else in_data

                # The resulting data after this augmentation is still raw spectogram data, however, since the nex transformations work on the mel features
                # we first need to extract these mel bins
                out_data = feature_extractor.fe_transform(out_data)

                out_data = mixup(out_data, files_list, mixup_alpha, mixup_probability) if augment[2] else out_data

                # Now that the sample is converted to mel bins, we can pass it through the offline specaugment to get the final output
                out_data = spec_augment_offline(out_data, specaugment_configurations) if augment[1] else out_data
            else:
                # Just generate the features
                out_data = feature_extractor.fe_transform(in_data)

            
            # We finally add the copy number 
            out_file_name = isolated_file_name[:-4] + '-' + str(copy)

            # Save the out_data features
            np.save(join(result_export_path, f'{out_file_name}.npy'), out_data)

            # Also save a spectrogram of the features
            plt = feature_extractor.create_plot(out_data, file)
            plt.savefig(join(result_export_path, f'{out_file_name}.png'))
            plt.close()

def mixup(original_data, files_list, mixup_alpha, mixup_probability):
    if random.random() < mixup_probability:
        # Read in a random audio file from the file list
        random_idx = random.randint(0,len(files_list)-1)
        random_audiofile = sf.read(files_list[random_idx])

        random_audiofile_mel_features = feature_extractor.fe_transform(random_audiofile)

        # 
        l = sample_beta_distribution(mixup_alpha)
        resulting_image = images_one * x_l + images_two * (1 - x_l)
        resulting_label = labels_one * y_l + labels_two * (1 - y_l)


def sample_beta_distribution(mixup_alpha):
    gamma_1_sample = tf.random.gamma(alpha=mixup_alpha)
    gamma_2_sample = tf.random.gamma(alpha=mixup_alpha)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)