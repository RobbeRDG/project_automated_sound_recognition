import random
import numpy as np
from tqdm import tqdm
import soundfile as sf
import os
from os.path import join
from util import config, feat_extract

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

    if augment:
        # Unpack the audiomentations config
        audiomentations_augmentations = audiomentations_config["augmentations"]

        # Unpack the specaugment config
        specaugment_configurations = specaugment_config["configuration"]

    # Initialize the feature extractor class
    feature_extractor = feat_extract.FE(config.featconf)

    # Go through each audio file in the files list and generate augmented copies
    for file in tqdm(files_list):
        for copy in tqdm(range(n_copies)):
            # Read the audio data
            in_data, samplerate = sf.read(join(config.RAW_DATA_PATH, file))

            # TODO, find out why this is needed
            #if in_data.shape[1] >=2:
            #    in_data = in_data[:,1]
            
            # Give error if samplerate is different from featconf['samFreq']
            assert samplerate == config.featconf['samFreq'], "Samplerate of the .wav file differs from the feature configuration!"

            if augment:
                # Augment the audio data using audiomentations
                out_data = audiomentations_augmentations(samples=in_data, sample_rate=samplerate)

                # The resulting data after this augmentation is still raw spectogram data, however, since the specaugment transformations work on the mel features
                # we first need to extract these mel bins
                out_data = feature_extractor.fe_transform(out_data)

                # Now that the sample is converted to mel bins, we can pass it through the offline specaugment to get the final output
                out_data = spec_augment_offline(out_data, specaugment_configurations)
            else:
                # Just generate the features
                out_data = feature_extractor.fe_transform(in_data)

            # The file string hast the form "audio/airport-barcelona-203-6129-0-a.wav"
            # We remove the audio folder prefix and '.wav' extension
            # We finally add the copy number 
            out_file_name = file[6:-4] + '-' + str(copy)

            # Save the out_data features
            np.save(join(result_export_path, f'{out_file_name}.npy'), out_data)

            # Also save a spectrogram of the features
            plt = feature_extractor.create_plot(out_data, file)
            plt.savefig(join(result_export_path, f'{out_file_name}.png'))
            plt.close()