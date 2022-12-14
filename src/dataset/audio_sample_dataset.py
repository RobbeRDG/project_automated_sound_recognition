import random
from torch.utils.data import Dataset
import os
from os.path import join
from util import util_functions, config
import torchaudio
from torchaudio import transforms
#from torch_audiomentations import PitchShift
from audiomentations import AddGaussianNoise, PitchShift
import numpy as np
import torch
import seaborn as sn
from util import feat_extract

class AudioSampleDataset(Dataset):
    def __init__(
            self,
            raw_audio_samples_path,
            augmentations
        ):

        # Store the base path
        self.raw_audio_samples_base_path = raw_audio_samples_path

        # Get the raw audio sample filenames
        self.sample_filenames = os.listdir(raw_audio_samples_path)

        # Set the augmentations
        self.augmentations = augmentations

        # Set a feature extractor instance
        self.feature_extractor = feat_extract.FE(config.featconf)

    def __len__(self):
        return len(self.sample_filenames)

    def __getitem__(self, idx):
        # Get the raw sample
        waveform, label, sample_rate = self.get_raw_sample(idx)

        # Apply the requested "raw data" transformations
        waveform, label = self.apply_raw_transforms(waveform, label, sample_rate)

        # Transform to mel spectrogram
        waveform = self.raw_to_mel(waveform[0])
        
        # Apply the requested "mel data" transformations
        waveform, label = self.apply_mel_transforms(waveform, label)
        

        # TODO Save image of spectrogram to see if everything is working

        # Finally, we use a CV network for RGB data, but we only have 1 channel
        # So, copy this channel 3 times
        waveform = torch.stack([waveform, waveform, waveform], dim=0)

        return waveform, label

    def get_raw_sample(self, idx):
        # Get the requested sample filename
        sample_filename = self.sample_filenames[idx]

        # Get the onehot label for the image
        label = util_functions.extract_onehot_label_from_filename(sample_filename)

        # Set the sample file path
        sample_file_path = join(self.raw_audio_samples_base_path, sample_filename)

        # Load in the audio file
        waveform, sample_rate = torchaudio.load(sample_file_path)

        return waveform, label, sample_rate


    def apply_raw_transforms(self, waveform, label, sample_rate):
        if self.augmentations['noise']['enabled']:
                transform = AddGaussianNoise(
                    min_amplitude= self.augmentations['noise']['min_amplitude'],
                    max_amplitude= self.augmentations['noise']['max_amplitude'],
                    p= self.augmentations['noise']['p']
                )
                waveform = transform(waveform, sample_rate)

        if self.augmentations['pitch_shift']['enabled']:
            transform = PitchShift(
                min_semitones= self.augmentations['pitch_shift']['min_semitones'],
                max_semitones= self.augmentations['pitch_shift']['max_semitones'],
                p= self.augmentations['pitch_shift']['p']
            )
            waveform = transform(waveform, sample_rate)

            # This method produces a list, so convert it back to tensor
            waveform = torch.Tensor(waveform)

        return waveform, label

    def raw_to_mel(self, waveform):
        waveform = self.feature_extractor.fe_transform(waveform)

        # This method produces a list, so convert it back to tensor
        waveform = torch.Tensor(waveform)

        # Transpose the matrix
        waveform = torch.transpose(waveform, 0, 1)

        return waveform

    def apply_mel_transforms(self, waveform, label):
        if self.augmentations['mixup']['enabled']:
            if random.random() < self.augmentations['mixup']['p']:
                # Pick another audio sample randomly
                random_idx = random.randint(0, len(self.sample_filenames)-1)

                # Get the raw sample
                waveform_other_sample, label_other_sample, sample_rate_other_sample = self.get_raw_sample(random_idx)

                # Apply the requested "raw data" transformations
                waveform_other_sample, label_other_sample = self.apply_raw_transforms(waveform_other_sample, label_other_sample, sample_rate_other_sample)

                # Transform to mel spectrogram
                waveform_other_sample = self.raw_to_mel(waveform_other_sample[0])

                # Mixup sample features
                alpha = self.augmentations['mixup']['alpha']
                lam = np.random.beta(alpha, alpha)
                waveform = torch.add(torch.mul(lam, waveform), torch.mul((1 - lam), waveform_other_sample))

                # Also mixup labels
                label = torch.add(torch.mul(lam, label), torch.mul((1 - lam), label_other_sample))

                # Since the produced lambda will usually be very big or very small,
                # a good simplification is to just pick the majority label
                #mayority_class_idx = torch.argmax(label)
                #label = util_functions.idx_to_onehot(mayority_class_idx)


        if self.augmentations['freq_mask']['enabled']:
            if random.random() < self.augmentations['freq_mask']['p']:
                transform = transforms.FrequencyMasking(
                    freq_mask_param= self.augmentations['freq_mask']['freq_mask_param']
                )
                waveform = transform(waveform)

        if self.augmentations['time_mask']['enabled']:
            if random.random() < self.augmentations['time_mask']['p']:

                # TimeMasking wants extra dimention in front
                waveform = waveform.unsqueeze(0)

                transform = transforms.TimeMasking(
                    time_mask_param= self.augmentations['time_mask']['time_mask_param']
                )
                waveform = transform(waveform)

                # Get rid of extra dimention
                waveform = waveform[0]

        return waveform, label

