# Copyright 2020 Jaime Tierney, Adam Luchies, and Brett Byram

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the license at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

import h5py
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import warnings


class ApertureDataset(Dataset):
    """Aperture domain dataset."""

    def __init__(self, fname, num_samples, k=4, shuffle=False, target_is_data=False):
        """
        Args:
            fname: file name for aperture domain data
            num_samples: number of samples to use from data set
            k: frequency or depth bin to use
            shuffle: shuffle data when loading
            target_is_data: return data as the target (autoencoder)
        """

        self.fname = fname
        self.num_samples = num_samples
        self.k = k
        self.shuffle = shuffle
        self.target_is_data = target_is_data

        # check if files exist
        if not os.path.isfile(fname):
            raise IOError(fname + ' does not exist.')
        
        # Open file
        f = h5py.File(fname, 'r')

        # Get number of samples available for each type
        real_available = f['/' + str(self.k) + '/X/real'].shape[0]
        imag_available = f['/' + str(self.k) + '/X/imag'].shape[0]
        self.samples_available = min(real_available, imag_available)

        # set num_samples
        if not num_samples:
            num_samples = self.samples_available

        # make sure num_samples is less than samples_available
        if num_samples > self.samples_available:
            warnings.warn('data_size > self.samples_available. Setting data_size to samples_available')
            self.num_samples = self.samples_available
        else:
            self.num_samples = num_samples

        # load the data
        if self.shuffle:
            inputs = np.hstack([ f['/' + str(self.k) + '/X/real'][0:self.samples_available], 
                                f['/' + str(self.k) + '/X/imag'][0:self.samples_available] ] )

            if self.target_is_data:
                targets = np.hstack([ f['/' + str(self.k) + '/X/real'][0:samples_available], 
                                    f['/' + str(self.k) + '/X/imag'][0:self.samples_available] ] )
            else:
                targets = np.hstack([ f['/' + str(self.k) + '/Y/real'][0:self.samples_available], 
                                    f['/' + str(self.k) + '/Y/imag'][0:self.samples_available] ] )

            # shuffle data
            n = inputs.shape[1]
            XY = np.hstack([inputs, targets])
            np.random.shuffle(XY)
            inputs = XY[:, :n]
            targets = XY[:, n:]

            # keep only first num_samples
            inputs = inputs[0:self.num_samples]
            targets = targets[0:self.num_samples]

        else:
            inputs = np.hstack([ f['/' + str(self.k) + '/X/real'][0:self.num_samples], 
                                f['/' + str(self.k) + '/X/imag'][0:self.num_samples] ] )

            if self.target_is_data:
                targets = np.hstack([ f['/' + str(self.k) + '/X/real'][0:self.num_samples], 
                                    f['/' + str(self.k) + '/X/imag'][0:self.num_samples] ] )
            else:
                targets = np.hstack([ f['/' + str(self.k) + '/Y/real'][0:self.num_samples], 
                                    f['/' + str(self.k) + '/Y/imag'][0:self.num_samples] ] )

        # convert data to single precision pytorch tensors
        self.data_tensor = torch.from_numpy(inputs).float()
        self.target_tensor = torch.from_numpy(targets).float()

        # close file
        f.close()

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]

    def __str__(self):
        newline = "\n"
        a_str = f"ApertureDataset( {self.fname} )"
        a_str += newline
        a_str += f"{self.num_samples} / {self.samples_available}"
        a_str += newline
        a_str += f"k: {self.k}"

        return a_str
