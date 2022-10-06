# EEG data preprocessing & feature extraction
# Foundations of Neural & Cognitive modelling, Lab 4
# 27-02-2020

import numpy as np
import pandas as pd
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import scipy                    
from scipy import signal
import pywt
import copy
import random
from random import shuffle
import sklearn
from sklearn import feature_selection

## Data loading & preprocessing

# Data for the motor imagery task available from PhysioNet is loaded through
# the MNE library. An example in the MNE documentation using this dataset
# (but using a different decoding procedure) is available here: 
# https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html

# data will be loaded from t - 0.5 sec to t + 0.5 sec
tmin, tmax = -.5, .5
# from subject 1
subject = 1
# for the hands vs. feet motor imagery task (runs 6, 10, 14)
runs = [6, 10, 14]
event_id = dict(hands=2, feet=3)

# load the data
raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)
fs = raw.info['sfreq']
raw.rename_channels(lambda x: x.strip('.'))
events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# epochs are the EEG data from tmin before the cue to tmax after the cue
full_epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=(tmin,0), preload=True)
# target labels: imagining moving hands is coded 0, moving feet is coded 1
labels = full_epochs.events[:, -1] - 2

# crop to only data from t = 0
cropped_epochs = copy.deepcopy(full_epochs)
cropped_epochs.crop(0, None)

## Feature extraction

# We use a feature extraction and selection method very loosely based on 
# a different paper (Belal et al., 2018, doi=10.1016/j.neuroimage.2018.04.029), 
# but very much simplified to fit our toy problem. 
# For each trial, every channel's raw EEG signal consists of 81 datapoints. 
# That means there are 64 (channels) x 81 (datapoints) = 5184 datapoints 
# per trial. Instead of using all of those points we try to capture the 
# important parts of the EEG in a few computed features. 

# We compute 15 features per channel: 
# - 11 discrete wavelet transform (DWT) features 
#   (describing the EEG wave's shape over time)
# - 3 spectral power features (describing the 
#   amount of movement in different frequency 
#   bands)
# - 1 'time domain' feature (in this case the 
#   mean over the full raw EEG)

# The discrete wavelet transform can be thought of as a denoising 
# procedure: it returns a signal and a noise part. The first gives us 
# an approximation of the EEG signal in 44 points. 
# (https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html)
# We then compute a moving average over every 4 points, leaving us with 
# the 11 DWT features.

# The spectral power describes how much the EEG wave is moving for given 
# frequencies. (https://en.wikipedia.org/wiki/Spectral_density)
# In our case the three spectral power features are the power spectral 
# densities within the theta (3.5-7.5 Hz), alpha (7.5-13 Hz) and beta 
# (14+ Hz) ranges.
 
# For the time domain feature we just compute the mean over all 81 raw 
# EEG values.

# These features are computed for every channel. This gives us 
# 64 × 15 = 960 features in total for every trial. 
# The temporal (DWT & time domain) and spectral features are then
# separately coded into three levels (0, 1, 2) using quantile-based 
# discretization (see visualizations in notebook). After this, We have
# a list of numbers (just 0s, 1s and 2s) for each trial that very 
# crudely encodes some information about the spectral and temporal
# properties of the EEG signal.
# The discretization is needed for the next step: from the lists of 960
# numbers for each trial, we want to select only the best / most useful
# 10% to actually give as input to the models. This is done by ranking 
# the features according to joint mutual information (Yang & Moody, 1999;
# doi=10.1.1.41.4424).
# After obtaining this ranking, we just select the 10% top-ranked features,
# and end up with 96 features per trial to use as input to the MLP.

# shuffle data
idx = list(range(len(cropped_epochs)))
shuffle(idx)
labels_shuffle = [labels[i] for i in idx]

all_features= []
# loop over trials (45)
for i in idx:
    trial_features = []
    epoch = cropped_epochs[i].get_data()[0]
    
    # loop over channels (64)
    for c in range(epoch.shape[0]):
        # channel data
        ch = epoch[c]
        
        # discrete wavelet transform & moving average
        (a,b) = pywt.dwt(ch, 'db4')
        a_ds = list(np.mean(a.reshape(-1, 4), 1))
        
        # spectral power densities
        f, d = scipy.signal.periodogram(ch,fs)
        theta = d[2]
        alpha = d[5]
        beta = d[10]
        sf = list([theta,alpha,beta])
        
        # time domain (channel mean)
        td = [np.mean(ch)]
        a_td = a_ds + td
        
        # discretize 
        a_td_disc = pd.qcut(a_td, 3, [0,1,2])
        sf_disc = pd.qcut(sf, 3, [0,1,2])
        
        # add channel features together
        ch_features = list(a_td_disc) + list(sf_disc)
        
        # add to trial features
        trial_features += ch_features
        
    all_features.append(trial_features)
    
# select 96 best features based on mutual information
f_size = 96
vals = feature_selection.mutual_info_classif(all_features, labels_shuffle, 
                                             discrete_features=True)
ranking = np.argsort(vals)[::-1]
top = ranking[:f_size]
best_features = np.array([ [row[i] for i in top] for row in all_features])

# dictionary for corresponding trial numbers, selected features and labels
EEG_imagery = {'trial_numbers': np.array(idx), 
               'features': best_features, 
               'labels': np.array(labels_shuffle)}