# 1. Import Required Packages

import mne
import torch
import os
from pathlib import Path 
import logging
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mne.preprocessing import annotate_muscle_zscore
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_morlet
import math
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               corrmap)
matplotlib.use('Qt5Agg')
mne.set_log_level('warning')

# Set seed
#random.seed(42) 

# 2. Load Raw EEG Data

epochDuration = 1

study_epochs = {
    'td': [],
    'asd': []
}

label_ids = {
    "asd": 1,
    "td": 2
}


data_path = Path("data/")
raw_eeg_path = data_path / "raw_eeg"

# Assumes file is formatted as {ID}_{type}_{XXXHz}.{extension}. Example TD100_raw_512Hz.asc
def csvToRaw(file, fMax=40):
    print(f"Processing {file.stem}")
    data = pd.read_csv(file, sep='\t')

    try:
        data =  data.drop(['VEOG - LkE', 'HEOG - LkE', 'Unnamed: 34'], axis=1)
    except:
        print("No EOG Channels found")
    # Get Channels
    channels = list(data.columns)

    # Format Channel names
    f = lambda str: str.split("-")[0].replace(" ", "")
    channels = [f(x) for x in channels]
    channel_count = len(channels)

    # Load Data
    data = data.transpose()
    ch_types = np.full((channel_count), "eeg")
    sfreq = int(file.stem.split("_")[2].replace("Hz", ""))
    info = mne.create_info(ch_names = channels, sfreq = sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Format data date for annotations later
    raw.set_meas_date(0)
    raw.set_montage("standard_1020")

    # Convert from uV to V for MNE
    raw.apply_function(lambda x: x * 1e-6)

    # Mark bad data
    # Addressing this later now
    markMuscleArtifacts(raw, 2)

    filtered = raw.copy().filter(l_freq=1.0, h_freq=fMax)

    return raw, filtered

# Find bad spans of data using mne.preprocessing.annotate_muscle_zscore
def markMuscleArtifacts(raw, threshold, plot=False):
    #print("markMuscleArtifacts")
    threshold_muscle = threshold  # z-score
    annot_muscle, scores_muscle = annotate_muscle_zscore(
    raw, ch_type="eeg", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[0, 60])
    raw.set_annotations(annot_muscle)

    if plot:
        fig, ax = plt.subplots()
        start = 512 * 10
        end = 512 * 20
        ax.plot(raw.times[:end], scores_muscle[:end])
        ax.axhline(y=threshold_muscle, color='r')
        ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
        plt.show()

# Reject epochs based on maximum acceptable peak-to-peak amplitude 
# https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html#sphx-glr-auto-tutorials-preprocessing-20-rejecting-bad-data-py
def dropBadEpochs(epochs, plotLog=False):
    reject_criteria = dict(eeg=150e-6) # 150 µV
    flat_criteria = dict(eeg=1e-6) # 1 µV
    epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)
    if plotLog: epochs.plot_drop_log()

# Get ICs 
def runICA(epochs):
    #print("Running ICA")
    n_components = 0.99  # Should normally be higher, like 0.999!!
    method = 'picard'
    fit_params = dict(fastica_it=5)
    random_state = 42

    ica = mne.preprocessing.ICA(n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=random_state)

    ica.fit(epochs)
    return ica

# Create epoch data, get ICs, and add to study_epochs object
def addEpochs(data, raw, start, file_length_seconds, label, file=""):
    stop = start + epochDuration
    events = mne.make_fixed_length_events(data,  label_ids[label], start=start, stop=file_length_seconds, duration=epochDuration)
    epochs = mne.Epochs(data, events, tmin=0, tmax=epochDuration, event_id={label: label_ids[label]}, baseline=(0, 0), preload=True)
    #epochs.plot(title="before")
    dropBadEpochs(epochs)
   
    # Run ICA
    ica = runICA(epochs)
    eog_epochs = create_eog_epochs(raw, "Fp1")
    eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, "Fp2",  threshold=1.25)
    ica.exclude = eog_inds
    #print(eog_inds)

    # Get Clean epochs
    cleaned_epochs = ica.apply(epochs.copy())
    #fig = cleaned_epochs.compute_psd().plot()
    #fig.suptitle(file, fontsize=16)

    #cleaned_epochs.plot(title="after")
    study_epochs[label].append(cleaned_epochs)
    # update with cleaned epochs
    return epochs

# Process raw file. 
def process(file):
    raw, filtered = csvToRaw(file)
    #fig = filtered.compute_psd().plot()
    file_length = math.floor(len(filtered.times) / float(filtered.info['sfreq']))
    #print(f"{file_length}s file length")
    label = file.parent.stem
    epochs = addEpochs(filtered, raw, 1.0, file_length, label, file)  # first ten seconds
    return filtered, epochs


# Clean all data in the file paths added to data_path_list
def cleanData(dir):
    data_path_list = list(raw_eeg_path.glob(f"{dir}/*/*.asc"))
    count = 0

    for file in data_path_list:
        process(file)
        count += 1

    asd_concat_epochs = mne.concatenate_epochs(study_epochs['asd'])
    td_concat_epochs = mne.concatenate_epochs(study_epochs['td'])
    print(f"{count} files processed.")
    print(f"{len(study_epochs['asd'])} asd epoch objects")
    print(f"{len(study_epochs['td'])} td td objects")
    #asd_concat_epochs.plot()
    #td_concat_epochs.plot()
    asd_concat_epochs.save(Path('out_data') / f'{dir}_asd_concat_cleaned_1_40hz_epo.fif', overwrite=True)
    td_concat_epochs.save(Path('out_data') / f'{dir}_td_concat_cleaned_1_40hz_epo.fif', overwrite=True)
    train_all_epochs = mne.concatenate_epochs([asd_concat_epochs, td_concat_epochs])
    train_all_epochs.equalize_event_counts(train_all_epochs.event_id)
    train_all_epochs.save(Path('out_data') / f'{dir}_all_epo.fif')
    
    plt.show()
    
cleanData("train")
# processRandomFile()
# icaClean()


asd_epochs = mne.read_epochs(Path('out_data') / 'train_asd_concat_cleaned_1_40hz_epo.fif')
td_epochs = mne.read_epochs(Path('out_data') / 'train_td_concat_cleaned_1_40hz_epo.fif')

#fig = asd_epochs.compute_psd(fmax=60).plot(average=True)
#fig.suptitle("asd")





# 3. Additional Helper Functions

def icaClean():
    asd_epochs = mne.read_epochs(Path('out_data') / 'asd_concat_cleaned_1_40hz_epo.fif')
    td_epochs = mne.read_epochs(Path('out_data') / 'td_concat_cleaned_1_40hz_epo.fif')
    #asd_epochs.plot(title="ASD")
    #td_epochs.plot(title="TD")
    ica = runICA(asd_epochs, "ASD")
    ica.plot(title="ASD")

def checkPath(dir):
    # If the image folder doesn't exist, download it and prepare it... 
    if not data_path.is_dir():
        logging.error(f"{data_path} directory DOES NOT exists.")

def walkThroughDir(dir_path):
    checkPath(dir_path)
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} file(s) in '{dirpath}'.")

def createEvent(onset, label, sFreq):
    return [onset * sFreq, 0, label]


def getRandomFile(data_list):
    random_raw_eeg_path = random.choice(data_list)
    label = random_raw_eeg_path.parent.stem
    return random_raw_eeg_path, label

def processRandomFile():
    data_path_list = list(raw_eeg_path.glob("train/*/*.asc"))
    eeg_path, label = getRandomFile(data_path_list)
    filtered, epochs = process(eeg_path)
    #epochs.compute_psd().plot()

def visualizePower(minFreq, maxFreq, epochs, plotChannels=True, plotTopo=True, event=""):
    freqs = np.logspace(*np.log10([minFreq, maxFreq]), num=128)
    # freqs = np.arange(1, 32, 0.25)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=3, n_jobs=None)

    if plotChannels:
        for i in range(len(power.ch_names)):
            power.plot([i], baseline=(-0.5, 0), mode='logratio', title=(event + " " + power.ch_names[i]))


    topomap_kw = dict(ch_type='eeg', tmin=0, tmax=1.0, mode='logratio', show=False)
    plot_dict = dict(Delta=dict(fmin=0, fmax=4), Alpha=dict(fmin=7, fmax=12), Beta=dict(fmin=13, fmax=25))
    
    if plotTopo:
        power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
        fig, axes = plt.subplots(1, 3, figsize=(10, 7))
        for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
            power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw, show_names=True)
            ax.set_title(event + " " + title)
    
        fig.tight_layout()
        fig.show()
    return power


# 4. (Example) Load and Combine Existing Epochs

test_asd_epochs = mne.read_epochs(Path('out_data') / 'test_asd_concat_cleaned_1_40hz_epo.fif')
test_td_epochs = mne.read_epochs(Path('out_data') / 'test_td_concat_cleaned_1_40hz_epo.fif')
test_all_epochs = mne.concatenate_epochs([test_asd_epochs, test_td_epochs])

# Set if you want to equalize the event counts
test_all_epochs.equalize_event_counts(test_all_epochs.event_id)

test_all_epochs.save(Path('out_data') / 'test_all_cleaned_equal_1_40hz_epo.fif')


# 4. (Example) Create and Save Combined Raws and Process

def combineRaws():
    data_path_list = list(raw_eeg_path.glob("train/*/*.asc"))
    raws = {
        "td":[],
        "asd":[]
    }
    
    count = 0

    for file in data_path_list:
        raw, filtered = csvToRaw(file)
        file_length = math.floor(len(filtered.times) / float(filtered.info['sfreq']))
        raws[file.parent.stem].append(raw)
        print(f"{file_length}s file length - {file.parent.stem}")

    
    print(f"{len(raws['asd'])} ASD raws. ")
    print(f"{len(raws['td'])} td raws. ")

    combined_raw_asd = concatenate_raws(raws['asd'])
    combined_raw_asd.save(Path('out_data') / 'raw_asd_combined_eeg.fif', overwrite=True)
    #combined_raw_asd.plot()

    combined_raw_td = concatenate_raws(raws['td'])
    combined_raw_td.save(Path('out_data') / 'raw_td_combined_eeg.fif', overwrite=True)
    #combined_raw_td.plot()

#combineRaws()