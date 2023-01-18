import mne
from pathlib import Path 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import preprocess
import pandas as pd

import seaborn as sns
sns.set_theme()

#%matplotlib
#tips = sns.load_dataset("tips")
#print(tips)

matplotlib.use('Qt5Agg')
mne.set_log_level('warning')

# https://mne.discourse.group/t/psd-multitaper-output-conversion-to-db-welch/4445
def scaleEEGPower(powerArray):
    powerArray = powerArray * 1e6**2 
    powerArray = (10 * np.log10(powerArray))
    return powerArray

def extractFetures(signal):
    delta = np.mean(signal[:4])
    theta = np.mean(signal[4:8])
    alpha = np.mean(signal[8:12])
    beta = np.mean(signal[13:30])
    gamma = np.mean(signal[30:40])
    return [alpha]

def getPSD(epochs, fmax, condition):
    mean_data = epochs.compute_psd(fmax=fmax).get_data().mean(axis=0)
    print(mean_data.shape)
    condition_arr = np.array([np.full((mean_data.shape[1]), condition) for freq in range(mean_data.shape[0])]).reshape(1, -1).squeeze()
    channel_ids = np.array([np.full((mean_data.shape[1]), channel_name) for channel_name in epochs.ch_names]).reshape(1, -1).squeeze()
    freq_ids = np.array([np.arange(mean_data.shape[1]) for channel_name in epochs.ch_names]).reshape(1, -1).squeeze()
    mean_data = mean_data.reshape(1, -1).squeeze()
    mean_data = scaleEEGPower(mean_data)
    return np.stack((channel_ids, condition_arr, mean_data, freq_ids), axis=1)

def main():
    all_epochs, idx_asd, idx_td, np_all_epochs, y = preprocess.getInput('train_all_epo.fif')

    #all_epochs.compute_psd(fmax=30).plot()

    asd_stacked = getPSD(all_epochs['asd'], 30, "ASD")
    td_stacked = getPSD(all_epochs['td'], 30, "TD")
    all_stacked = np.concatenate((asd_stacked, td_stacked))
    df = pd.DataFrame(all_stacked, columns = ['Channel','Group','Power', "Frequency"])
    df['Power'] = df['Power'].astype('float')
    ax = sns.lineplot(data=df, x="Frequency", y="Power", hue="Group")
    ax.set(ylabel=' $µV^2$/Hz (dB)')
    plt.xticks(fontsize=8)
    plt.show()
    #print(df)

#main()



def scaleEEGPower(powerArray):
    powerArray = powerArray * 1e6**2 
    powerArray = (10 * np.log10(powerArray))
    return powerArray

def extractFetures(signal):
    delta = np.mean(signal[:4])
    theta = np.mean(signal[4:8])
    alpha = np.mean(signal[8:12])
    beta = np.mean(signal[13:30])
    gamma = np.mean(signal[30:40])
    return [alpha]

def main():
    all_epochs, idx_asd, idx_td, np_all_epochs, y = preprocess.getInput('train_all_epo.fif')
    X_2d = preprocess.getProcessedInput(1, 40, np_all_epochs, all_epochs, extractFetures) 
    mean_channel_power = np.mean(X_2d, axis=2).T.flatten()
    mean_channel_power = scaleEEGPower(mean_channel_power)
    print(mean_channel_power.shape, "mean_channel_power", mean_channel_power)
    channel_id = np.array([np.full((np_all_epochs.shape[0]), channel_name) for channel_name in all_epochs.ch_names]).flatten()
    event_ids = np.array([all_epochs.events[:, 2] for channel in all_epochs.ch_names]).flatten()
    event_ids = np.where(event_ids == 1, "ASD", "TD")
    print(channel_id.shape, "channel_id", channel_id)
    print(event_ids.shape, "event_ids", event_ids)
    reformatted_data = np.stack((channel_id, event_ids, mean_channel_power), axis=1)
    df = pd.DataFrame(reformatted_data, columns = ['Channel','Group','Power'])
    df['Power'] = df['Power'].astype('float')
    select_channels = ['Fp1', 'Fp2', 'O1', "O2"]
    ch = df[df['Channel'].isin(select_channels)]
    ax = sns.swarmplot(data=ch, x="Group", y="Power", hue="Channel")
    ax.set_title("Alpha Power")
    plt.show()

main()


# TODO
- Create ML model base on min and max power values


#asd_psd.plot()
#asd_psd = np.mean(psd, axis=0)
#asd_psd = np.mean(psd, axis=0)
#asd_psd = scaleEEGPower(psd)
#asd_psd = asd_psd * 1e6**2 
#asd_psd = (10 * np.log10(asd_psd))
#asd_psd = np.log10(np.absolute(asd_psd))

#i = np.arange(len(asd_psd))
#np_df = np.stack((asd_psd, i), axis=1)
#df = pd.DataFrame(np_df, columns = ['Power','Freq'])
#sns.lineplot(data=df, x="Freq", y="Power")


#print(i.shape)
#print(np_df.shape)
#print(np_df[:3,:])
#print(df)

#all_epochs["td"].compute_psd(fmax=30).plot()
#print(X_2d.shape, all_epochs.events.shape)
  

#np_all_epochs.shape[0]


#print("epochs", np_all_epochs.shape[0], all_epochs.events.shape, channel_id.shape, event_ids.shape, mean_channel_power.shape)
#print(channel_id.shape, channel_id[:5])
#print(event_ids.shape, event_ids[:5])
#print(mean_channel_power.T.shape, mean_channel_power[:5])


channel_id = channel_id.flatten()
event_ids = event_ids.flatten()
event_ids = np.where(event_ids == 1, "ASD", "TD")
mean_channel_power = mean_channel_power.T.flatten()
mean_channel_power = np.square(10 * np.absolute(np.log10(mean_channel_power))) / 512
#print(type(mean_channel_power))

#print(channel_id.shape, channel_id)
#print(event_ids.shape, event_ids)
#print(mean_channel_power.shape, mean_channel_power)

reformatted_data = np.stack((channel_id, event_ids, mean_channel_power), axis=1)
df = pd.DataFrame(reformatted_data, columns = ['Channel','Condition','Power'])
df['Power'] = df['Power'].astype('float')
select_channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6']
ch = df[df['Channel'].isin(select_channels)]
#print(ch.head)
#sns.swarmplot(data=ch, x="Channel", y="Power", hue="Condition")
#print(all_epochs.ch_names)
#sns.barplot(data=ch, x="Channel", y="Power",  hue="Condition")
plt.show()