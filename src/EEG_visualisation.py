import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import label
from scipy.interpolate import griddata
import mne
import numpy as np
import mne
from mne.channels import make_standard_montage
from mne import create_info
from mne.viz import plot_topomap




def EEG_ERP():
    """
    EEG basic visualization
        - Event related potential (ERP)
        - Topography


    **Event related potential (ERP):**
        Is simply the mean value of the different trials


        Example of ERP for one channel:

        
        .. image:: _static/images/EEG_ERP.png


        Example of ERP for all channel using MNE library: It give directly additional information on topography
        
        
        .. image:: _static/images/EEG_ERP_MNE.png


    **Topography:**
        .. image:: _static/images/EEG_ERP_topography.png


    """
    # Charger les données EEG
    EEG = loadmat(r"SignalProcessing_on_neural_data\uANTS_intro_MATLABfiles\sampleEEGdata.mat")

    data = EEG['EEG']['data'][0][0]
    chanlocs = EEG['EEG']['chanlocs'][0][0][0]
    times = EEG['EEG']['times'][0][0][0]
    srate = EEG['EEG']['srate']
    ntrials = EEG['EEG']['trials']

    chan_name_arr = np.array([channel[0] for channel in chanlocs["labels"]])
    x = [channel[0][0] for channel in chanlocs["X"]]
    y = [channel[0][0]  for channel in chanlocs["Y"]]
    z = [channel[0][0]  for channel in chanlocs["Z"]]
    position = np.vstack([x, y, z]).T

    data.shape #(64, 640, 99) electrode, timepoints, trial
    data = transposed_array = np.transpose(data, (2, 0, 1))
    data.shape #(99, 64, 640 )trial, electrode, timepoints


    # Calculer l'ERP de chaque canal
    erp = np.mean(data, axis=0)

    # Tracer l'ERP pour un canal spécifique
    chan2plot = 'FCz'
    chan_index = np.where(chan_name_arr == chan2plot)[0]

    if len(chan_index)>0:
        chan_index = chan_index[0]
        plt.figure(figsize=(8, 4))
        plt.title("ERP")
        plt.plot(times, erp[chan_index, :], linewidth=2)
        plt.xlabel('Temps (ms)')
        plt.ylabel('Activité (\muV)')
        plt.xlim([-400, 1200])
        plt.show()

    # Tracer les cartes topographiques
    time2plot = 300  # en ms
    tidx = np.argmin(np.abs(times - time2plot))

    info = mne.create_info(ch_names=list(chan_name_arr), sfreq=srate, ch_types='eeg')
    evoked_array = mne.EvokedArray(erp, info)
    evoked_array._set_channel_positions(position, chan_name_arr)


    #plot the ERP of every channel
    evoked_array.plot()

    #plot the ERP topography
    # times = np.arange(0.01, 0.05, 0.1)
    # evoked_array.plot_topomap(times)
    evoked_array.plot_topomap()

    print('ja')
    epoch = mne.EpochsArray(data[:,:,:], info)
    fig, ax = plot_topomap(data, pos=info, names=info.ch_names)
    # # Afficher la figure
    # mne.viz.tight_layout()
    # mne.viz.show()





    n_sensor = 10
    n_trial = 5

    # npoints = 100000

    fs = 1000
    t = np.arange(0, 2, 1/fs)
    npoints = len(t)
    # t = np.linspace(0, 100, npoints)
    f_high = 500
    f_low = 100
    False

    data = np.random.rand(10, npoints)
    data = np.random.rand(5, 10, npoints)

    for i in range(0, n_sensor):
        for j in range(0, n_trial):
            sig_high = np.sin(2 * np.pi * t * f_high * np.random.rand(npoints))*0.1
            sig_medium = abs(np.sin(2 * np.pi * t * f_low)*(np.random.rand(npoints)>0.9)*1)
            data[j, i,:] = sig_high + sig_medium
            data[j, i,:] /= max(abs(data[j, i,:]))


    # Créer un objet d'info fictif pour décrire les propriétés des capteurs
    info = mne.create_info(ch_names=['sensor_{}'.format(i) for i in range(10)],
                        sfreq=fs, ch_types='eeg')

    # Créer un objet mne.EpochsArray en utilisant les données, l'info et les événements (peut être vide)
    epochs = mne.EpochsArray(data, info)
    epochs.plot( show_scrollbars=False, events=True)
    epochs.average().plot()




n_channels = 32
sampling_freq = 200  


times = np.linspace(0, 1, sampling_freq, endpoint=False)
sine = np.sin(20 * np.pi * times)
cosine = np.cos(10 * np.pi * times)
data = np.array([sine, cosine])

info = mne.create_info(
    ch_names=["10 Hz sine", "5 Hz cosine"], ch_types=["misc"] * 2, sfreq=sampling_freq
)

simulated_raw = mne.io.RawArray(data, info)
simulated_raw.plot(show_scrollbars=False, show_scalebars=False)

data = np.array(
    [
        [0.2 * sine, 1.0 * cosine],
        [0.4 * sine, 0.8 * cosine],
        [0.6 * sine, 0.6 * cosine],
        [0.8 * sine, 0.4 * cosine],
        [1.0 * sine, 0.2 * cosine],
    ]
)

simulated_epochs = mne.EpochsArray(data, info)
simulated_epochs.plot(picks="misc", show_scrollbars=False, events=True)


def example_mne_evoked_plot_topomap():
    """
    Example of Topography visualization using MNE library

    .. image:: _static/images/topography_mne_example.png
    """
    from mne.datasets import sample
    from mne import read_evokeds

    # print(__doc__)

    path = sample.data_path()
    fname = path / "MEG" / "sample" / "sample_audvis-ave.fif"

    # load evoked corresponding to a specific condition
    # from the fif file and subtract baseline
    condition = "Left Auditory"
    evoked = read_evokeds(fname, condition=condition, baseline=(None, 0))

    times = np.arange(0.05, 0.151, 0.02)
    evoked.plot()
    evoked.plot_topomap(times, ch_type="mag")


# # Charger les données CSD V1
# csd_data = loadmat('v1_laminar.mat', squeeze_me=True)
# csd = csd_data['csd']
# timevec = csd_data['timevec']

# # Tracer l'ERP du canal 7
# plt.figure()
# plt.plot(timevec, np.mean(csd[6, :, :], axis=0))
# plt.axhline(0, color='k', linestyle='--')
# plt.axvline(0, color='k', linestyle='--')
# plt.axvline(0.5, color='k', linestyle='--')
# plt.xlabel('Temps (s)')
# plt.ylabel('Activité (\muV)')
# plt.xlim([-0.1, 1.4])
# plt.show()

# # Tracer l'image de profondeur par temps de l'ERP
# plt.figure()
# plt.contourf(timevec, np.arange(1, 17), np.mean(csd, axis=2), 40, cmap='viridis', alpha=0.7)
# plt.xlabel('Temps (sec.)')
# plt.ylabel('Profondeur corticale')
# plt.xlim([0, 1.3])
# plt.colorbar()
# plt.show()


if __name__ == "__main__":
    example_mne_evoked_plot_topomap()
    EEG_ERP()