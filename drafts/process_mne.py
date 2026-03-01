import mne
import numpy as np

import os

if __name__ == "__main__":

    root = mne.datasets.sample.data_path() / "MEG" / "sample"
    raw_file = root / "ernoise_raw.fif"
    raw = mne.io.read_raw_fif(raw_file, preload=False)

    events_file = root / "ernoise_raw-eve.fif"
    events = mne.read_events(events_file)

    raw.crop(tmax=90)  # in seconds (happens in-place)
    # discard events >90 seconds (not strictly necessary)
    events = events[events[:, 0] <= raw.last_samp]
    raw.pick("all").load_data()
    fig = raw.plot(duration=5, n_channels=30, scalings="auto", show_scrollbars=False)
    fig.savefig('./images/raw_timeseries.jpg')
    fig = raw.plot_psd(fmax=50)
    fig.savefig('./images/psd_plot.jpg')
    fig = raw.plot_sensors(show_names=True)
    fig.savefig('./images/sensors.jpg')