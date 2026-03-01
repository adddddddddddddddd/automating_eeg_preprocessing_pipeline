
import openneuro
import os
from mne_bids import BIDSPath, read_raw_bids

if not os.path.exists('./datasets/ds004504'):
    openneuro.download(
        dataset="ds004504",
        target_dir="./datasets/ds004504"
    )


bids_path = BIDSPath(
    subject='001',  # Replace with subject ID (e.g., '001' to '088')
    task='eyesclosed',
    root='./datasets/ds004504',
    datatype='eeg'
)
raw = read_raw_bids(bids_path, verbose=False)
raw.crop(tmax=90)  # in seconds (happens in-place)
events = raw.find_events()
events = events[events[:, 0] <= raw.last_samp]
raw.pick("all").load_data()
fig = raw.plot(duration=5, n_channels=30, scalings="auto", block=True, show_scrollbars=False)