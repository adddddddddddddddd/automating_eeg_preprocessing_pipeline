from mne_bids import (
    BIDSPath,
    find_matching_paths,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)
import logging
import os

logging.basicConfig(level=logging.INFO)

bids_path = BIDSPath(
    subject='001',  # Replace with subject ID (e.g., '001' to '088')
    task='eyesclosed',
    root='./datasets/ds004504',
    datatype='eeg'
)

logging.info("Finding matching paths...")
raw = read_raw_bids(bids_path=bids_path, verbose=False)

logging.info(f"\n{'='*60}")
logging.info(f"DATASET INFORMATION:")
logging.info(f"{'='*60}")
logging.info(f"File: {bids_path.basename}")
logging.info(f"Full path: {bids_path.fpath}")
logging.info(f"Subject: {bids_path.subject}")
logging.info(f"Task: {bids_path.task}")
logging.info(f"Channels: {len(raw.ch_names)} ({', '.join(raw.ch_names)})")
logging.info(f"Sampling rate: {raw.info['sfreq']} Hz")
logging.info(f"Original duration: {raw.times[-1]:.1f} seconds")
logging.info(f"{'='*60}\n")

raw.crop(tmax=90)  # in seconds (happens in-place)
raw.pick("all").load_data()
fig = raw.plot(duration=5, n_channels=30, scalings="auto", show_scrollbars=False)
fig.savefig('./images/raw_timeseries.jpg')
fig = raw.plot_psd(fmax=50)
fig.savefig('./images/psd_plot.jpg')
fig = raw.plot_sensors(show_names=True)
fig.savefig('./images/sensors.jpg')