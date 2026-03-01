"""
Helper utilities for EEG data processing and plot generation.
These utilities help implement the TBC (to-be-constructed) agents.
"""
import os
import base64
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import logging

import mne
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ============================================================================
# PLOT GENERATION UTILITIES
# ============================================================================

def generate_eeg_plot(raw: mne.io.Raw, duration: float = 10.0, n_channels: int = 20) -> plt.Figure:
    """
    Generate an EEG time series plot.
    
    Args:
        raw: MNE Raw object
        duration: Duration to plot in seconds
        n_channels: Number of channels to display
        
    Returns:
        Matplotlib figure
    """
    fig = raw.plot(
        duration=duration,
        n_channels=n_channels,
        scalings='auto',
        show=False,
        block=False
    )
    return fig


def generate_power_spectrum_plot(raw: mne.io.Raw, fmin: float = 0.5, fmax: float = 100.0) -> plt.Figure:
    """
    Generate a power spectral density plot.
    
    Args:
        raw: MNE Raw object
        fmin: Minimum frequency
        fmax: Maximum frequency
        
    Returns:
        Matplotlib figure
    """
    fig = raw.compute_psd(fmin=fmin, fmax=fmax).plot(show=False)
    return fig


def generate_ica_components_plot(raw: mne.io.Raw, ica: mne.preprocessing.ICA) -> plt.Figure:
    """
    Generate ICA components visualization.
    
    Args:
        raw: MNE Raw object
        ica: Fitted ICA object
        
    Returns:
        Matplotlib figure
    """
    fig = ica.plot_components(show=False)
    return fig


def generate_ica_sources_plot(raw: mne.io.Raw, ica: mne.preprocessing.ICA, picks: Optional[list] = None) -> plt.Figure:
    """
    Generate ICA source time courses plot.
    
    Args:
        raw: MNE Raw object
        ica: Fitted ICA object
        picks: Which components to plot (default: all)
        
    Returns:
        Matplotlib figure
    """
    fig = ica.plot_sources(raw, picks=picks, show=False)
    return fig


def generate_channel_comparison_plot(
    raw_before: mne.io.Raw, 
    raw_after: mne.io.Raw,
    duration: float = 10.0
) -> plt.Figure:
    """
    Generate side-by-side comparison of EEG before and after processing.
    
    Args:
        raw_before: Raw data before processing
        raw_after: Raw data after processing
        duration: Duration to display
        
    Returns:
        Matplotlib figure with subplots
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    raw_before.plot(
        duration=duration,
        n_channels=10,
        scalings='auto',
        show=False,
        ax=axes[0]
    )
    axes[0].set_title("Before Processing")
    
    raw_after.plot(
        duration=duration,
        n_channels=10,
        scalings='auto',
        show=False,
        ax=axes[1]
    )
    axes[1].set_title("After Processing")
    
    plt.tight_layout()
    return fig


# ============================================================================
# IMAGE HANDLING UTILITIES
# ============================================================================

def figure_to_base64(fig: plt.Figure, format: str = 'png', dpi: int = 100) -> str:
    """
    Convert matplotlib figure to base64 encoded string.
    
    Args:
        fig: Matplotlib figure
        format: Image format (png, jpg, etc.)
        dpi: Resolution
        
    Returns:
        Base64 encoded image string with data URL prefix
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    # Return data URL format that can be used directly in HTML/APIs
    return f"data:image/{format};base64,{image_base64}"


def save_figure_temp(fig: plt.Figure, prefix: str = "eeg_plot", format: str = 'png') -> str:
    """
    Save figure to temporary file and return path.
    
    Args:
        fig: Matplotlib figure
        prefix: Filename prefix
        format: Image format
        
    Returns:
        Path to saved file
    """
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"{prefix}_{os.getpid()}.{format}")
    
    fig.savefig(temp_file, format=format, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved figure to {temp_file}")
    return temp_file


def upload_to_image_service(fig: plt.Figure, service: str = "imgur") -> str:
    """
    Upload figure to image hosting service and return URL.
    
    Args:
        fig: Matplotlib figure
        service: Image hosting service (imgur, imgbb, etc.)
        
    Returns:
        Public URL to the uploaded image
    """
    # TODO: Implement actual upload logic based on service
    # For now, convert to base64 (can be used directly with Mistral)
    logger.warning("Image upload not implemented, using base64 encoding")
    return figure_to_base64(fig)


# ============================================================================
# EEG DATA LOADING AND SAVING
# ============================================================================

def load_eeg_data(file_path: str, preload: bool = True) -> mne.io.Raw:
    """
    Load EEG data from file (supports multiple formats).
    
    Args:
        file_path: Path to EEG file
        preload: Whether to load data into memory
        
    Returns:
        MNE Raw object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"EEG file not found: {file_path}")
    
    # Detect format and load accordingly
    if file_path.suffix in ['.fif', '.fif.gz']:
        raw = mne.io.read_raw_fif(file_path, preload=preload)
    elif file_path.suffix == '.edf':
        raw = mne.io.read_raw_edf(file_path, preload=preload)
    elif file_path.suffix == '.bdf':
        raw = mne.io.read_raw_bdf(file_path, preload=preload)
    elif file_path.suffix == '.set':
        raw = mne.io.read_raw_eeglab(file_path, preload=preload)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded EEG data: {raw.info['nchan']} channels, {raw.times[-1]:.2f}s duration")
    return raw


def save_eeg_data(raw: mne.io.Raw, output_path: str, overwrite: bool = True):
    """
    Save EEG data to file.
    
    Args:
        raw: MNE Raw object
        output_path: Path to save file
        overwrite: Whether to overwrite existing file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    raw.save(output_path, overwrite=overwrite)
    logger.info(f"Saved EEG data to {output_path}")


# ============================================================================
# FILTERING UTILITIES
# ============================================================================

def apply_notch_filter(
    raw: mne.io.Raw,
    freqs: float = 60.0,
    notch_widths: Optional[float] = None,
    method: str = 'fir'
) -> mne.io.Raw:
    """
    Apply notch filter to remove line noise.
    
    Args:
        raw: MNE Raw object (modified in-place)
        freqs: Frequency to notch out (50 or 60 Hz)
        notch_widths: Width of the notch
        method: Filter method ('fir' or 'iir')
        
    Returns:
        Filtered Raw object
    """
    logger.info(f"Applying notch filter at {freqs} Hz")
    raw.notch_filter(freqs=freqs, notch_widths=notch_widths, method=method)
    return raw


def apply_bandpass_filter(
    raw: mne.io.Raw,
    l_freq: float = 0.5,
    h_freq: float = 50.0,
    method: str = 'fir'
) -> mne.io.Raw:
    """
    Apply bandpass filter.
    
    Args:
        raw: MNE Raw object (modified in-place)
        l_freq: Low frequency cutoff
        h_freq: High frequency cutoff
        method: Filter method
        
    Returns:
        Filtered Raw object
    """
    logger.info(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq, method=method)
    return raw


def apply_highpass_filter(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    method: str = 'fir'
) -> mne.io.Raw:
    """
    Apply highpass filter (useful for slow drift removal).
    
    Args:
        raw: MNE Raw object (modified in-place)
        l_freq: Low frequency cutoff
        method: Filter method
        
    Returns:
        Filtered Raw object
    """
    logger.info(f"Applying highpass filter: {l_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=None, method=method)
    return raw


# ============================================================================
# CHANNEL AND ICA UTILITIES
# ============================================================================

def remove_bad_channels(raw: mne.io.Raw, bad_channels: list[str]) -> mne.io.Raw:
    """
    Mark and drop bad channels.
    
    Args:
        raw: MNE Raw object (modified in-place)
        bad_channels: List of channel names to remove
        
    Returns:
        Raw object with channels removed
    """
    if not bad_channels:
        return raw
    
    logger.info(f"Removing {len(bad_channels)} bad channels: {bad_channels}")
    raw.info['bads'] = bad_channels
    raw.drop_channels(bad_channels)
    return raw


def interpolate_bad_channels(raw: mne.io.Raw, bad_channels: list[str], method: str = 'spline') -> mne.io.Raw:
    """
    Interpolate bad channels instead of removing them.
    
    Args:
        raw: MNE Raw object (modified in-place)
        bad_channels: List of channel names to interpolate
        method: Interpolation method
        
    Returns:
        Raw object with channels interpolated
    """
    if not bad_channels:
        return raw
    
    logger.info(f"Interpolating {len(bad_channels)} bad channels: {bad_channels}")
    raw.info['bads'] = bad_channels
    raw.interpolate_bads(reset_bads=True, mode=method)
    return raw


def fit_ica(
    raw: mne.io.Raw,
    n_components: Optional[int] = None,
    method: str = 'fastica',
    random_state: int = 42
) -> mne.preprocessing.ICA:
    """
    Fit ICA to raw data.
    
    Args:
        raw: MNE Raw object
        n_components: Number of ICA components (default: all)
        method: ICA algorithm ('fastica', 'infomax', 'picard')
        random_state: Random seed for reproducibility
        
    Returns:
        Fitted ICA object
    """
    logger.info(f"Fitting ICA with method={method}, n_components={n_components}")
    
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state
    )
    
    ica.fit(raw)
    logger.info(f"ICA fitted: {ica.n_components_} components")
    
    return ica


def apply_ica(raw: mne.io.Raw, ica: mne.preprocessing.ICA, exclude: list[int]) -> mne.io.Raw:
    """
    Remove ICA components from raw data.
    
    Args:
        raw: MNE Raw object (modified in-place)
        ica: Fitted ICA object
        exclude: List of component indices to remove
        
    Returns:
        Raw object with ICA components removed
    """
    if not exclude:
        logger.info("No ICA components to remove")
        return raw
    
    logger.info(f"Removing {len(exclude)} ICA components: {exclude}")
    ica.exclude = exclude
    ica.apply(raw)
    
    return raw


# ============================================================================
# VALIDATION METRICS
# ============================================================================

def calculate_snr(raw: mne.io.Raw, fmin: float = 0.5, fmax: float = 50.0) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Args:
        raw: MNE Raw object
        fmin: Minimum frequency for signal band
        fmax: Maximum frequency for signal band
        
    Returns:
        SNR value
    """
    psd = raw.compute_psd(fmin=fmin, fmax=fmax)
    signal_power = psd.get_data().mean()
    
    # Rough SNR estimation (can be improved)
    noise_band_psd = raw.compute_psd(fmin=50, fmax=100)
    noise_power = noise_band_psd.get_data().mean()
    
    snr = 10 * mne.fixes.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return snr


def calculate_channel_variance(raw: mne.io.Raw) -> dict:
    """
    Calculate variance for each channel.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        Dictionary mapping channel names to variance values
    """
    data = raw.get_data()
    variances = data.var(axis=1)
    
    return {ch_name: var for ch_name, var in zip(raw.ch_names, variances)}


def detect_line_noise_frequency(raw: mne.io.Raw) -> Tuple[float, float]:
    """
    Detect whether 50Hz or 60Hz line noise is present.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        Tuple of (frequency, power) for the dominant line noise
    """
    psd = raw.compute_psd(fmin=40, fmax=70)
    freqs = psd.freqs
    power = psd.get_data().mean(axis=0)
    
    # Check peaks around 50 and 60 Hz
    idx_50 = mne.fixes.argmin(abs(freqs - 50))
    idx_60 = mne.fixes.argmin(abs(freqs - 60))
    
    power_50 = power[idx_50]
    power_60 = power[idx_60]
    
    if power_50 > power_60:
        return 50.0, power_50
    else:
        return 60.0, power_60


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_complete_preprocessing_workflow(input_file: str, output_file: str):
    """
    Example showing complete preprocessing workflow using these utilities.
    This demonstrates how to implement the TBC agents.
    """
    # Load data
    raw = load_eeg_data(input_file)
    
    # Generate initial plot
    fig = generate_eeg_plot(raw)
    plot_url = upload_to_image_service(fig)
    # Send plot_url to LLM for analysis...
    
    # Detect and apply notch filter
    line_freq, power = detect_line_noise_frequency(raw)
    if power > threshold:  # Define your threshold
        raw = apply_notch_filter(raw, freqs=line_freq)
    
    # Bad channel detection (placeholder for LLM result)
    bad_channels = ['Fp1', 'F7']  # From LLM analysis
    raw = remove_bad_channels(raw, bad_channels)
    
    # Apply highpass for slow drift
    raw = apply_highpass_filter(raw, l_freq=1.0)
    
    # ICA
    ica = fit_ica(raw, n_components=0.95, method='fastica')
    
    # Generate ICA plots
    ica_fig = generate_ica_components_plot(raw, ica)
    ica_plot_url = upload_to_image_service(ica_fig)
    # Send to LLM for component selection...
    
    # Apply ICA (placeholder for LLM result)
    exclude_components = [0, 1, 5]  # From LLM analysis
    raw = apply_ica(raw, ica, exclude=exclude_components)
    
    # Interpolate bad channels
    raw = interpolate_bad_channels(raw, bad_channels)
    
    # Calculate validation metrics
    snr = calculate_snr(raw)
    logger.info(f"Final SNR: {snr:.2f} dB")
    
    # Save result
    save_eeg_data(raw, output_file)
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("EEG Processing Utilities Loaded")
    print("\nAvailable functions:")
    print("  - Plot generation: generate_eeg_plot, generate_power_spectrum_plot, generate_ica_components_plot")
    print("  - Image handling: figure_to_base64, upload_to_image_service")
    print("  - Data I/O: load_eeg_data, save_eeg_data")
    print("  - Filtering: apply_notch_filter, apply_bandpass_filter, apply_highpass_filter")
    print("  - Channel utilities: remove_bad_channels, interpolate_bad_channels")
    print("  - ICA: fit_ica, apply_ica")
    print("  - Validation: calculate_snr, detect_line_noise_frequency")
