// ── Experiments / runs ──────────────────────────────────────
export const EXPERIMENTS = [
  { id: 'RUN-004', status: 'running', file: 'sub-01_task-rest_eeg.edf',  time: '14:32' },
  { id: 'RUN-003', status: 'done',    file: 'sub-01_task-motor_eeg.edf', time: '13:58' },
  { id: 'RUN-002', status: 'done',    file: 'sub-02_task-rest_eeg.edf',  time: '13:21' },
  { id: 'RUN-001', status: 'error',   file: 'sub-03_task-rest_eeg.edf',  time: '12:47' },
  { id: 'RUN-000', status: 'done',    file: 'sub-04_task-rest_eeg.edf',  time: '11:03' },
]

export const JOURNAL = [
  { id: 'RUN-004', status: 'running', time: '14:32', file: 'sub-01_task-rest_eeg.edf',  note: null },
  { id: 'RUN-003', status: 'done',    time: '13:58', file: 'sub-01_task-motor_eeg.edf', note: '2 ICA components removed. 289/291 epochs retained. Quality: high.' },
  { id: 'RUN-002', status: 'done',    time: '13:21', file: 'sub-02_task-rest_eeg.edf',  note: '5 ICA components removed. 201/310 epochs. Quality: moderate.' },
  { id: 'RUN-001', status: 'error',   time: '12:47', file: 'sub-03_task-rest_eeg.edf',  note: 'ICA failed to converge. Suspected disconnected channel. Skipped.' },
]

// ── Pipeline steps ───────────────────────────────────────────
export const PIPELINE_STEPS = [
  { name: 'File ingest',       detail: 'parsing .edf header' },
  { name: 'Quality check',     detail: 'impedance · SNR' },
  { name: 'Bandpass filter',   detail: '1–40 Hz · notch 50 Hz' },
  { name: 'ICA decomposition', detail: '64 components' },
  { name: 'Artifact removal',  detail: 'ocular · cardiac' },
  { name: 'Feature export',    detail: 'vision pipeline' },
]

// ── Agent log messages ───────────────────────────────────────
export const AGENT_THOUGHTS = [
  [0,     'Loading raw EEG data from sub-001_task-eyesclosed_eeg.set...'],
  [1200,  'Reading channel info. Found 19 channels, 299900 samples (599.8 sec @ 500 Hz).'],
  [2500,  '[INITIAL QC] Starting quality assessment. Analyzing signal integrity...'],
  [4000,  'Generating PSD plot for visual inspection. Computing power spectral density...'],
  [5500,  'Initial QC complete. No stages to skip. Proceeding with full pipeline.'],
  [7000,  '[BANDPASS FILTERING] Designing 0.5-40 Hz filter. Hamming window, 3301 samples.'],
  [8500,  'Applying bandpass filter to preserve delta/theta/alpha/beta bands.'],
  [10000, 'Filter applied. Removes slow drifts and high-frequency muscle artifacts.'],
  [11500, '[BAD CHANNEL DETECTION] Analyzing channel quality across frequency bands...'],
  [13000, 'Identified bad channel: Fp1 (flat line, no significant activity).'],
  [14500, 'Marking Fp1 for interpolation. Other 18 channels show normal waveforms.'],
  [16000, '[NOTCH FILTERING] Examining PSD for 50 Hz line noise...'],
  [17500, 'Detected narrow peak at 50 Hz across all channels. Notch filter recommended.'],
  [19000, 'Model decision: Apply notch at 50 Hz to all 19 channels.'],
  [20500, '[RESAMPLING] Downsampling from 500 Hz to 250 Hz for ICA computation...'],
  [22000, 'Polyphase resampling complete. Filtered 1-45 Hz. New length: 149950 samples.'],
  [23500, '[ICA FITTING] Fitting ICA using 18 components (excluding bad channels)...'],
  [25000, 'ICA decomposition complete in 1.1s. Ready for artifact discrimination.'],
  [26500, '[ICA DISCRIMINATION] Analyzing components for eye, heart, muscle artifacts...'],
  [28000, 'Component ICA001: Eye blinks (vertical). Frontal loading pattern detected.'],
  [29000, 'Component ICA002: Horizontal eye movement. Step function characteristic.'],
  [29500, 'Component ICA012: Cardiac artifact. QRS complex signature confirmed.'],
  [30000, 'Component ICA016: Muscle artifact. High-freq bursts near face/neck channels.'],
  [31500, 'Removing 4 artifact components (ICA001, ICA002, ICA012, ICA016).'],
  [33000, 'Reconstructing clean signal using 14 neural components...'],
  [34500, '[INTERPOLATION] Interpolating bad channel Fp1 using spherical spline...'],
  [36000, 'Channel interpolation complete. All 19 channels now active.'],
  [37500, '[FINAL QC] Generating post-processing PSD. Verifying artifact removal...'],
  [39000, 'Pipeline complete. Clean EEG ready for analysis. 599.8s @ 500 Hz.'],
]

// ── ICA component data ───────────────────────────────────────
export const ICA_DATA = [
  { id: 'IC001', type: 'ocular',  variance: 8.4, loading: 0.91, bad: true  },
  { id: 'IC002', type: 'ocular',  variance: 6.1, loading: 0.83, bad: true  },
  { id: 'IC003', type: 'clean',   variance: 4.2, loading: 0.31, bad: false },
  { id: 'IC004', type: 'clean',   variance: 3.8, loading: 0.22, bad: false },
  { id: 'IC005', type: 'clean',   variance: 3.1, loading: 0.18, bad: false },
  { id: 'IC006', type: 'clean',   variance: 2.9, loading: 0.14, bad: false },
  { id: 'IC023', type: 'cardiac', variance: 2.1, loading: 0.87, bad: true  },
  { id: 'IC024', type: 'clean',   variance: 1.8, loading: 0.12, bad: false },
]

// ── Sensor / impedance data ──────────────────────────────────
export const SENSOR_DATA = [
  { name: 'Fp1', imp: 3.2,  ok: true  },
  { name: 'Fp2', imp: 4.1,  ok: true  },
  { name: 'F3',  imp: 5.8,  ok: true  },
  { name: 'F4',  imp: 3.9,  ok: true  },
  { name: 'C3',  imp: 4.4,  ok: true  },
  { name: 'C4',  imp: 18.2, ok: false },
  { name: 'P3',  imp: 6.1,  ok: true  },
  { name: 'P4',  imp: 22.7, ok: false },
  { name: 'O1',  imp: 5.0,  ok: true  },
  { name: 'O2',  imp: 4.8,  ok: true  },
]

// ── EEG channels ─────────────────────────────────────────────
export const EEG_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']
export const ARTIFACT_CHANNELS = new Set([0, 1])

// ── Helpers ──────────────────────────────────────────────────
export function genSignal(n, freq, noise, artifact) {
  return Array.from({ length: n }, (_, i) => {
    const t = i / n
    const a = Math.sin(2 * Math.PI * freq * t) * 0.6
    const b = Math.sin(2 * Math.PI * 20  * t) * 0.18
    const art = artifact && t > 0.3 && t < 0.4
      ? Math.sin(2 * Math.PI * 2 * t) * 2.2 : 0
    return a + b + (Math.random() - 0.5) * noise + art
  })
}

export function genPSD(seed, noise = 0.15) {
  return Array.from({ length: 39 }, (_, i) => {
    const f = i + 1
    const alpha = f > 7 && f < 13 ? 0.7 : 0
    return Math.min(1, Math.max(0.05, 0.6 / f + alpha + (Math.random() - 0.5) * noise + seed))
  })
}