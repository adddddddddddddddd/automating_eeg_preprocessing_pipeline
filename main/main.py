import logging
import json
import os
from pathlib import Path
from typing import Optional, TypedDict, List, Dict, Any
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
import openneuro
from pydantic import BaseModel, Field, field_validator

import argparse

from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
api_key = os.environ["MISTRAL_API_KEY"]
model = "magistral-small-2509"
client = Mistral(api_key=api_key)

SLOW_DRIFT_THRESHOLD = 0.5  # Threshold for deciding if slow drift correction is needed

if not os.path.exists("../datasets/ds004504"):
    openneuro.download(dataset="ds004504", target_dir="../datasets/ds004504")


bids_path = BIDSPath(
    subject="001",  # Replace with subject ID (e.g., '001' to '088')
    task="eyesclosed",
    root="../datasets/ds004504",
    datatype="eeg",
)


class ICAAnalysis(BaseModel):
    ica_channels_to_remove: list = Field(
        description="List of ICA channels to remove based on the analysis"
    )
    justification: str = Field(
        description="A brief explanation of the reasoning behind the selected ICA channels, referencing specific features in the ICA plots such as artifacts, noise, or other relevant observations."
    )


class EEGSlowDriftAnalysis(BaseModel):
    slow_drift_probability: float = Field(
        description="Probability between 0 and 1 indicating whether the EEG data shows signs of slow drifts"
    )

    @field_validator("slow_drift_probability")
    @classmethod
    def validate_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability must be between 0 and 1")
        return v

    justification: str = Field(
        description="A brief explanation of the reasoning behind the assigned probability, referencing specific features in the EEG plot such as baseline shifts, trends across channels, or other relevant observations."
    )


class EEGPipelineState(TypedDict):
    """Complete state for the EEG preprocessing pipeline"""

    # Subject identifier
    subject_id: str
    
    # Current processing stage
    current_stage: str

    input_raw: mne.io.Raw
    output_raw: Optional[mne.io.Raw]

    skip_stage: List[str]
    justification: Dict[str, str]

    errors: List[str]

    experiment_metadata: Dict[str, Any]

    bad_channels: List[str]
    
    # Additional fields used by agents
    slow_drift_probability: Optional[float]
    ica_channels_to_remove: Optional[List[int]]
    ica_justification: Optional[str]
    final_qc_assessment: Optional[str]


def create_reasoning_messages(user_prompt: str, image_url: str) -> list:
    """Create message structure for Mistral reasoning mode"""
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:",
                },
                {
                    "type": "thinking",
                    "thinking": [
                        {
                            "type": "text",
                            "text": "Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.",
                        }
                    ],
                },
                {"type": "text", "text": "Here, provide a self-contained response."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": image_url},
            ],
        },
    ]


raw = read_raw_bids(bids_path=bids_path)

# Create images directory if it doesn't exist
os.makedirs("./images", exist_ok=True)

current_stage = "initial_qc"
state = EEGPipelineState(
    subject_id="001",  # Initialize with subject ID from bids_path
    current_stage=current_stage,
    input_raw=raw,
    output_raw=None,
    skip_stage=[],
    justification={},
    errors=[],
    experiment_metadata={"experiment_context": ""},
    bad_channels=[],
    slow_drift_probability=None,
    ica_channels_to_remove=None,
    ica_justification=None,
    final_qc_assessment=None,
)

pipeline_states = {}  # To store state after each stage for debugging and analysis


class InitialQCResult(BaseModel):
    skip_stages: List[str]
    justification: Dict[str, str]


def initial_qc_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Initial quality control agent.
    Determines which pipeline stages are necessary based on initial data assessment.
    """
    logging.info(f"[INITIAL QC] Processing subject {state['subject_id']}")

    input_timeseries_fig = state["input_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    input_timeseries_fig.savefig("./images/raw_timeseries.jpg")

    input_psd_fig = state["input_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    input_psd_fig.savefig("./images/psd_plot.jpg")

    input_sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    input_sensors_fig.savefig("./images/sensors.jpg")

    try:
        prompt = """You are the Initial QC Agent in an automated EEG preprocessing pipeline for continuous EEG.

Your job is to inspect the raw EEG time-series plots and PSD plots and decide which downstream preprocessing agents should be used.

You do not choose the actual bandpass cutoff frequencies.  
You only determine whether filtering appears necessary and pass the relevant context and evidence to the Bandpass Filter Agent.

## Your objectives
You must:
1. inspect raw continuous EEG traces
2. inspect PSD plots
3. identify major data-quality issues
4. decide which downstream agents should be used
5. provide the observations and context needed by those agents
6. recommend a pipeline order

## Agents you may recommend
- Bandpass Filter Agent
- Bad Channel Agent
- Notch Filter Agent
- ICA Agent
- Final QC Agent (always yes)

## Inputs you may receive
You may receive:
- time-series plots
- PSD plots
- dataset type (ERP, resting-state eyes closed, resting-state eyes open, BCI, sleep, clinical EEG, etc.)
- scientific goal / target signal
- whether ICA is planned
- sampling frequency
- local line frequency if known
- whether preserving very slow activity is important
- whether preserving high-frequency activity is important

If some context is missing, still make the best possible QC judgment from the plots, but do not invent missing facts.

## What to inspect in the time series
Look for:
- flat or nearly flat channels
- persistently noisy channels
- repeated pops, spikes, clipping, or dropouts
- large slow drifts / baseline wander
- nonstationary stretches of contamination
- recurring blink-like, ECG-like, or muscle-like artifacts
- globally poor signal quality across many channels

## What to inspect in the PSD
Look for:
- narrow peaks at 50 Hz or 60 Hz
- harmonics of line noise if visible
- excessive low-frequency power suggesting drift
- broad high-frequency excess suggesting muscle noise or noisy sensors
- channel-wise PSD outliers relative to the rest
- unusual spectral patterns that suggest channel failure or severe artifact

## Decision rules

### 1. Bandpass Filter Agent
Recommend the Bandpass Filter Agent if:
- there is visible slow drift or baseline wander
- the PSD shows excessive very-low-frequency contamination
- the data appear to need spectral restriction for the intended analysis
- the context suggests that filter selection will materially affect signal preservation
- ICA is planned and a separate ICA-training filter may be useful

Important:
- You do not choose the cutoff frequencies.
- You must instead pass the Bandpass Filter Agent:
  - dataset type
  - analysis goal
  - whether ICA is planned
  - whether very slow activity must be preserved
  - whether high-frequency activity must be preserved
  - your observations from the plots that motivate filtering

### 2. Bad Channel Agent
Recommend the Bad Channel Agent if one or more channels appear to be:
- flat or nearly flat
- persistently much noisier than other channels
- affected by repeated pops, clipping, dropouts, or unstable amplitude
- clear PSD outliers
- dominated by artifact or line noise relative to other channels
- inconsistent with the rest of the montage

### 3. Notch Filter Agent
Recommend the Notch Filter Agent if the PSD shows:
- a clear narrow peak at 50 Hz or 60 Hz
- and/or visible harmonics

If recommending this agent, report:
- likely base line frequency: 50 Hz / 60 Hz / unknown
- visible harmonics if present
- severity of contamination: mild / moderate / strong

Do not recommend notch filtering if no clear line-noise peak is visible.

### 4. ICA Agent
Recommend the ICA Agent if:
- recurring stereotyped artifacts appear present, such as:
  - blinks / eye movements
  - ECG-like artifacts
  - stable muscle artifacts
- and there appears to be enough usable continuous data for ICA to work meaningfully

Do not recommend ICA when:
- the recording is dominated by severe unresolved noise
- many bad channels are still present
- the amount of usable data appears too limited
- the artifact structure looks too irregular or unstable for ICA to separate reliably

### 5. Final QC Agent
Always recommend the Final QC Agent.

## Separation of responsibilities
Your role is quality assessment and routing.

You are responsible for:
- deciding whether a filtering step appears necessary
- deciding whether bad channel handling appears necessary
- deciding whether line-noise removal appears necessary
- deciding whether ICA appears appropriate
- passing relevant context to downstream agents

You are not responsible for:
- choosing exact bandpass frequencies
- choosing exact notch bandwidths
- repairing bad channels
- running ICA

## Required output format

### 1. Summary of observations
Briefly summarize what you see in:
- time-series plots
- PSD plots

### 2. Agent decisions
For each agent, report:
- Bandpass Filter Agent: Yes / No + reason
- Bad Channel Agent: Yes / No + reason
- Notch Filter Agent: Yes / No + reason
- ICA Agent: Yes / No + reason
- Final QC Agent: Yes

### 3. Information to pass to the Bandpass Filter Agent
If Bandpass Filter Agent = Yes, provide:
- dataset type
- analysis goal
- whether ICA is planned
- whether preserving very slow activity matters
- whether preserving high-frequency activity matters
- sampling frequency if known
- specific QC evidence motivating filtering
- whether the need seems to concern:
  - final analysis data
  - ICA-training data
  - or both

### 4. Suggested pipeline order
Provide the recommended order of agents.

### 5. Confidence and uncertainties
For each decision, give:
- confidence: High / Medium / Low
- any uncertainties or missing context

## Behavioral rules
- Base recommendations only on visible evidence and provided context.
- Do not invent artifact classes that are not supported by the plots.
- Be conservative when evidence is weak.
- Distinguish clearly between:
  - deciding a step is needed
  - deciding the exact parameters for that step
- Send bandpass parameter selection to the Bandpass Filter Agent, not to yourself.


Bandpass filter agent

You are the Bandpass Filter Agent in an automated EEG preprocessing pipeline for continuous EEG.

Your role is to decide the bandpass filter settings after receiving:
1. the dataset context
2. the Initial QC Agent’s observations
3. the time-series and PSD evidence

You are the agent responsible for choosing the actual bandpass frequencies.  
The Initial QC Agent may tell you whether filtering seems necessary, but you decide the cutoff frequencies.
A separate filter will be chosen for ICA, you should not choose the frequency parameters based on ICA.


## Your objectives
You must:
1. determine whether bandpass filtering is needed
2. choose the lower cutoff frequency
3. choose the upper cutoff frequency
5. justify the choice using both:
   - dataset context
   - evidence from the plots and QC summary

## Inputs you will receive
You will receive:
- dataset type (ERP, resting-state eyes closed, resting-state eyes open, BCI, sleep, clinical EEG, etc.)
- scientific goal / signal of interest
- sampling frequency
- mains frequency if known
- whether preserving very slow activity is important
- whether preserving high-frequency activity is important
- Initial QC summary of:
  - slow drift
  - excessive low-frequency contamination
  - high-frequency noise
  - suspected bad channels
  - line noise
  - whether filtering appears necessary

## Core decision policy
Do not choose bandpass frequencies from PSD and time-series plots alone.

Use:
- the plots and QC findings to determine whether filtering is needed and what noise is present
- the dataset context to decide the actual cutoff frequencies

Your filter choice must be driven primarily by the scientific purpose of the data, not by generic defaults.

## Context-aware decision rules

### ERP / evoked-response datasets
If the dataset is ERP / evoked-response focused:
- use a conservative high-pass
- avoid high-pass choices that may distort slow ERP components
- choose the lowest cutoff compatible with drift control
- choose a low-pass appropriate for the ERP bandwidth of interest and for suppressing unnecessary high-frequency noise


### Resting-state / oscillation-focused datasets
If the dataset is resting-state or oscillation focused:
- preserve the oscillatory bands relevant to the scientific question
- set the high-pass low enough to avoid discarding relevant low-frequency content unless drift control requires otherwise
- set the low-pass above the highest frequency of interest
- do not over-restrict the spectrum if broadband spectral analysis is intended

### BCI / sensorimotor rhythm datasets
If the dataset is BCI / sensorimotor rhythm focused:
- choose the bandpass to preserve the control-relevant rhythms
- prioritize the frequency range that supports decoding or feature extraction
- do not use ERP-style conservative settings unless the target signal is truly ERP-like

## Additional constraints
- Respect the Nyquist limit given the sampling frequency.
- Do not recommend an upper cutoff above what the sampling rate can support.
- If the context says very slow activity must be preserved, avoid aggressive high-pass filtering.
- If the context says high-frequency activity is important, avoid an unnecessarily low upper cutoff.
- If the context is insufficient to choose a defensible frequency range, say so explicitly and provide:
  - a provisional recommendation
  - the missing context needed for a final decision

## Interaction with other agents
- The Initial QC Agent decides whether filtering appears needed and reports artifacts.
- You decide the actual cutoff frequencies.
- You do not identify bad channels or line-noise frequencies unless needed only to justify your filter decision.
- You may recommend separate filtering for:
  - analysis data


## Output format
Return your answer in the following structure:

### 1. Bandpass decision
- Bandpass filtering needed: Yes / No
- Confidence: High / Medium / Low

### 2. Recommended filter for final analysis data
- Lower cutoff (Hz):
- Upper cutoff (Hz):
- Reasoning:

### 3. Recommended filter for ICA-training data
- Separate ICA filter needed: Yes / No
- Lower cutoff (Hz):
- Upper cutoff (Hz):
- Reasoning:

### 4. Evidence used
- Dataset type:
- Analysis goal:
- Important preservation constraints:
- QC findings used:
- PSD / time-series features used:

### 5. Cautions
List any risks, tradeoffs, or uncertainties, such as:
- possible distortion of slow components
- possible loss of high-frequency information
- missing context
- need to review filter settings against the final analysis goal

## Behavioral rules
- Be explicit about the difference between noise removal and signal preservation.
- Do not use one-size-fits-all defaults.
- Prioritize the scientific goal of the dataset.
- If ICA is planned, consider whether the final analysis data and ICA-training data should use different high-pass settings.
- If evidence is weak or context is incomplete, say so clearly.
"""
        messages = create_reasoning_messages(
            user_prompt=prompt, image_url="./images/raw_timeseries.jpg"
        )  # TODO - Add current eeg plot url
        ## Ajout du psd plot pour le notch filtering
        messages += create_reasoning_messages(
            user_prompt="Here is the power spectrum of the data. Does it show signs of line noise or other issues that would inform preprocessing decisions?",
            image_url="./images/psd_plot.jpg",
        )
        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=InitialQCResult,
            temperature=0.1,
        )
        result = chat_response.choices[0].message.parsed
        state.skip_stage = result.skip_stages
        state.justification["initial_qc"] = result.justification

    except Exception as e:
        logging.error(f"Error in initial QC: {e}")
        state["errors"].append(f"Initial QC failed: {str(e)}")
        state.skip_stage = []

    state["current_stage"] = "bandpass_filtering"

    logging.info("[INITIAL QC] Placeholder: proceeding with all stages")
    return state


def bandpass_filter(
    raw: mne.io.Raw, l_freq: float = 0.5, h_freq: float = 45.0
) -> mne.io.Raw:
    return raw.filter(l_freq=l_freq, h_freq=h_freq)


names_to_functions = {
    "bandpass_filter": bandpass_filter,
}
bandpass_filtering_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "bandpass_filter",
            "description": "Apply bandpass filter to the data with specified low and high cutoff frequencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "l_freq": {
                        "type": "number",
                        "description": "Low cutoff frequency in Hz. Example: 0.5.",
                    },
                    "h_freq": {
                        "type": "number",
                        "description": "High cutoff frequency in Hz. Example: 45.0.",
                    },
                },
                "required": ["l_freq", "h_freq"],
            },
        },
    }
]


def bandpass_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Bandpass filtering agent.
    Applies bandpass filter to the data if not skipped.
    """
    logging.info(f"[BANDPASS FILTERING] Processing subject {state['subject_id']}")

    if "bandpass_filtering" in state.skip_stage:
        logging.info(
            "[BANDPASS FILTERING] Skipping bandpass filtering based on initial QC."
        )
        state["justification"][
            "bandpass_filtering"
        ] = "Skipped based on initial QC assessment."
        return state

    state["input_raw"] = state["output_raw"]
    state["output_raw"] = None  # Reset output raw for this stage

    experiment_context = state["experiment_metadata"]["experiment_context"]

    try:
        prompt = f"""Here is the context of the experiment: {experiment_context}. 
        You are the Bandpass Filter Agent in an automated EEG preprocessing pipeline for continuous EEG.

Your role is to decide the bandpass filter settings after receiving:
1. the dataset context
2. the Initial QC Agent’s observations
3. the time-series and PSD evidence

You are the agent responsible for choosing the actual bandpass frequencies.  
The Initial QC Agent may tell you whether filtering seems necessary, but you decide the cutoff frequencies.
A separate filter will be chosen for ICA, you should not choose the frequency parameters based on ICA.


## Your objectives
You must:
1. determine whether bandpass filtering is needed
2. choose the lower cutoff frequency
3. choose the upper cutoff frequency
5. justify the choice using both:
   - dataset context
   - evidence from the plots and QC summary

## Inputs you will receive
You will receive:
- dataset type (ERP, resting-state eyes closed, resting-state eyes open, BCI, sleep, clinical EEG, etc.)
- scientific goal / signal of interest
- sampling frequency
- mains frequency if known
- whether preserving very slow activity is important
- whether preserving high-frequency activity is important
- Initial QC summary of:
  - slow drift
  - excessive low-frequency contamination
  - high-frequency noise
  - suspected bad channels
  - line noise
  - whether filtering appears necessary

## Core decision policy
Do not choose bandpass frequencies from PSD and time-series plots alone.

Use:
- the plots and QC findings to determine whether filtering is needed and what noise is present
- the dataset context to decide the actual cutoff frequencies

Your filter choice must be driven primarily by the scientific purpose of the data, not by generic defaults.

## Context-aware decision rules

### ERP / evoked-response datasets
If the dataset is ERP / evoked-response focused:
- use a conservative high-pass
- avoid high-pass choices that may distort slow ERP components
- choose the lowest cutoff compatible with drift control
- choose a low-pass appropriate for the ERP bandwidth of interest and for suppressing unnecessary high-frequency noise


### Resting-state / oscillation-focused datasets
If the dataset is resting-state or oscillation focused:
- preserve the oscillatory bands relevant to the scientific question
- set the high-pass low enough to avoid discarding relevant low-frequency content unless drift control requires otherwise
- set the low-pass above the highest frequency of interest
- do not over-restrict the spectrum if broadband spectral analysis is intended

### BCI / sensorimotor rhythm datasets
If the dataset is BCI / sensorimotor rhythm focused:
- choose the bandpass to preserve the control-relevant rhythms
- prioritize the frequency range that supports decoding or feature extraction
- do not use ERP-style conservative settings unless the target signal is truly ERP-like

## Additional constraints
- Respect the Nyquist limit given the sampling frequency.
- Do not recommend an upper cutoff above what the sampling rate can support.
- If the context says very slow activity must be preserved, avoid aggressive high-pass filtering.
- If the context says high-frequency activity is important, avoid an unnecessarily low upper cutoff.
- If the context is insufficient to choose a defensible frequency range, say so explicitly and provide:
  - a provisional recommendation
  - the missing context needed for a final decision

## Interaction with other agents
- The Initial QC Agent decides whether filtering appears needed and reports artifacts.
- You decide the actual cutoff frequencies.
- You do not identify bad channels or line-noise frequencies unless needed only to justify your filter decision.
- You may recommend separate filtering for:
  - analysis data


## Output format
Return your answer in the following structure:

### 1. Bandpass decision
- Bandpass filtering needed: Yes / No
- Confidence: High / Medium / Low

### 2. Recommended filter for final analysis data
- Lower cutoff (Hz):
- Upper cutoff (Hz):
- Reasoning:

### 3. Recommended filter for ICA-training data
- Separate ICA filter needed: Yes / No
- Lower cutoff (Hz):
- Upper cutoff (Hz):
- Reasoning:

### 4. Evidence used
- Dataset type:
- Analysis goal:
- Important preservation constraints:
- QC findings used:
- PSD / time-series features used:

### 5. Cautions
List any risks, tradeoffs, or uncertainties, such as:
- possible distortion of slow components
- possible loss of high-frequency information
- missing context
- need to review filter settings against the final analysis goal

## Behavioral rules
- Be explicit about the difference between noise removal and signal preservation.
- Do not use one-size-fits-all defaults.
- Prioritize the scientific goal of the dataset.
- If ICA is planned, consider whether the final analysis data and ICA-training data should use different high-pass settings.
- If evidence is weak or context is incomplete, say so clearly.

        """
        messages = [
            {
                "role": "system",
                "content": "you are a helpful assistant for EEG data analysis.Based on the experiment context, determine the optimal bandpass filter settings (low and high cutoff frequencies) for preprocessing EEG data. Call the tools after you have determined the optimal settings and always provide a justification for your choices based on the experiment context.",
            },
            {"role": "user", "content": prompt},
        ]
        chat_response = client.chat.complete(
            model="mistral-large-2512",
            messages=messages,
            tools=bandpass_filtering_agent_tools,
        )
        
        # Check if tool_calls is available
        if not chat_response.choices[0].message.tool_calls:
            logging.error("No tool calls returned from the model")
            state["errors"].append("Bandpass filtering failed: No tool calls from model")
            return state
            
        # while chat_response.choices[0].message.tool_calls:
        if chat_response.choices[0].message.content:
            logging.info(f"Model response: {chat_response.choices[0].message.content}")
        tool_call = chat_response.choices[0].message.tool_calls[0]
        function_name = tool_call.name
        function_to_call = names_to_functions[function_name]
        arguments = tool_call.arguments
        raw_filtered = function_to_call(raw=state["input_raw"], **arguments)

        # chat_response = client.chat.complete(
        #     model="mistral-large-2512",
        #     messages=messages,
        #     tools=bandpass_filtering_agent_tools,
        #     tool_results=[{"name": function_name, "result": str(raw_filtered)}],
        # )
        # messages.append({"role": "tool", "content": {"name": function_name, "result": str(raw_filtered)}})
        justification = chat_response.choices[0].message.content
        state["output_raw"] = (
            raw_filtered  # Assuming the function result is the filtered raw data
        )
        state["justification"]["bandpass_filtering"] = justification
        return state

    except Exception as e:
        logging.error(f"Error in bandpass filtering: {e}")
        state["errors"].append(f"Bandpass filtering failed: {str(e)}")
        return state


class BadChannelAnalysis(BaseModel):
    bad_channels_to_remove: list = Field(
        description="List of bad channels to remove based on the analysis"
    )
    justification: str = Field(
        description="A brief explanation of the reasoning behind the selected bad channels"
    )


def bad_channel_identifier_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    Identifies bad EEG channels that should be removed.
    """
    logging.info(f"[BAD CHANNEL IDENTIFIER] Processing subject {state['subject_id']}")

    if "bad_channel_identification" in state.skip_stage:
        logging.info(
            "[BAD CHANNEL IDENTIFIER] Skipping bad channel identification based on initial QC."
        )
        state["justification"][
            "bad_channel_identification"
        ] = "Skipped based on initial QC assessment."
        return state

    state["input_raw"] = state["output_raw"]
    state["output_raw"] = None  # Reset output raw for this stage

    input_timeseries_fig = state["input_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    input_timeseries_fig.savefig("./images/raw_timeseries.jpg")

    input_psd_fig = state["input_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    input_psd_fig.savefig("./images/psd_plot.jpg")

    input_sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    input_sensors_fig.savefig("./images/sensors.jpg")

    try:
        prompt = """Prompt for the Bad Channel Agent
You are the Bad Channel Agent in an automated EEG preprocessing pipeline for continuous EEG.
Your role is to inspect the EEG time-series plots and PSD plots and determine which channels are:
Good
Suspicious
Bad
You do not repair, interpolate, remove, or re-reference channels.
You only identify channels that appear abnormal and provide evidence for that decision.
Your objectives
You must:
inspect channel time-series traces
inspect channel PSD plots
identify channels that are visually abnormal
classify each flagged channel as Suspicious or Bad
explain why each flagged channel was marked
Core decision principle
A channel should be considered bad when its behavior is persistently inconsistent with plausible EEG and/or clearly inconsistent with the rest of the montage.
Do not rely on a single weak cue when multiple interpretations are possible.
Use combined evidence from:
waveform shape
amplitude stability
variance
dropout / clipping behavior
spectral shape
line-noise prominence
consistency with neighboring or comparable channels
Visual criteria in the time series
Mark a channel as bad if you observe:
flatline or near-flatline behavior, especially if sustained
repeated dropouts, signal loss, or interruptions
repeated clipping or saturation
frequent large pops, step-like jumps, or abrupt transients
persistently excessive amplitude or variance relative to other channels
signal morphology that looks persistently nonphysiologic and unlike the rest of the montage
Mark a channel as suspicious if:
the channel is only intermittently abnormal
amplitude is somewhat elevated but not clearly pathological
there are occasional transients without persistent instability
the trace differs from neighbors but not enough for a confident bad label
Visual criteria in the PSD
Mark a channel as bad if you observe:
a PSD that is a strong outlier relative to other channels
persistently elevated broadband high-frequency power
unusually strong low-frequency dominance suggesting severe drift / instability
a disproportionately large narrow 50/60 Hz peak relative to other channels
a spectrum that is grossly inconsistent with comparable electrodes
Mark a channel as suspicious if:
the PSD is somewhat atypical but not clearly pathological
a line-noise peak is elevated but not dramatically worse than other channels
spectral elevation is present only in a limited band and may reflect local physiology or temporary artifact
Reference heuristics
Use the following as supporting heuristics, not rigid rules:
a flatline of about 5 seconds or more strongly supports marking a channel abnormal
very low agreement with neighboring or comparable channels supports a bad-channel interpretation
unusually high line noise relative to total channel signal supports a bad-channel interpretation
if a channel appears poorly correlated with the rest of the montage, that supports marking it suspicious or bad
Important distinctions
Do not confuse:
a bad channel
with
a good channel during a brief artifact burst affecting many channels
A bad channel is typically:
persistently abnormal
consistently out of family with the montage
spectrally atypical in a stable way
degraded beyond what would be expected from normal physiological artifact alone
If many channels are simultaneously contaminated, that may indicate a bad segment rather than a single bad channel.
Output format
1. Summary
Briefly summarize the overall channel quality:
number of clearly bad channels
number of suspicious channels
whether abnormalities are isolated or widespread
2. Flagged channels
For each flagged channel, report:
Channel name:
Classification: Suspicious / Bad
Time-series evidence:
PSD evidence:
Confidence: High / Medium / Low
3. Decision rationale
Explain the main reasons you used to mark channels, such as:
flatline
clipping
repeated pops
persistent high variance
broadband noise
excessive low-frequency drift
excess line noise
spatial inconsistency with nearby channels
4. Global caution
State whether the observed problems look like:
isolated bad channels
widespread recording-quality problems
possible bad time segments rather than bad sensors
Behavioral rules
Base your decisions only on the time-series and PSD evidence available.
Be conservative when evidence is weak.
Use combined evidence rather than a single ambiguous cue.
Distinguish Bad from Suspicious.
If a channel is borderline, prefer Suspicious over Bad.
If abnormalities are widespread across many channels, explicitly say that the issue may not be limited to individual bad channels.
"""

        messages = create_reasoning_messages(prompt, "./images/raw_timeseries.jpg")

        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=BadChannelAnalysis,
            temperature=0.1,
        )

        result = chat_response.choices[0].message.parsed
        state["bad_channels"] = result.bad_channels_to_remove

        logging.info(
            f"Identified {len(result.bad_channels_to_remove)} bad channels: {result.bad_channels_to_remove}"
        )

    except Exception as e:
        logging.error(f"Error in bad channel detection: {e}")
        state["errors"].append(f"Bad channel detection failed: {str(e)}")

    state["current_stage"] = "notch_filtering"
    return state


def annotate_bad_channels(raw: mne.io.Raw, bad_channels: List[str]) -> mne.io.Raw:
    """Annotate bad channels in the raw object"""
    raw.info["bads"] = bad_channels
    return raw


def notch_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Notch filtering agent.
    Applies notch filter to the data if not skipped.
    """
    logging.info(f"[NOTCH FILTERING] Processing subject {state['subject_id']}")

    if "notch_filtering" in state.skip_stage:
        logging.info("[NOTCH FILTERING] Skipping notch filtering based on initial QC.")
        state["justification"][
            "notch_filtering"
        ] = "Skipped based on initial QC assessment."
        return state

    state["input_raw"] = state["output_raw"]
    state["output_raw"] = None  # Reset output raw for this stage

    input_timeseries_fig = state["input_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    input_timeseries_fig.savefig("./images/raw_timeseries.jpg")

    input_psd_fig = state["input_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    input_psd_fig.savefig("./images/psd_plot.jpg")

    input_sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    input_sensors_fig.savefig("./images/sensors.jpg")

    try:
        prompt = """
You are the Notch Filter Agent in an automated EEG preprocessing pipeline for continuous EEG.

Your role is to decide whether notch filtering should be applied, and if so:
- which frequency or frequencies to remove
- which channels to apply it to
- whether the result looks appropriate after filtering

You are responsible for narrow-band line-noise removal, not broad filtering.

## Your purpose
A notch filter is used to remove a narrow-band sinusoidal contaminant while preserving nearby frequencies as much as possible.

In EEG, this is most commonly used to suppress:
- 50 Hz line noise
- 60 Hz line noise
- and sometimes their harmonics, if those harmonics are clearly present and fall inside the frequencies being preserved for analysis

## Your inputs
You will receive:
- the PSD plots
- the time-series plots
- the analysis passband or intended kept frequency range


## Core decision policy
Use notch filtering only if the PSD shows a clear narrow line-noise peak in frequencies that are intended to be kept for analysis.

Do not apply notch filtering just because 50 Hz or 60 Hz exists in theory.

You must decide from the actual PSD:
- whether a narrow mains-related peak is present
- whether that peak lies inside the kept band
- whether harmonics are also clearly present and relevant

## When to use a notch filter
Recommend notch filtering only when all of the following are true:
1. the PSD shows a clear narrow peak at a mains frequency such as 50 Hz or 60 Hz
2. that frequency lies inside the analysis band being preserved
3. removing it is likely to improve the data without unnecessarily distorting nearby frequencies

## When not to use a notch filter
Do not recommend notch filtering when:
- the PSD does not show a clear narrow mains peak
- the line-noise frequency is already outside the retained passband
- the apparent contamination is broad-band rather than narrow-band
- the notch would remove frequencies that are not actually problematic
- the filtering step is unnecessary because the passband already excludes the mains frequency

## Passband-aware decision rules
You must always consider the intended analysis passband before recommending a notch filter.

Examples:
- If the analysis band is 0.5–45 Hz, then 50/60 Hz is outside the retained band, so a notch filter is often unnecessary.
- If the analysis band extends to 80 Hz or 100 Hz, then 50/60 Hz may still be inside the kept band, so notch filtering may be appropriate.
- If harmonics such as 100 Hz or 120 Hz are outside the kept band, they usually do not need notching.

## How to choose notch frequencies
Choose notch frequencies from the PSD itself.

Rules:
- choose the actual narrow peak you see
- notch 50 Hz only if you see a narrow peak at 50 Hz
- notch 60 Hz only if you see a narrow peak at 60 Hz
- add harmonics only if:
  - they are also clearly visible in the PSD
  - they are inside the kept analysis band

Examples:
- Notch 50 Hz if there is a clear narrow peak at 50 Hz
- Notch 60 Hz if there is a clear narrow peak at 60 Hz
- Add 100 Hz or 120 Hz only if those harmonics are clearly present and also relevant to the preserved band

Do not automatically notch harmonics unless they are both visible and relevant.

## Which channels to apply it to
Apply notch filtering to the channels that:
- actually carry the line-noise contamination
- are relevant to the analysis

Do not assume all channels need notch filtering unless the contamination is clearly widespread.

If contamination appears limited to only certain channels, say so explicitly.

## What to verify after filtering

### In the PSD
After appropriate notch filtering, you should expect:
- the narrow spike at 50 Hz or 60 Hz to be reduced or removed
- nearby frequencies to remain mostly preserved
- no broad reshaping of the spectrum

### In the time series
After appropriate notch filtering, you should expect:
- little or no dramatic overall waveform change
- traces that may look slightly cleaner
- no major transformation of the signal shape

If the waveform looks radically different after filtering, that may suggest:
- the filter was too broad
- the removed frequency contained more than narrow line noise
- the notch step was unnecessary
- the wrong frequency was targeted

## Your decision process
Follow this order:
1. inspect the PSD
2. identify the intended analysis passband
3. determine whether line noise lies inside that passband
4. decide whether the contamination is narrow-band and appropriate for notch filtering
5. choose the notch frequency or frequencies from the PSD
6. determine which channels are affected
7. verify that the filtered result reduces the narrow spectral peak without causing broader distortion

## Output format

### 1. Notch decision
- Notch filtering needed: Yes / No
- Confidence: High / Medium / Low

### 2. Frequencies to notch
- Primary frequency:
- Harmonics to notch:
- Reasoning:

### 3. Channels to filter
- Apply to:
- Why these channels:

### 4. Evidence used
- PSD evidence:
- Passband considered:
- Time-series evidence:
- Whether contamination is narrow-band or broad-band:

### 5. Post-filter expectations / verification
State what should happen after filtering:
- in the PSD
- in the time series

### 6. Warnings or cautions
List any concerns, such as:
- frequency already outside the passband
- no clear narrow line-noise peak
- possible over-filtering
- possible unnecessary harmonics
- contamination looks broad-band rather than narrow-band

## Behavioral rules
- Only recommend notch filtering when the PSD supports it.
- Always consider the intended analysis passband.
- Choose frequencies from the observed spectrum, not from assumptions.
- Add harmonics only if they are clearly present and relevant.
- Prefer narrow, targeted intervention over unnecessary filtering.
- Be conservative: if there is no clear narrow line-noise peak, do not recommend notch filtering.
"""
        messages = create_reasoning_messages(prompt, "./images/psd_plot.jpg")
        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=InitialQCResult,
            temperature=0.1,
        )
        result = chat_response.choices[0].message.parsed
        if "notch_filtering" in result.skip_stages:
            state["justification"]["notch_filtering"] = result.justification.get(
                "notch_filtering", "No justification provided."
            )
            logging.info(
                "[NOTCH FILTERING] Skipping notch filtering based on PSD analysis."
            )
            return state
        else:
            logging.info(
                "[NOTCH FILTERING] Applying notch filter based on PSD analysis."
            )
            # Apply notch filter - detect frequency based on location (50Hz for Europe, 60Hz for US)
            # Here we default to 50Hz, but this could be determined from metadata
            freqs_to_notch = [50.0, 100.0]  # 50 Hz and its harmonic
            raw_notched = state["input_raw"].copy().notch_filter(
                freqs=freqs_to_notch, picks="eeg", method="fir", phase="zero"
            )
            state["output_raw"] = raw_notched
            state["justification"]["notch_filtering"] = result.justification.get(
                "notch_filtering", "No justification provided."
            )
            return state
    except Exception as e:
        logging.error(f"Error in notch filtering decision: {e}")
        state["errors"].append(f"Notch filtering decision failed: {str(e)}")
        return state


def apply_slow_drift_correction(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply slow drift correction to the raw data."""
    raw_corrected = raw.copy().filter(
        l_freq=0.5, h_freq=None
    )  # High-pass filter to remove slow drifts
    return raw_corrected


def slow_drift_analysis_agent(state: EEGPipelineState) -> EEGPipelineState:

    state["input_raw"] = state["output_raw"]
    state["output_raw"] = None  # Reset output raw for this stage

    input_timeseries_fig = state["input_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    input_timeseries_fig.savefig("./images/raw_timeseries.jpg")

    try:
        prompt = """You are a helpful assistant for EEG data analysis. I will give you an image of an EEG plot. I want you to analyze the plot and say whether the data shows signs of slow drifts or not. In raw EEG recordings, slow drifts appear as a gradual upward or downward shift in the signal baseline across channels. Respond by giving a probability between 0 and 1. 1 means the data is very likely to show slow drifts, 0 means it is very unlikely. Add a brief justification for your answer."""
        messages = create_reasoning_messages(prompt, "./images/raw_timeseries.jpg")
        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=EEGSlowDriftAnalysis,
        )
        result = chat_response.choices[0].message.parsed
        state["slow_drift_probability"] = result.slow_drift_probability
        state["justification"]["slow_drift"] = result.justification

        if result.slow_drift_probability > SLOW_DRIFT_THRESHOLD:
            logging.info(
                f"Data shows signs of slow drifts with probability {result.slow_drift_probability}."
            )
            raw_corrected = apply_slow_drift_correction(state["input_raw"])
            state["output_raw"] = raw_corrected
        else:
            logging.info(
                f"Data does not show strong signs of slow drifts (probability {result.slow_drift_probability}). Skipping slow drift correction."
            )
            state["output_raw"] = state["input_raw"]
        state["current_stage"] = "bad_channel_identification"
    except Exception as e:
        logging.error(f"Error in slow drift analysis: {e}")
        state["errors"].append(f"Slow drift analysis failed: {str(e)}")
        return state
    return state


def resampling(state: EEGPipelineState) -> EEGPipelineState:
    """[TBC] Resampling agent."""
    logging.info(f"[RESAMPLING] Processing subject {state['subject_id']}")
    state["input_raw"] = state["output_raw"]
    state["output_raw"] = None  # Reset output raw for this stage
    ### Optional resampling before ICA
    raw_rs = state["input_raw"].copy()
    raw_rs.resample(
        sfreq=250.0,
        method="polyphase",  # study this vs default fft
    )
    state["output_raw"] = raw_rs
    return state


### Make a dedicated ICA copy with stronger high-pass
def prepare_ica_copy(state: EEGPipelineState) -> EEGPipelineState:
    state["input_raw"] = state["output_raw"].copy()
    raw_ica = state["input_raw"].copy()
    raw_ica.filter(
        l_freq=1.0,
        h_freq=45.0,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
    )
    state["output_raw"] = raw_ica
    return state


def apply_ica(state: EEGPipelineState, n_components: int = 20) -> ICA:

    ica = ICA(n_components=n_components, random_state=97)
    ica.fit(state["output_raw"])
    return ica


def ica_discrimination_agent(
    state: EEGPipelineState, ica: ICA
) -> EEGPipelineState:
    """
    [TBC] ICA discrimination agent.
    Identifies which ICA components to remove based on the analysis of ICA plots.
    """
    logging.info(f"[ICA DISCRIMINATION] Processing subject {state['subject_id']}")
    if "ica" in state.skip_stage:
        logging.info(
            "[ICA DISCRIMINATION] Skipping ICA discrimination based on initial QC."
        )
        state["justification"][
            "ica_discrimination"
        ] = "Skipped based on initial QC assessment."
        return state
    
    # Generate ICA components plot
    input_ica_fig = ica.plot_components(picks=range(min(20, ica.n_components_)), show=False)
    input_ica_fig.savefig("./images/ica_components.jpg")
    try:

        messages = create_reasoning_messages(
            user_prompt="""
You are the ICA Agent in an automated EEG preprocessing pipeline for continuous EEG.

Your role is to inspect ICA components and decide which components should be removed.

You will receive, for each ICA component:
- the component time series
- the component scalp map
- the component PSD

Your task is to determine whether each component is:
- Keep
- Suspicious
- Remove

You must base your decision on known ICA artifact patterns, especially those described in MNE and EEGLAB documentation for:
- eye blinks
- vertical EOG
- horizontal eye movements / saccades
- ECG / heartbeat
- muscle artifacts, including jaw clenching / temporalis EMG
- other obvious non-neural artifacts when strongly supported by the plots

## Core role
You are not fitting ICA and you are not applying the component removal yourself.

You only:
1. inspect the ICA component plots
2. compare them to known artifact patterns
3. decide which components should be removed
4. explain the evidence for each decision

## Main decision principle
A component should be removed only when the evidence across the time series, scalp map, and PSD strongly supports that it represents an artifact rather than plausible neural activity.

Do not remove a component based on only one weak cue.

Use combined evidence from:
- spatial pattern on the scalp map
- temporal pattern in the component time series
- spectral pattern in the PSD

If evidence is incomplete or mixed, label the component Suspicious rather than Remove.

## Artifact templates to use

### 1. Eye blink / vertical EOG artifact
Typical blink-related components often show:
- a strong frontal or far-frontal scalp projection
- a smooth, low-frequency-dominated spectrum
- large intermittent deflections in the component time series corresponding to blink events

Use these cues:
- Scalp map:
  - strongest weights over frontal poles / front of the head
  - often broad and frontal rather than focal and posterior
- Time series:
  - large, intermittent, slow transient events
  - repeated blink-like bursts rather than sustained oscillations
- PSD:
  - dominant low-frequency content
  - smoothly decreasing spectrum
  - not primarily high-frequency

Mark as Remove when the frontal map, low-frequency spectrum, and blink-like transients all align.

### 2. Horizontal eye movement / saccade artifact
Typical lateral eye-movement components often show:
- a left-right frontal dipole
- slow transient activity related to eye movements
- low-frequency dominance in the PSD

Use these cues:
- Scalp map:
  - opposite polarities over left and right frontal regions
  - lateral frontal distribution
- Time series:
  - transient step-like or slow eye-movement-like events
  - repeated episodes rather than stationary rhythmic activity
- PSD:
  - mainly low-frequency energy
  - smoothly decreasing spectrum

Mark as Remove when the lateral frontal map and eye-movement-like temporal pattern are both clear.

### 3. ECG / heartbeat artifact
Heartbeat-related components often show:
- a highly repetitive, pulse-like temporal pattern
- a scalp map that is not necessarily neural-looking
- a PSD that may show repeated rhythmic contamination rather than a classic EEG shape

Use these cues:
- Time series:
  - very regular repeating pulse pattern
  - periodic activity at the heart rate
- Scalp map:
  - may be diffuse or atypical
  - often not a clean dipolar cortical-looking map
- PSD:
  - may show rhythm related to pulse repetition and harmonics
  - should be interpreted together with the strong periodicity in the time series

Mark as Remove when the component time series shows clear heartbeat periodicity and the scalp map / PSD do not support a plausible brain source.

### 4. Muscle artifact / EMG / jaw clenching / temporalis artifact
Use MNE-style muscle criteria.

Typical muscle-related components often show:
- peripheral scalp focus
- focal or low-smoothness map
- increased high-frequency power
- spiky, fuzzy, or bursty component time course

Use these cues:
- Scalp map:
  - peripheral, temporal, jaw, cheek, or edge-of-head focus
  - single focal point or low spatial smoothness
  - not a broad smooth cortical dipole
- Time series:
  - fast, irregular, spiky, fuzzy, or bursty activity
  - can include repeated bursts consistent with jaw clenching or facial tension
- PSD:
  - elevated high-frequency power
  - a spectrum compatible with muscle contamination
  - MNE specifically describes muscle ICs using a positive log-log spectral slope in about the 7–45 Hz range, together with peripheral focus and low spatial smoothness

Mark as Remove when the map is peripheral/focal and the PSD/time series show classic EMG-like activity.

### 5. Other obvious non-neural artifact
You may mark a component for removal if all three plot types strongly support a non-neural source, even if it does not fit neatly into one of the major classes above.

Examples may include:
- line-noise-dominated components
- electrode-pop-like components
- clearly nonphysiologic components with abnormal time series and non-brain scalp maps

However:
- be conservative
- do not remove components just because they are unusual
- if unsure, mark Suspicious

## Features that support keeping a component
A component is more likely to be kept if it shows:
- a smooth, plausible, cortical-looking scalp map
- a PSD more consistent with neural 1/f-like activity than with narrow-band noise or EMG
- a time series that does not show stereotyped artifact events
- no strong match to blink, saccade, ECG, or muscle patterns

## Important cautions
- Do not remove components only because they are large.
- Do not remove components only because the scalp map is frontal; frontal neural sources can exist.
- Do not remove components only because the PSD is unusual.
- A removal decision should ideally be supported by at least two plot types, and preferably all three.
- If a component could plausibly be neural, prefer Suspicious over Remove.

## Decision policy
For each component:
- Remove = strong multi-plot evidence for a known artifact pattern
- Suspicious = some artifact-like evidence, but not conclusive
- Keep = no convincing evidence of artifact

## Required output format

### 1. Overall summary
- Number of components marked Remove:
- Number of components marked Suspicious:
- Main artifact classes detected:

### 2. Component-by-component decisions
For each component, report:
- Component ID:
- Decision: Keep / Suspicious / Remove
- Most likely label:
  - Blink / vertical EOG
  - Horizontal eye movement
  - ECG
  - Muscle / jaw clench / temporalis EMG
  - Other artifact
  - Plausible neural / unclear
- Scalp map evidence:
- Time-series evidence:
- PSD evidence:
- Confidence: High / Medium / Low

### 3. Removal list
Provide:
- Components recommended for removal:
- Components that are borderline / suspicious but not confidently removable:

### 4. Global cautions
State:
- whether artifact labeling was straightforward or ambiguous
- whether any components may require human review
- whether multiple components appear to reflect the same artifact class

## Behavioral rules
- Base all judgments only on the provided plots.
- Use known MNE / EEGLAB artifact patterns.
- Be conservative when evidence is mixed.
""",
            image_url="./images/ica_components.jpg",
        )

        chat_response = client.chat.parse(
            model=model,
            messages=messages,
            prompt_mode="reasoning",
            response_format=ICAAnalysis,
            temperature=0.1,
        )
        result = chat_response.choices[0].message.parsed
        state["ica_channels_to_remove"] = result.ica_channels_to_remove
        state["ica_justification"] = result.justification
    except Exception as e:
        logging.error(f"Error in ICA discrimination: {e}")
        state["errors"].append(f"ICA discrimination failed: {str(e)}")
        return state


def apply_ica_correction(state: EEGPipelineState, ica: ICA) -> EEGPipelineState:
    """Apply ICA correction by removing identified components"""
    ica.exclude = state["ica_channels_to_remove"]
    raw_corrected = state["input_raw"].copy()
    ica.apply(raw_corrected)
    state["output_raw"] = raw_corrected
    return state


def interpolate_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Interpolate bad channels in the raw object"""
    raw.interpolate_bads(reset_bads=True)
    return raw


def final_qc_agent(state: EEGPipelineState) -> EEGPipelineState:
    """[TBC] Final quality control agent."""
    logging.info(f"[FINAL QC] Processing subject {state['subject_id']}")

    state["input_raw"] = state["output_raw"]
    state["output_raw"] = None  # Reset output raw for this stage

    timeseries_fig = state["input_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    timeseries_fig.savefig("./images/final_qc_timeseries.jpg")
    psd_fig = state["input_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    psd_fig.savefig("./images/final_qc_psd.jpg")
    sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    sensors_fig.savefig("./images/final_qc_sensors.jpg")

    try:
        prompt = """You are the Final QC Agent in an automated EEG preprocessing pipeline for continuous EEG.

Your role is to inspect the post-processing EEG output and determine whether the preprocessing pipeline was successful.

You evaluate the cleaned data after the applied steps, which may include:
- bandpass filtering
- bad channel handling
- notch filtering
- ICA-based artifact removal

You do not choose preprocessing parameters and you do not decide which agents to run.  
Your task is to verify whether the final output is acceptable, whether artifacts were appropriately reduced, and whether the cleaned data still look physiologically plausible.

## Your objectives
You must:
1. inspect the final cleaned EEG time-series plots
2. inspect the final cleaned EEG PSD plots
3. compare the cleaned output to the expected effects of the preprocessing steps
4. determine whether the preprocessing appears successful
5. identify any remaining problems
6. flag signs of over-cleaning, distortion, or unresolved artifacts

## Inputs you may receive
You may receive:
- final cleaned EEG time-series plots
- final cleaned EEG PSD plots
- summary of the preprocessing steps that were applied
- list of bad channels that were removed / interpolated / flagged
- list of ICA components removed
- intended analysis context:
  - ERP
  - resting-state eyes closed / open
  - BCI / sensorimotor rhythm
  - sleep
  - clinical continuous EEG
  - other continuous EEG context
- intended analysis passband or kept frequency range

## Core decision principle
Your role is to decide whether the final cleaned data are:
- Acceptable
- Acceptable with cautions
- Needs review / reprocessing
- Poor / failed preprocessing

You must balance two goals:
1. artifact reduction
2. signal preservation

Good preprocessing should reduce noise and artifacts without destroying plausible neural signal.

## What to inspect in the final time series
Look for:
- whether the traces look cleaner than before, if before/after context is available
- whether obvious line-noise, drift, dropouts, or unstable channels remain
- whether large blink, eye movement, ECG, or muscle bursts still dominate the signal
- whether the waveform now looks unnaturally flattened, distorted, or over-cleaned
- whether many channels still look globally noisy
- whether remaining abnormalities are isolated or widespread

## What to inspect in the final PSD
Look for:
- whether narrow 50/60 Hz peaks have been appropriately reduced, if notch filtering was applied
- whether broad spectral shape still looks physiologically plausible
- whether excessive low-frequency drift has been reduced, if filtering was intended to remove it
- whether excessive high-frequency contamination has been reduced, if filtering / ICA was meant to reduce it
- whether the spectrum shows signs of over-filtering, spectral holes, or broad distortion
- whether any channels still show outlier spectra suggesting unresolved bad channels

## Step-specific expectations

### 1. After bandpass filtering
Expected:
- reduced unwanted low-frequency drift and/or high-frequency contamination
- preservation of the intended analysis band
- no obvious broad spectral distortion beyond the intended passband
- no implausible flattening of the signal

Possible concerns:
- overly aggressive high-pass filtering may distort slow activity
- overly aggressive low-pass filtering may remove relevant high-frequency information
- traces may look unnaturally smoothed or altered if filtering was too strong

### 2. After bad channel handling
Expected:
- obvious bad channels should no longer dominate the montage
- the remaining channels should look more consistent with one another
- no unresolved flat, clipped, dropout-heavy, or grossly noisy channels should remain if they were supposed to be addressed

Possible concerns:
- clearly bad channels still visible
- too many channels still appear abnormal
- widespread poor data quality suggesting a bad recording rather than isolated bad sensors

### 3. After notch filtering
Expected:
- the narrow 50/60 Hz spike, and only the intended harmonics if targeted, should be reduced
- nearby frequencies should remain mostly preserved
- time traces should not look radically transformed

Possible concerns:
- no real reduction of the narrow spectral line
- broad reshaping of nearby frequencies
- dramatic waveform changes suggesting over-filtering or unnecessary notching

### 4. After ICA artifact removal
Expected:
- reduction of stereotyped artifacts such as:
  - blinks
  - eye movements
  - ECG
  - muscle bursts
- preserved overall physiological structure of the EEG
- no obvious removal of large amounts of plausible neural signal

Possible concerns:
- blink / EOG / ECG / muscle patterns still strongly present
- the data now look unnaturally attenuated or distorted
- frontal activity appears excessively erased
- too much signal seems removed relative to the expected artifact reduction

## Main artifact questions to answer
In the final cleaned data, determine whether the following are:
- resolved
- partially resolved
- still present
- possibly over-corrected

Assess:
- slow drift
- bad channels
- line noise
- blink / EOG contamination
- ECG contamination
- muscle contamination
- broad nonstationary noise

## How to judge overall success

### Acceptable
Use this when:
- major targeted artifacts are reduced
- the PSD looks plausible for the intended analysis
- no major residual artifacts dominate the recording
- no strong evidence of over-cleaning is present

### Acceptable with cautions
Use this when:
- preprocessing mostly worked
- some residual artifacts or uncertainties remain
- the data may still be usable, but with caveats

### Needs review / reprocessing
Use this when:
- important artifacts remain
- bad channels still appear unresolved
- notch or bandpass effects look questionable
- ICA removal appears incomplete or possibly too aggressive
- the output may be usable only after manual review or adjustment

### Poor / failed preprocessing
Use this when:
- the cleaned data still look dominated by artifacts
- preprocessing introduced major distortion
- signal preservation appears poor
- the output is not trustworthy without substantial reprocessing

## Required output format

### 1. Overall verdict
Choose one:
- Acceptable
- Acceptable with cautions
- Needs review / reprocessing
- Poor / failed preprocessing

### 2. Summary of final data quality
Briefly summarize:
- overall cleanliness of the time series
- overall plausibility of the PSD
- whether the cleaned output appears suitable for the intended analysis

### 3. Step-by-step QC assessment
For each applied preprocessing step, report:
- Step:
- Expected effect:
- Observed effect:
- Verdict: Successful / Partially successful / Unclear / Unsuccessful

Steps to assess if present:
- Bandpass filtering
- Bad channel handling
- Notch filtering
- ICA artifact removal

### 4. Residual artifact assessment
For each of the following, report:
- slow drift: resolved / partial / present / over-corrected
- bad channels: resolved / partial / present
- line noise: resolved / partial / present / over-corrected
- blink / EOG: resolved / partial / present / over-corrected
- ECG: resolved / partial / present / over-corrected
- muscle artifact: resolved / partial / present / over-corrected
- broad nonstationary noise: resolved / partial / present

### 5. Signs of over-processing
Explicitly state whether there is evidence of:
- excessive filtering
- spectral distortion
- unnatural flattening or attenuation
- excessive ICA correction
- loss of plausible neural signal

### 6. Confidence and cautions
Report:
- confidence: High / Medium / Low
- remaining concerns
- whether human review is recommended

## Behavioral rules
- Judge preprocessing success, not preprocessing intention.
- Always balance artifact removal against signal preservation.
- Do not assume “cleaner” automatically means “better.”
- Be conservative when declaring success if important artifacts remain.
- Explicitly flag signs of over-cleaning or distortion.
- Base all conclusions only on the provided plots and preprocessing summary.
- If evidence is mixed, say so clearly.
"""
        messages = create_reasoning_messages(prompt, "./images/final_qc_timeseries.jpg")
        messages += [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the power spectrum of the final preprocessed data. Analyze it and consider if there are any remaining issues that could be addressed with further preprocessing steps.",
                    },
                    {"type": "image_url", "image_url": "./images/final_qc_psd.jpg"},
                    {"type": "image_url", "image_url": "./images/final_qc_sensors.jpg"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the initial raw data plots for comparison. Analyze it in conjunction with the final preprocessed data to assess the effectiveness of the preprocessing steps and identify any remaining issues.",
                    },
                    {"type": "image_url", "image_url": "./images/raw_timeseries.jpg"},
                    {"type": "image_url", "image_url": "./images/raw_psd.jpg"},
                    {"type": "image_url", "image_url": "./images/raw_sensors.jpg"},
                ],
            },
        ]
        chat_response = client.chat.parse(
            model=model,
            messages=messages,
            prompt_mode="reasoning",
            response_format=InitialQCResult,
            temperature=0.1,
        )
        state["final_qc_assessment"] = chat_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in final QC assessment: {e}")
        state["errors"].append(f"Final QC assessment failed: {str(e)}")

    return state


if __name__ == "__main__":
    state = initial_qc_agent(state)
    SKIP_STAGE = state['skip_stage']
    logging.info(f"Initial QC completed. Stages to skip: {SKIP_STAGE}")
    if SKIP_STAGE:
        logging.info(f"Skipping stages: {SKIP_STAGE}")
    else:
        logging.info("No stages to skip. Proceeding with full pipeline.")
    if "bandpass_filtering" not in SKIP_STAGE:
        state = bandpass_filtering_agent(state)
    logging.info(
        f"Bandpass filtering completed. Justification: {state['justification'].get('bandpass_filtering', 'No justification provided.')}"
    )

    if "bad_channel_identification" not in SKIP_STAGE:
        state = bad_channel_identifier_agent(state)
    logging.info(
        f"Bad channel identification completed. Justification: {state['justification'].get('bad_channel_identification', 'No justification provided.')}"
    )
    state["output_raw"] = annotate_bad_channels(
        state["input_raw"], state["bad_channels"]
    )
    if "notch_filtering" not in SKIP_STAGE:
        state = notch_filtering_agent(state)
    logging.info(
        f"Notch filtering decision completed. Justification: {state['justification'].get('notch_filtering', 'No justification provided.')}"
    )
    if "ica" not in SKIP_STAGE:
        state = resampling(state)
        state = prepare_ica_copy(state)
        ica = apply_ica(state)
        state = ica_discrimination_agent(state, ica)
        state = apply_ica_correction(state, ica)
    logging.info(
        f"ICA correction completed. Justification: {state.get('ica_justification', 'No justification provided.')}"
    )

    if state["bad_channels"]:
        state["output_raw"] = interpolate_bad_channels(state["output_raw"])
    timeseries_fig = state["output_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    timeseries_fig.savefig("./images/current_eeg_timeseries.jpg")
    psd_fig = state["output_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    psd_fig.savefig("./images/current_eeg_psd.jpg")
    sensors_fig = state["output_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    sensors_fig.savefig("./images/current_eeg_sensors.jpg")
