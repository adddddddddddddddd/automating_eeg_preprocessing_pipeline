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

import requests
import json
import httpx


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

# if not os.path.exists("../datasets/ds004504"):
#     openneuro.download(dataset="ds004504", target_dir="../datasets/ds004504")


bids_path = BIDSPath(
    subject="001",  # Replace with subject ID (e.g., '001' to '088')
    task="eyesclosed",
    root="../datasets/ds004504",
    datatype="eeg",
)

logger.info("Reading raw EEG data from BIDS path...")

ALZEIMER_EXPERIMENT_CONTEXT = """This dataset contains the EEG resting state-closed eyes recordings from 88 subjects in total.

Participants: 36 of them were diagnosed with Alzheimer's disease (AD group), 23 were diagnosed with Frontotemporal Dementia (FTD group) and 29 were healthy subjects (CN group). Cognitive and neuropsychological state was evaluated by the international Mini-Mental State Examination (MMSE). MMSE score ranges from 0 to 30, with lower MMSE indicating more severe cognitive decline. The duration of the disease was measured in months and the median value was 25 with IQR range (Q1-Q3) being 24 - 28.5 months. Concerning the AD groups, no dementia-related comorbidities have been reported. The average MMSE for the AD group was 17.75 (sd=4.5), for the FTD group was 22.17 (sd=8.22) and for the CN group was 30. The mean age of the AD group was 66.4 (sd=7.9), for the FTD group was 63.6 (sd=8.2), and for the CN group was 67.9 (sd=5.4).

Recordings: Recordings were aquired from the 2nd Department of Neurology of AHEPA General Hospital of Thessaloniki by an experienced team of neurologists. For recording, a Nihon Kohden EEG 2100 clinical device was used, with 19 scalp electrodes (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2) according to the 10-20 international system and 2 reference electrodes (A1 and A2) placed on the mastoids for impendance check, according to the manual of the device. Each recording was performed according to the clinical protocol with participants being in a sitting position having their eyes closed. Before the initialization of each recording, the skin impedance value was ensured to be below 5k?. The sampling rate was 500 Hz with 10uV/mm resolution. The recording montages were anterior-posterior bipolar and referential montage using Cz as the common reference. The referential montage was included in this dataset. The recordings were received under the range of the following parameters of the amplifier: Sensitivity: 10uV/mm, time constant: 0.3s, and high frequency filter at 70 Hz. Each recording lasted approximately 13.5 minutes for AD group (min=5.1, max=21.3), 12 minutes for FTD group (min=7.9, max=16.9) and 13.8 for CN group (min=12.5, max=16.5). In total, 485.5 minutes of AD, 276.5 minutes of FTD and 402 minutes of CN recordings were collected and are included in the dataset.

Preprocessing: The EEG recordings were exported in .eeg format and are transformed to BIDS accepted .set format for the inclusion in the dataset. Automatic annotations of the Nihon Kohden EEG device marking artifacts (muscle activity, blinking, swallowing) have not been included for language compatibility purposes (If this is an issue, please use the preprocessed dataset in Folder: derivatives). The unprocessed EEG recordings are included in folders named: sub-0XX. Folders named sub-0XX in the subfolder derivatives contain the preprocessed and denoised EEG recordings. The preprocessing pipeline of the EEG signals is as follows. First, a Butterworth band-pass filter 0.5-45 Hz was applied and the signals were re-referenced to A1-A2. Then, the Artifact Subspace Reconstruction routine (ASR) which is an EEG artifact correction method included in the EEGLab Matlab software was applied to the signals, removing bad data periods which exceeded the max acceptable 0.5 second window standard deviation of 17, which is considered a conservative window. Next, the Independent Component Analysis (ICA) method (RunICA algorithm) was performed, transforming the 19 EEG signals to 19 ICA components. ICA components that were classified as “eye artifacts” or “jaw artifacts” by the automatic classification routine “ICLabel” in the EEGLAB platform were automatically rejected. It should be noted that, even though the recording was performed in a resting state, eyes-closed condition, eye artifacts of eye movement were still found at some EEG recordings."""

import base64


def upload_image_to_catbox(image_path: str) -> str:
    """
    Upload an image to Catbox.moe and return the public URL.
    No API key required, files stored indefinitely.

    Args:
        image_path: Path to the image file

    Returns:
        Public URL of the uploaded image
    """
    try:
        with open(image_path, "rb") as f:
            files = {"fileToUpload": f}
            data = {"reqtype": "fileupload"}

            response = requests.post(
                "https://catbox.moe/user/api.php", files=files, data=data, timeout=30
            )

        if response.status_code == 200:
            # Catbox returns just the URL in plain text
            return response.text.strip()
        else:
            raise Exception(f"Catbox upload failed: {response.text}")

    except Exception as e:
        logging.error(f"Error uploading to Catbox: {e}")
        raise


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


# Create images directory if it doesn't exist
os.makedirs("./images", exist_ok=True)


class InitialQCResult(BaseModel):
    skip_stages: List[str]
    justification: Dict[str, str]


logger.info("Starting EEG preprocessing pipeline...")


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

    timeseries_url = upload_image_to_catbox("./images/raw_timeseries.jpg")

    input_psd_fig = state["input_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    input_psd_fig.savefig("./images/psd_plot.jpg")

    psd_url = upload_image_to_catbox("./images/psd_plot.jpg")

    input_sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    input_sensors_fig.savefig("./images/sensors.jpg")

    sensors_url = upload_image_to_catbox("./images/sensors.jpg")

    try:
        prompt = """ You are a helpful assistant for EEG data analysis. I will give you an image of raw EEG data and its power spectrum density (PSD) and sensor layout. Based on these images, I want you to determine which preprocessing steps are necessary for this data. The possible preprocessing steps are: bandpass filtering, notch filtering, bad channel identification, slow drift correction, and ICA-based artifact removal. For each step, decide whether it is necessary or can be skipped for this dataset. Provide a justification for each decision based on the features observed in the raw EEG plot, the PSD, and the sensor layout. Consider factors such as noise levels, presence of line noise peaks in the PSD, flat channels in the raw data, signs of slow drifts, and any other relevant observations that would inform your decisions about which preprocessing steps are needed.
"""
        messages = create_reasoning_messages(
            user_prompt=prompt, image_url=timeseries_url
        )  # TODO - Add current eeg plot url
        ## Ajout du psd plot pour le notch filtering
        messages[-1]["content"] += [
            {
                "type": "text",
                "text": "Here is the power spectrum of the data. Does it show signs of line noise or other issues that would inform preprocessing decisions?",
            },
            {"type": "image_url", "image_url": psd_url},
        ]
        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=InitialQCResult,
            temperature=0.1,
        )
        result = chat_response.choices[0].message.parsed
        logger.info(f"Initial QC result: {result}")
        state["skip_stage"] = result.skip_stages
        state.justification["initial_qc"] = result.justification

    except Exception as e:
        logging.error(f"Error in initial QC: {e}")
        state["errors"].append(f"Initial QC failed: {str(e)}")
        state["skip_stage"] = []

    state["output_raw"] = state["input_raw"]
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


class BandpassFilterSettings(BaseModel):
    l_freq: float = Field(
        description="Low cutoff frequency in Hz for bandpass filtering. Example: 0.5."
    )
    h_freq: float = Field(
        description="High cutoff frequency in Hz for bandpass filtering. Example: 45.0."
    )


def bandpass_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Bandpass filtering agent.
    Applies bandpass filter to the data if not skipped.
    """
    logging.info(f"[BANDPASS FILTERING] Processing subject {state['subject_id']}")

    if "bandpass_filtering" in state["skip_stage"]:
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
            Estimate the optimal bandpass filter settings (low and high cutoff frequencies) for preprocessing this EEG data.
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
            tool_choice="any",
        )

        # Check if tool_calls is available
        if not chat_response.choices[0].message.tool_calls:
            logging.error("No tool calls returned from the model")
            state["errors"].append(
                "Bandpass filtering failed: No tool calls from model"
            )
            return state

        # while chat_response.choices[0].message.tool_calls:
        if chat_response.choices[0].message.content:
            logging.info(f"Model response: {chat_response.choices[0].message.content}")
        tool_call = chat_response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_to_call = names_to_functions[function_name]
        arguments = json.loads(tool_call.function.arguments)
        raw_filtered = function_to_call(raw=state["input_raw"], **arguments)
        logger.info(f"Applied bandpass filter with settings: {arguments}")
        logger.info(f"Bandpass filtering completed. Filtered data has {raw_filtered}")
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
        logging.info(f"Bandpass filtering justification: {justification}")
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

    if "bad_channel_identification" in state["skip_stage"]:
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

    timeseries_url = upload_image_to_catbox("./images/raw_timeseries.jpg")

    input_psd_fig = state["input_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    input_psd_fig.savefig("./images/psd_plot.jpg")

    input_sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    input_sensors_fig.savefig("./images/sensors.jpg")

    try:
        prompt = """You are a helpful assistant for EEG data analysis. I will give you an image of EEG channels. 
                I want you to analyze the plot and identify which channels should be removed. To answer, here are the rule to identify bad channels:
                - Channels with flat line or almost flat line across the entire recording are likely bad channels and should be tagged as removed.
                Also do a brief justification for your answer."""

        messages = create_reasoning_messages(prompt, timeseries_url)

        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=BadChannelAnalysis,
            temperature=0.1,
        )

        result = chat_response.choices[0].message.parsed
        logger.info(f"Bad channel identification result: {result}")
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


def notch_filter(
    raw: mne.io.Raw, freqs: List[float], picks: List[str] = None
) -> mne.io.Raw:
    """Apply notch filter to the raw data at specified frequencies"""
    return raw.notch_filter(freqs=freqs, picks=picks)


names_to_functions["notch_filter"] = notch_filter

notch_filtering_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "notch_filter",
            "description": "Apply notch filter to the data at specified frequencies.",
            "parameters": {
                "type": "object",
                "required": ["freqs"],
                "properties": {
                    "freqs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Frequencies in Hz to remove, for example [50] or [60, 120, 180].",
                    },
                    "picks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": 'Channel names to include, for example ["Fz", "Cz", "Pz"].',
                    },
                },
            },
        },
    }
]


def notch_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Notch filtering agent.
    Applies notch filter to the data if not skipped.
    """
    logging.info(f"[NOTCH FILTERING] Processing subject {state['subject_id']}")

    if "notch_filtering" in state["skip_stage"]:
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

    psd_url = upload_image_to_catbox("./images/psd_plot.jpg")

    input_sensors_fig = state["input_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    input_sensors_fig.savefig("./images/sensors.jpg")

    try:
        available_channels = state["input_raw"].ch_names

        prompt = f"""Role: Decide if narrow-band notch is needed; choose frequencies and channels, verify expected effect.
Use notch only if: PSD shows a clear narrow 50/60 peak inside the kept analysis band.
Don’t notch if: no narrow peak OR mains already outside passband OR contamination is broadband.
Frequency selection: pick exactly the observed peak (50 or 60). Add harmonics only if clearly present and inside kept band.
Channels: apply to channels that actually carry line noise (or all EEG if clearly widespread). Here is the list of available channels: {', '.join(available_channels)}.
."""
        messages = create_reasoning_messages(prompt, psd_url)
        chat_response = client.chat.complete(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            temperature=0.1,
            tools=notch_filtering_agent_tools,
            tool_choice="any",
        )
        result = chat_response.choices[0].message.content
        logger.info(f"Notch filtering decision result: {result}")

        if not chat_response.choices[0].message.tool_calls:
            logger.info("No notch filter needed based on model decision.")
            state["output_raw"] = state["input_raw"]
            state["justification"]["notch_filtering"] = result
            return state
        tool_call = chat_response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_to_call = names_to_functions[function_name]
        arguments = json.loads(tool_call.function.arguments)
        raw_notched = function_to_call(**arguments, raw=state["input_raw"])
        state["output_raw"] = raw_notched
        state["justification"]["notch_filtering"] = result.justification.get(
            "notch_filtering", "No justification provided."
        )
        return state
    except Exception as e:
        logger.error(f"Error in notch filtering decision: {e}")
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
    timeseries_url = upload_image_to_catbox("./images/raw_timeseries.jpg")

    try:
        prompt = """You are a helpful assistant for EEG data analysis. I will give you an image of an EEG plot. I want you to analyze the plot and say whether the data shows signs of slow drifts or not. In raw EEG recordings, slow drifts appear as a gradual upward or downward shift in the signal baseline across channels. Respond by giving a probability between 0 and 1. 1 means the data is very likely to show slow drifts, 0 means it is very unlikely. Add a brief justification for your answer."""
        messages = create_reasoning_messages(prompt, timeseries_url)
        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=EEGSlowDriftAnalysis,
        )
        result = chat_response.choices[0].message.parsed
        logger.info(f"Slow drift analysis result: {result}")
        state["slow_drift_probability"] = result.slow_drift_probability
        state["justification"]["slow_drift"] = result.justification

        if result.slow_drift_probability > SLOW_DRIFT_THRESHOLD:
            logger.info(
                f"Data shows signs of slow drifts with probability {result.slow_drift_probability}."
            )
            raw_corrected = apply_slow_drift_correction(state["input_raw"])
            state["output_raw"] = raw_corrected
        else:
            logger.info(
                f"Data does not show strong signs of slow drifts (probability {result.slow_drift_probability}). Skipping slow drift correction."
            )
            state["output_raw"] = state["input_raw"]
        state["current_stage"] = "bad_channel_identification"
    except Exception as e:
        logger.error(f"Error in slow drift analysis: {e}")
        state["errors"].append(f"Slow drift analysis failed: {str(e)}")
        return state
    return state


def resampling(state: EEGPipelineState) -> EEGPipelineState:
    """[TBC] Resampling agent."""
    logger.info(f"[RESAMPLING] Processing subject {state['subject_id']}")
    # state["input_raw"] = state["output_raw"]
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
    logger.info(f"[ICA FITTING] Processing subject {state['subject_id']}")
    n_channels = len(state["output_raw"].ch_names)

    n_components = min(n_channels - 1, 20)

    ica = ICA(n_components=n_components, random_state=97)
    ica.fit(state["output_raw"])
    return ica


def ica_discrimination_agent(state: EEGPipelineState, ica: ICA) -> EEGPipelineState:
    """
    [TBC] ICA discrimination agent.
    Identifies which ICA components to remove based on the analysis of ICA plots.
    """
    logger.info(f"[ICA DISCRIMINATION] Processing subject {state['subject_id']}")
    if "ica" in state["skip_stage"]:
        logger.info(
            "[ICA DISCRIMINATION] Skipping ICA discrimination based on initial QC."
        )
        state["justification"][
            "ica_discrimination"
        ] = "Skipped based on initial QC assessment."
        return state

    # Generate ICA components plot
    input_ica_fig = ica.plot_components(
        picks=range(min(20, ica.n_components_)), show=False
    )
    input_ica_fig.savefig("./images/ica_components.jpg")
    ica_components_url = upload_image_to_catbox("./images/ica_components.jpg")
    ica_graph_fig = ica.plot_sources(
        state["input_raw"], picks=range(min(20, ica.n_components_)), show=False
    )
    ica_graph_fig.savefig("./images/ica_properties.jpg")
    ica_properties_url = upload_image_to_catbox("./images/ica_properties.jpg")
    try:

        messages = create_reasoning_messages(
            user_prompt="""
                You are a helpful assistant for EEG data analysis. I will give you an image of ICA plots. I want you to analyze the plot and identify which ICA channels should be removed. To answer, here are some rules to identify bad ICA components:
                - Vertical eye movement components will contain blinks in the data
                - Horizontal eye movement components will look like step functions
                - The pattern generated by the heart is very typical and is known as a QRS complex (it looks like a sharp peak followed by a smaller inverted peak)
                - Muscle artifacts typically have a high-frequency pattern and are often localized to specific channels, especially those near the face and neck. They can appear as bursts of high-frequency activity in the ICA components
                - Strong peak in power spectrum at either 50Hz or 60Hz
                - If an ICA component looks like the EOG signal (which is at the bottom of the plot) it is likely an eye movement artifact and should be removed.
                Respond by providing a list of ICA channels to remove based on the analysis and the rules. I want only to identify the channels associated to these rules.
                Also do a brief justification for your answer.""",
            image_url=ica_properties_url,
        )
        messages[-1]["content"] += [
            {
                "type": "text",
                "text": "Here is the components of the ICA components. Does it show features that would inform your decision on which ICA components to remove?",
            },
            {"type": "image_url", "image_url": ica_components_url},
        ]
        chat_response = client.chat.parse(
            model=model,
            messages=messages,
            prompt_mode="reasoning",
            response_format=ICAAnalysis,
            temperature=0.1,
        )
        result = chat_response.choices[0].message.parsed
        logger.info(f"ICA discrimination result: {result}")
        state["ica_channels_to_remove"] = result.ica_channels_to_remove
        assert(type(state["ica_channels_to_remove"]) == list and len(state["ica_channels_to_remove"])>0 )
        state["ica_justification"] = result.justification
        assert(type(state["ica_justification"]) == str)
        return state
    except Exception as e:
        logging.error(f"Error in ICA discrimination: {e}")
        state["errors"].append(f"ICA discrimination failed: {str(e)}")
        return state


def apply_ica_correction(state: EEGPipelineState, ica: ICA) -> EEGPipelineState:
    """Apply ICA correction by removing identified components"""
    ica_channels = state["ica_channels_to_remove"]  
    ica.exclude = [int(comp.replace('ICA', '')) - 1 for comp in ica_channels]
    raw_corrected = state["input_raw"].copy()
    ica.apply(raw_corrected)
    state["output_raw"] = raw_corrected
    return state


def interpolate_bad_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """Interpolate bad channels in the raw object"""
    raw.interpolate_bads(reset_bads=True)
    return raw


# def final_qc_agent(state: EEGPipelineState) -> EEGPipelineState:
#     """[TBC] Final quality control agent."""
#     logging.info(f"[FINAL QC] Processing subject {state['subject_id']}")

#     state["input_raw"] = state["output_raw"]
#     state["output_raw"] = None  # Reset output raw for this stage

#     timeseries_fig = state["input_raw"].plot(
#         duration=5, n_channels=30, scalings="auto", show_scrollbars=False
#     )  # Generate raw EEG plot without displaying
#     timeseries_fig.savefig("./images/final_qc_timeseries.jpg")
#     timeseries_url = upload_image_to_catbox("./images/final_qc_timeseries.jpg")
#     psd_fig = state["input_raw"].plot_psd(
#         fmax=50
#     )  # Generate PSD plot without displaying
#     psd_fig.savefig("./images/final_qc_psd.jpg")

#     sensors_fig = state["input_raw"].plot_sensors(
#         show_names=True
#     )  # Generate sensor layout plot without displaying
#     sensors_fig.savefig("./images/final_qc_sensors.jpg")

#     try:
#         prompt = """You are the Final QC Agent in an automated EEG preprocessing pipeline for continuous EEG.

# Your role is to inspect the post-processing EEG output and determine whether the preprocessing pipeline was successful.

# You evaluate the cleaned data after the applied steps, which may include:
# - bandpass filtering
# - bad channel handling
# - notch filtering
# - ICA-based artifact removal

# You do not choose preprocessing parameters and you do not decide which agents to run.
# Your task is to verify whether the final output is acceptable, whether artifacts were appropriately reduced, and whether the cleaned data still look physiologically plausible.

# ## Your objectives
# You must:
# 1. inspect the final cleaned EEG time-series plots
# 2. inspect the final cleaned EEG PSD plots
# 3. compare the cleaned output to the expected effects of the preprocessing steps
# 4. determine whether the preprocessing appears successful
# 5. identify any remaining problems
# 6. flag signs of over-cleaning, distortion, or unresolved artifacts

# ## Inputs you may receive
# You may receive:
# - final cleaned EEG time-series plots
# - final cleaned EEG PSD plots
# - summary of the preprocessing steps that were applied
# - list of bad channels that were removed / interpolated / flagged
# - list of ICA components removed
# - intended analysis context:
#   - ERP
#   - resting-state eyes closed / open
#   - BCI / sensorimotor rhythm
#   - sleep
#   - clinical continuous EEG
#   - other continuous EEG context
# - intended analysis passband or kept frequency range

# ## Core decision principle
# Your role is to decide whether the final cleaned data are:
# - Acceptable
# - Acceptable with cautions
# - Needs review / reprocessing
# - Poor / failed preprocessing

# You must balance two goals:
# 1. artifact reduction
# 2. signal preservation

# Good preprocessing should reduce noise and artifacts without destroying plausible neural signal.

# ## What to inspect in the final time series
# Look for:
# - whether the traces look cleaner than before, if before/after context is available
# - whether obvious line-noise, drift, dropouts, or unstable channels remain
# - whether large blink, eye movement, ECG, or muscle bursts still dominate the signal
# - whether the waveform now looks unnaturally flattened, distorted, or over-cleaned
# - whether many channels still look globally noisy
# - whether remaining abnormalities are isolated or widespread

# ## What to inspect in the final PSD
# Look for:
# - whether narrow 50/60 Hz peaks have been appropriately reduced, if notch filtering was applied
# - whether broad spectral shape still looks physiologically plausible
# - whether excessive low-frequency drift has been reduced, if filtering was intended to remove it
# - whether excessive high-frequency contamination has been reduced, if filtering / ICA was meant to reduce it
# - whether the spectrum shows signs of over-filtering, spectral holes, or broad distortion
# - whether any channels still show outlier spectra suggesting unresolved bad channels

# ## Step-specific expectations

# ### 1. After bandpass filtering
# Expected:
# - reduced unwanted low-frequency drift and/or high-frequency contamination
# - preservation of the intended analysis band
# - no obvious broad spectral distortion beyond the intended passband
# - no implausible flattening of the signal

# Possible concerns:
# - overly aggressive high-pass filtering may distort slow activity
# - overly aggressive low-pass filtering may remove relevant high-frequency information
# - traces may look unnaturally smoothed or altered if filtering was too strong

# ### 2. After bad channel handling
# Expected:
# - obvious bad channels should no longer dominate the montage
# - the remaining channels should look more consistent with one another
# - no unresolved flat, clipped, dropout-heavy, or grossly noisy channels should remain if they were supposed to be addressed

# Possible concerns:
# - clearly bad channels still visible
# - too many channels still appear abnormal
# - widespread poor data quality suggesting a bad recording rather than isolated bad sensors

# ### 3. After notch filtering
# Expected:
# - the narrow 50/60 Hz spike, and only the intended harmonics if targeted, should be reduced
# - nearby frequencies should remain mostly preserved
# - time traces should not look radically transformed

# Possible concerns:
# - no real reduction of the narrow spectral line
# - broad reshaping of nearby frequencies
# - dramatic waveform changes suggesting over-filtering or unnecessary notching

# ### 4. After ICA artifact removal
# Expected:
# - reduction of stereotyped artifacts such as:
#   - blinks
#   - eye movements
#   - ECG
#   - muscle bursts
# - preserved overall physiological structure of the EEG
# - no obvious removal of large amounts of plausible neural signal

# Possible concerns:
# - blink / EOG / ECG / muscle patterns still strongly present
# - the data now look unnaturally attenuated or distorted
# - frontal activity appears excessively erased
# - too much signal seems removed relative to the expected artifact reduction

# ## Main artifact questions to answer
# In the final cleaned data, determine whether the following are:
# - resolved
# - partially resolved
# - still present
# - possibly over-corrected

# Assess:
# - slow drift
# - bad channels
# - line noise
# - blink / EOG contamination
# - ECG contamination
# - muscle contamination
# - broad nonstationary noise

# ## How to judge overall success

# ### Acceptable
# Use this when:
# - major targeted artifacts are reduced
# - the PSD looks plausible for the intended analysis
# - no major residual artifacts dominate the recording
# - no strong evidence of over-cleaning is present

# ### Acceptable with cautions
# Use this when:
# - preprocessing mostly worked
# - some residual artifacts or uncertainties remain
# - the data may still be usable, but with caveats

# ### Needs review / reprocessing
# Use this when:
# - important artifacts remain
# - bad channels still appear unresolved
# - notch or bandpass effects look questionable
# - ICA removal appears incomplete or possibly too aggressive
# - the output may be usable only after manual review or adjustment

# ### Poor / failed preprocessing
# Use this when:
# - the cleaned data still look dominated by artifacts
# - preprocessing introduced major distortion
# - signal preservation appears poor
# - the output is not trustworthy without substantial reprocessing

# ## Required output format

# ### 1. Overall verdict
# Choose one:
# - Acceptable
# - Acceptable with cautions
# - Needs review / reprocessing
# - Poor / failed preprocessing

# ### 2. Summary of final data quality
# Briefly summarize:
# - overall cleanliness of the time series
# - overall plausibility of the PSD
# - whether the cleaned output appears suitable for the intended analysis

# ### 3. Step-by-step QC assessment
# For each applied preprocessing step, report:
# - Step:
# - Expected effect:
# - Observed effect:
# - Verdict: Successful / Partially successful / Unclear / Unsuccessful

# Steps to assess if present:
# - Bandpass filtering
# - Bad channel handling
# - Notch filtering
# - ICA artifact removal

# ### 4. Residual artifact assessment
# For each of the following, report:
# - slow drift: resolved / partial / present / over-corrected
# - bad channels: resolved / partial / present
# - line noise: resolved / partial / present / over-corrected
# - blink / EOG: resolved / partial / present / over-corrected
# - ECG: resolved / partial / present / over-corrected
# - muscle artifact: resolved / partial / present / over-corrected
# - broad nonstationary noise: resolved / partial / present

# ### 5. Signs of over-processing
# Explicitly state whether there is evidence of:
# - excessive filtering
# - spectral distortion
# - unnatural flattening or attenuation
# - excessive ICA correction
# - loss of plausible neural signal

# ### 6. Confidence and cautions
# Report:
# - confidence: High / Medium / Low
# - remaining concerns
# - whether human review is recommended

# ## Behavioral rules
# - Judge preprocessing success, not preprocessing intention.
# - Always balance artifact removal against signal preservation.
# - Do not assume “cleaner” automatically means “better.”
# - Be conservative when declaring success if important artifacts remain.
# - Explicitly flag signs of over-cleaning or distortion.
# - Base all conclusions only on the provided plots and preprocessing summary.
# - If evidence is mixed, say so clearly.
# """
#         messages = create_reasoning_messages(prompt, timeseries_url)
#         messages += [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Here is the power spectrum of the final preprocessed data. Analyze it and consider if there are any remaining issues that could be addressed with further preprocessing steps.",
#                     },
#                     {"type": "image_url", "image_url": psd_url},
#                     {"type": "image_url", "image_url": final_qc_sensors_url},
#                 ],
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Here is the initial raw data plots for comparison. Analyze it in conjunction with the final preprocessed data to assess the effectiveness of the preprocessing steps and identify any remaining issues.",
#                     },
#                     {"type": "image_url", "image_url": "./images/raw_timeseries.jpg"},
#                     {"type": "image_url", "image_url": "./images/raw_psd.jpg"},
#                     {"type": "image_url", "image_url": "./images/raw_sensors.jpg"},
#                 ],
#             },
#         ]
#         chat_response = client.chat.parse(
#             model=model,
#             messages=messages,
#             prompt_mode="reasoning",
#             response_format=InitialQCResult,
#             temperature=0.1,
#         )
#         state["final_qc_assessment"] = chat_response.choices[0].message.content
#     except Exception as e:
#         logging.error(f"Error in final QC assessment: {e}")
#         state["errors"].append(f"Final QC assessment failed: {str(e)}")

#     return state


if __name__ == "__main__":
    raw = read_raw_bids(bids_path=bids_path)
    raw.pick("all").load_data()

    current_stage = "initial_qc"
    state = EEGPipelineState(
        subject_id="001",  # Initialize with subject ID from bids_path
        current_stage=current_stage,
        input_raw=raw,
        output_raw=None,
        skip_stage=[],
        justification={},
        errors=[],
        experiment_metadata={"experiment_context": ALZEIMER_EXPERIMENT_CONTEXT},
        bad_channels=[],
        slow_drift_probability=None,
        ica_channels_to_remove=None,
        ica_justification=None,
        final_qc_assessment=None,
    )

    pipeline_states = {}  # To store state after each stage for debugging and analysis

    state = initial_qc_agent(state)
    SKIP_STAGE = state["skip_stage"]
    logger.info(f"Initial QC completed. Stages to skip: {SKIP_STAGE}")
    if SKIP_STAGE:
        logger.info(f"Skipping stages: {SKIP_STAGE}")
    else:
        logger.info("No stages to skip. Proceeding with full pipeline.")
    if "bandpass_filtering" not in SKIP_STAGE:
        state = bandpass_filtering_agent(state)
    logger.info(
        f"Bandpass filtering completed. Justification: {state['justification'].get('bandpass_filtering', 'No justification provided.')}"
    )

    if "bad_channel_identification" not in SKIP_STAGE:
        state = bad_channel_identifier_agent(state)
    logger.info(
        f"Bad channel identification completed. Justification: {state['justification'].get('bad_channel_identification', 'No justification provided.')}"
    )
    state["output_raw"] = annotate_bad_channels(
        state["input_raw"], state["bad_channels"]
    )
    if "notch_filtering" not in SKIP_STAGE:
        state = notch_filtering_agent(state)
    logger.info(
        f"Notch filtering decision completed. Justification: {state['justification'].get('notch_filtering', 'No justification provided.')}"
    )
    if "ica" not in state["skip_stage"]:
        state = resampling(state)
        state = prepare_ica_copy(state)
        ica = apply_ica(state)
        state = ica_discrimination_agent(state, ica)
        state = apply_ica_correction(state, ica)
    logger.info(
        f"ICA correction completed. Justification: {state.get('ica_justification', 'No justification provided.')}"
    )

    if state["bad_channels"]:
        state["output_raw"] = interpolate_bad_channels(state["output_raw"])
    timeseries_fig = state["output_raw"].plot(
        duration=5, n_channels=30, scalings="auto", show_scrollbars=False
    )  # Generate raw EEG plot without displaying
    timeseries_fig.savefig("./images/final_timeseries.jpg")
    psd_fig = state["output_raw"].plot_psd(
        fmax=50
    )  # Generate PSD plot without displaying
    psd_fig.savefig("./images/final_psd.jpg")
    sensors_fig = state["output_raw"].plot_sensors(
        show_names=True
    )  # Generate sensor layout plot without displaying
    sensors_fig.savefig("./images/final_sensors.jpg")
