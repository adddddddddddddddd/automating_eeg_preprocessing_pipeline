"""
Agent implementations for the EEG preprocessing pipeline.
Each agent follows a common interface that takes and returns pipeline state.
"""

import os
import logging
from typing import Dict, Any
from mistralai import Mistral
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from pipeline_state import (
    EEGPipelineState,
    ProcessingDecision,
    ValidationResult,
    PipelineStage,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key) if api_key else None


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================


class ICAAnalysis(BaseModel):
    ica_channels_to_remove: list = Field(
        description="List of ICA channels to remove based on the analysis"
    )
    justification: str = Field(
        description="A brief explanation of the reasoning behind the selected ICA channels"
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
        description="A brief explanation of the reasoning behind the assigned probability"
    )


class BadChannelAnalysis(BaseModel):
    bad_channels_to_remove: list = Field(
        description="List of bad channels to remove based on the analysis"
    )
    justification: str = Field(
        description="A brief explanation of the reasoning behind the selected bad channels"
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


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


def initial_qc_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Initial quality control agent.
    Determines which pipeline stages are necessary based on initial data assessment.
    """
    logging.info(f"[INITIAL QC] Processing subject {state['subject_id']}")

    # TODO: Implement multimodal analysis of raw EEG to determine:
    # - Whether notch filtering is needed
    # - Whether slow drift correction is needed
    # - Whether ICA is needed
    # - Overall data quality

    # Placeholder: proceed with all stages
    state["skip_stages"] = []
    state["current_stage"] = PipelineStage.NOTCH_FILTERING

    logging.info("[INITIAL QC] Placeholder: proceeding with all stages")
    return state


def notch_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Applies notch filtering if determined necessary by planner.
    Selects appropriate filter based on context and few-shot prompting.
    """
    logging.info(f"[NOTCH FILTERING] Processing subject {state['subject_id']}")

    if PipelineStage.NOTCH_FILTERING in state["skip_stages"]:
        logging.info("Skipping notch filtering as per initial QC")
        state["current_stage"] = PipelineStage.BAD_CHANNEL_DETECTION
        return state

    # TODO: Implement filter selection logic
    # - Analyze power spectrum
    # - Detect line noise frequency (50Hz or 60Hz)
    # - Select appropriate notch filter parameters
    # - Apply filter and save result

    # Placeholder
    filter_params = {"frequency": 60, "bandwidth": 2}
    state["applied_filters"].append(
        {"type": "notch", "stage": "primary", "params": filter_params}
    )

    state["current_stage"] = PipelineStage.NOTCH_VALIDATION
    state["notch_filter_retries"] = 0

    logging.info(
        f"[NOTCH FILTERING] Placeholder: applied filter with params {filter_params}"
    )
    return state


def bad_channel_identifier_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    Identifies bad EEG channels that should be removed.
    """
    logging.info(f"[BAD CHANNEL IDENTIFIER] Processing subject {state['subject_id']}")

    if not state.get("current_eeg_plot_url"):
        logging.error("No EEG plot URL available for bad channel detection")
        state["errors"].append("Missing EEG plot for bad channel detection")
        return state

    try:
        prompt = """You are a helpful assistant for EEG data analysis. I will give you an image of EEG channels. 
        I want you to analyze the plot and identify which channels should be removed. To answer, here are the rules to identify bad channels:
        - Channels with flat line or almost flat line across the entire recording are likely bad channels and should be tagged as removed.
        Also do a brief justification for your answer."""

        messages = create_reasoning_messages(prompt, state["current_eeg_plot_url"])

        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=BadChannelAnalysis,
            temperature=0.1,
        )

        result = chat_response.choices[0].message.parsed
        state["bad_channels"] = result.bad_channels_to_remove

        decision = ProcessingDecision(
            stage=PipelineStage.BAD_CHANNEL_DETECTION,
            action="remove_channels",
            parameters={"channels": result.bad_channels_to_remove},
            confidence=None,
            justification=result.justification,
        )
        state["processing_history"].append(decision)

        logging.info(
            f"Identified {len(result.bad_channels_to_remove)} bad channels: {result.bad_channels_to_remove}"
        )

    except Exception as e:
        logging.error(f"Error in bad channel detection: {e}")
        state["errors"].append(f"Bad channel detection failed: {str(e)}")

    state["current_stage"] = PipelineStage.OPTIONAL_NOTCH_FILTERING
    return state


def optional_notch_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Optional notch filtering after bad channel removal.
    Independent from planner agent, always runs after bad channel detection.
    """
    logging.info(f"[OPTIONAL NOTCH FILTERING] Processing subject {state['subject_id']}")

    # TODO: Implement adaptive notch filtering
    # - Reassess power spectrum after bad channel removal
    # - Determine if additional filtering is needed
    # - Apply if necessary

    # Placeholder
    state["current_stage"] = PipelineStage.OPTIONAL_NOTCH_VALIDATION
    state["optional_notch_retries"] = 0

    logging.info(
        "[OPTIONAL NOTCH FILTERING] Placeholder: no additional filtering needed"
    )
    return state


def high_pass_filtering(raw, cutoff_frequency):
    return raw.copy().filter(l_freq=cutoff_frequency, h_freq=None)    

name_to_function_dict = {
    "high_pass_filter": high_pass_filtering,
}

    
    
slow_drift_correcting_tools = [
    {
        "type": "function",
        "function": {
            "name": "high_pass_filter",
            "description": "Apply a high-pass filter to remove slow drift artifacts from EEG data",
            "parameters": {
                "type": "object",
                "properties": {
                    "cutoff_frequency": {
                        "type": "number",
                        "description": "The cutoff frequency for the high-pass filter in Hz.",
                    }
                },
                "required": ["cutoff_frequency"],
            },
        },
    },
]


def slow_drift_detector_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    Detects slow drift artifacts in EEG data.
    Returns probability of slow drift presence.
    """
    logging.info(f"[SLOW DRIFT DETECTOR] Processing subject {state['subject_id']}")

    if not state.get("current_eeg_plot_url"):
        logging.error("No EEG plot URL available for slow drift detection")
        state["errors"].append("Missing EEG plot for slow drift detection")
        return state

    try:
        prompt = """
        You are a helpful assistant for EEG data analysis. I will give you an image of an EEG plot. 
        I want you to analyze the plot and say whether the data shows signs of slow drifts or not. 
        In raw EEG recordings, slow drifts appear as a gradual upward or downward shift in the signal baseline across channels.
        Respond by giving a probability between 0 and 1. 1 means the data is very likely to show slow drifts, 0 means it is very unlikely. 
        Add a brief justification for your answer.
        """

        messages = create_reasoning_messages(prompt, state["current_eeg_plot_url"])

        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=EEGSlowDriftAnalysis,
            tools=[],
        )

        result = chat_response.choices[0].message.parsed
        state["slow_drift_probability"] = result.slow_drift_probability

        decision = ProcessingDecision(
            stage=PipelineStage.SLOW_DRIFT_DETECTION,
            action="detect_slow_drift",
            parameters={"probability": result.slow_drift_probability},
            confidence=result.slow_drift_probability,
            justification=result.justification,
        )
        state["processing_history"].append(decision)

        logging.info(f"Slow drift probability: {result.slow_drift_probability}")

    except Exception as e:
        logging.error(f"Error in slow drift detection: {e}")
        state["errors"].append(f"Slow drift detection failed: {str(e)}")

    # Move to correction if drift detected (threshold > 0.5)
    if state.get("slow_drift_probability", 0) > 0.5:
        state["current_stage"] = PipelineStage.SLOW_DRIFT_CORRECTION
    else:
        state["current_stage"] = PipelineStage.ICA_APPLICATION

    return state


def slow_drift_corrector_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Corrects slow drift artifacts if detected.
    """
    logging.info(f"[SLOW DRIFT CORRECTOR] Processing subject {state['subject_id']}")

    # TODO: Implement drift correction
    # - Apply high-pass filtering or detrending
    # - Select parameters based on detected drift characteristics

    # Placeholder
    state["current_stage"] = PipelineStage.SLOW_DRIFT_VALIDATION
    state["slow_drift_retries"] = 0

    logging.info("[SLOW DRIFT CORRECTOR] Placeholder: drift correction applied")
    return state


def bad_ica_detector_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    Identifies ICA components that should be removed.
    Detects eye movements, heart artifacts, muscle artifacts, and line noise.
    """
    logging.info(f"[BAD ICA DETECTOR] Processing subject {state['subject_id']}")

    if not state.get("ica_components_plot_url"):
        logging.error("No ICA components plot URL available")
        state["errors"].append("Missing ICA components plot")
        return state

    try:
        prompt = """You are a helpful assistant for EEG data analysis. I will give you an image of ICA plots. 
        I want you to analyze the plot and identify which ICA channels should be removed. To answer, here are some rules to identify bad ICA components:
        - Vertical eye movement components will contain blinks in the data
        - Horizontal eye movement components will look like step functions
        - The pattern generated by the heart is very typical and is known as a QRS complex (it looks like a sharp peak followed by a smaller inverted peak)
        - Muscle artifacts typically have a high-frequency pattern and are often localized to specific channels, especially those near the face and neck. They can appear as bursts of high-frequency activity in the ICA components
        - Strong peak in power spectrum at either 50Hz or 60Hz
        - If an ICA component looks like the EOG signal (which is at the bottom of the plot) it is likely an eye movement artifact and should be removed.
        Respond by providing a list of ICA channels to remove based on the analysis and the rules. I want only to identify the channels associated to these rules.
        Also do a brief justification for your answer."""

        messages = create_reasoning_messages(prompt, state["ica_components_plot_url"])

        chat_response = client.chat.parse(
            model="magistral-small-2509",
            messages=messages,
            prompt_mode="reasoning",
            response_format=ICAAnalysis,
            temperature=0.1,
        )

        result = chat_response.choices[0].message.parsed
        state["ica_components_to_remove"] = result.ica_channels_to_remove

        decision = ProcessingDecision(
            stage=PipelineStage.BAD_ICA_DETECTION,
            action="remove_ica_components",
            parameters={"components": result.ica_channels_to_remove},
            confidence=None,
            justification=result.justification,
        )
        state["processing_history"].append(decision)

        logging.info(
            f"Identified {len(result.ica_channels_to_remove)} ICA components to remove: {result.ica_channels_to_remove}"
        )

    except Exception as e:
        logging.error(f"Error in ICA component detection: {e}")
        state["errors"].append(f"ICA detection failed: {str(e)}")

    state["current_stage"] = PipelineStage.STAGE_QC
    return state


def validation_agent(state: EEGPipelineState, validation_type: str) -> EEGPipelineState:
    """
    [TBC] Generic validation agent with hybrid approach.
    Combines statistical metrics with LLM visual analysis.
    """
    logging.info(
        f"[VALIDATION - {validation_type}] Processing subject {state['subject_id']}"
    )

    # TODO: Implement validation logic:
    # 1. Calculate statistical metrics (SNR, variance, etc.)
    # 2. Generate before/after comparison plots
    # 3. Use LLM to assess visual quality
    # 4. Combine metrics and LLM assessment

    # Placeholder: always pass validation
    validation_result = ValidationResult(
        passed=True,
        score=0.85,
        issues=[],
        retry_count=0,
        justification="Placeholder validation - metrics within acceptable range",
    )

    # Update appropriate validation field based on type
    if validation_type == "notch_filter":
        state["notch_filter_validation"] = validation_result
    elif validation_type == "optional_notch":
        state["optional_notch_validation"] = validation_result
    elif validation_type == "slow_drift":
        state["slow_drift_validation"] = validation_result
    elif validation_type == "stage_qc":
        state["stage_qc_validation"] = validation_result
    elif validation_type == "final_qc":
        state["final_qc_validation"] = validation_result

    logging.info(f"[VALIDATION - {validation_type}] Placeholder: validation passed")
    return state


def ica_application_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Applies ICA decomposition.
    Selects ICA method and parameters based on context and few-shot prompting.
    """
    logging.info(f"[ICA APPLICATION] Processing subject {state['subject_id']}")

    if PipelineStage.ICA_APPLICATION in state["skip_stages"]:
        logging.info("Skipping ICA as per initial QC")
        state["current_stage"] = PipelineStage.INTERPOLATION
        return state

    # TODO: Implement ICA application
    # - Select ICA algorithm (FastICA, Infomax, Picard)
    # - Set number of components
    # - Apply ICA
    # - Generate component plots

    # Placeholder
    state["current_stage"] = PipelineStage.BAD_ICA_DETECTION

    logging.info("[ICA APPLICATION] Placeholder: ICA decomposition completed")
    return state


def stage_qc_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Quality control after ICA component removal.
    Checks if results are acceptable, may trigger ICA re-run (max 3 retries).
    """
    logging.info(f"[STAGE QC] Processing subject {state['subject_id']}")

    # TODO: Implement QC logic
    # - Assess quality of ICA artifact removal
    # - Check for remaining artifacts
    # - Decide if re-run is needed

    # Placeholder: always pass
    validation_result = ValidationResult(
        passed=True,
        score=0.9,
        issues=[],
        retry_count=state["ica_qc_retries"],
        justification="Placeholder: ICA results acceptable",
    )
    state["stage_qc_validation"] = validation_result
    state["current_stage"] = PipelineStage.INTERPOLATION

    logging.info("[STAGE QC] Placeholder: ICA quality check passed")
    return state


def interpolation_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Interpolates removed bad channels.
    Selects interpolation method and parameters based on signal characteristics.
    """
    logging.info(f"[INTERPOLATION] Processing subject {state['subject_id']}")

    # TODO: Implement interpolation
    # - Select interpolation method (spherical spline, nearest neighbor, etc.)
    # - Determine parameters based on:
    #   * Number and location of bad channels
    #   * Overall signal characteristics
    #   * Montage/electrode layout

    # Placeholder
    if state["bad_channels"]:
        logging.info(
            f"[INTERPOLATION] Placeholder: would interpolate {len(state['bad_channels'])} channels"
        )

    state["current_stage"] = PipelineStage.FINAL_QC

    return state


def final_qc_agent(state: EEGPipelineState) -> EEGPipelineState:
    """
    [TBC] Overall quality control of the entire pipeline.
    Final check before marking pipeline as complete.
    """
    logging.info(f"[FINAL QC] Processing subject {state['subject_id']}")

    # TODO: Implement comprehensive QC
    # - Check all processing steps were completed
    # - Assess overall signal quality
    # - Verify no critical issues remain
    # - Generate final report

    # Placeholder: mark as complete
    validation_result = ValidationResult(
        passed=True,
        score=0.92,
        issues=[],
        retry_count=0,
        justification="Placeholder: All processing steps completed successfully",
    )
    state["final_qc_validation"] = validation_result
    state["pipeline_completed"] = True
    state["pipeline_success"] = True

    logging.info("[FINAL QC] Pipeline completed successfully")
    return state
