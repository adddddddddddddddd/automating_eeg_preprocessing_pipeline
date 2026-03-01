"""
State schema for the EEG preprocessing pipeline.
Defines the data structure passed between agents in the LangGraph workflow.
"""
from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum


class PipelineStage(str, Enum):
    """Enumeration of pipeline stages"""
    INITIAL_QC = "initial_qc"
    NOTCH_FILTERING = "notch_filtering"
    NOTCH_VALIDATION = "notch_validation"
    BAD_CHANNEL_DETECTION = "bad_channel_detection"
    OPTIONAL_NOTCH_FILTERING = "optional_notch_filtering"
    OPTIONAL_NOTCH_VALIDATION = "optional_notch_validation"
    SLOW_DRIFT_DETECTION = "slow_drift_detection"
    SLOW_DRIFT_CORRECTION = "slow_drift_correction"
    SLOW_DRIFT_VALIDATION = "slow_drift_validation"
    ICA_APPLICATION = "ica_application"
    BAD_ICA_DETECTION = "bad_ica_detection"
    STAGE_QC = "stage_qc"
    INTERPOLATION = "interpolation"
    FINAL_QC = "final_qc"


class ProcessingDecision(TypedDict):
    """Decision made by an agent"""
    stage: str
    action: str  # e.g., "skip", "apply_filter", "remove_channels"
    parameters: Optional[Dict[str, Any]]
    confidence: Optional[float]
    justification: str


class ValidationResult(TypedDict):
    """Result from a validation agent"""
    passed: bool
    score: Optional[float]
    issues: List[str]
    retry_count: int
    justification: str


class EEGPipelineState(TypedDict):
    """Complete state for the EEG preprocessing pipeline"""
    
    # Input data
    subject_id: str
    raw_eeg_file_path: str
    
    # Current processing stage
    current_stage: str
    
    # Image URLs for visual analysis (generated plots)
    raw_eeg_plot_url: Optional[str]
    current_eeg_plot_url: Optional[str]
    ica_components_plot_url: Optional[str]
    power_spectrum_plot_url: Optional[str]
    
    # Decisions and actions taken
    skip_stages: List[str]  # Stages marked as unnecessary by planner
    processing_history: List[ProcessingDecision]
    
    # Agent-specific results
    bad_channels: List[str]
    ica_components_to_remove: List[int]
    slow_drift_probability: Optional[float]
    applied_filters: List[Dict[str, Any]]
    
    # Validation tracking
    notch_filter_validation: Optional[ValidationResult]
    optional_notch_validation: Optional[ValidationResult]
    slow_drift_validation: Optional[ValidationResult]
    stage_qc_validation: Optional[ValidationResult]
    final_qc_validation: Optional[ValidationResult]
    
    # Retry counters
    notch_filter_retries: int
    optional_notch_retries: int
    slow_drift_retries: int
    ica_qc_retries: int
    
    # Final output path
    processed_eeg_file_path: Optional[str]
    
    # Error handling
    errors: List[str]
    pipeline_completed: bool
    pipeline_success: bool


def create_initial_state(subject_id: str, raw_eeg_file_path: str) -> EEGPipelineState:
    """Create initial state for the pipeline"""
    return EEGPipelineState(
        subject_id=subject_id,
        raw_eeg_file_path=raw_eeg_file_path,
        current_stage=PipelineStage.INITIAL_QC,
        raw_eeg_plot_url=None,
        current_eeg_plot_url=None,
        ica_components_plot_url=None,
        power_spectrum_plot_url=None,
        skip_stages=[],
        processing_history=[],
        bad_channels=[],
        ica_components_to_remove=[],
        slow_drift_probability=None,
        applied_filters=[],
        notch_filter_validation=None,
        optional_notch_validation=None,
        slow_drift_validation=None,
        stage_qc_validation=None,
        final_qc_validation=None,
        notch_filter_retries=0,
        optional_notch_retries=0,
        slow_drift_retries=0,
        ica_qc_retries=0,
        processed_eeg_file_path=None,
        errors=[],
        pipeline_completed=False,
        pipeline_success=False,
    )
