"""
LangGraph workflow for EEG preprocessing pipeline.
Orchestrates sequential processing with conditional logic and retry mechanisms.
"""
import logging
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from pipeline_state import EEGPipelineState, PipelineStage
from agents import (
    initial_qc_agent,
    notch_filtering_agent,
    validation_agent,
    bad_channel_identifier_agent,
    optional_notch_filtering_agent,
    slow_drift_detector_agent,
    slow_drift_corrector_agent,
    ica_application_agent,
    bad_ica_detector_agent,
    stage_qc_agent,
    interpolation_agent,
    final_qc_agent,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3


# ============================================================================
# WRAPPER NODES FOR VALIDATION AGENTS
# ============================================================================

def notch_validation_node(state: EEGPipelineState) -> EEGPipelineState:
    """Validate notch filtering results"""
    return validation_agent(state, "notch_filter")


def optional_notch_validation_node(state: EEGPipelineState) -> EEGPipelineState:
    """Validate optional notch filtering results"""
    return validation_agent(state, "optional_notch")


def slow_drift_validation_node(state: EEGPipelineState) -> EEGPipelineState:
    """Validate slow drift correction results"""
    return validation_agent(state, "slow_drift")


# ============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# ============================================================================

def should_skip_notch_filtering(state: EEGPipelineState) -> Literal["apply", "skip"]:
    """Determine if notch filtering should be skipped"""
    if PipelineStage.NOTCH_FILTERING in state["skip_stages"]:
        return "skip"
    return "apply"


def should_retry_notch_filtering(state: EEGPipelineState) -> Literal["retry", "proceed"]:
    """Determine if notch filtering should be retried after validation failure"""
    validation = state.get("notch_filter_validation")
    if validation and not validation["passed"] and state["notch_filter_retries"] < MAX_RETRIES:
        return "retry"
    return "proceed"


def should_retry_optional_notch(state: EEGPipelineState) -> Literal["retry", "proceed"]:
    """Determine if optional notch filtering should be retried after validation failure"""
    validation = state.get("optional_notch_validation")
    if validation and not validation["passed"] and state["optional_notch_retries"] < MAX_RETRIES:
        return "retry"
    return "proceed"


def should_correct_slow_drift(state: EEGPipelineState) -> Literal["correct", "skip"]:
    """Determine if slow drift correction is needed based on detection probability"""
    probability = state.get("slow_drift_probability", 0)
    return "correct" if probability > 0.5 else "skip"


def should_retry_slow_drift(state: EEGPipelineState) -> Literal["retry", "proceed"]:
    """Determine if slow drift correction should be retried after validation failure"""
    validation = state.get("slow_drift_validation")
    if validation and not validation["passed"] and state["slow_drift_retries"] < MAX_RETRIES:
        return "retry"
    return "proceed"


def should_skip_ica(state: EEGPipelineState) -> Literal["apply", "skip"]:
    """Determine if ICA should be skipped"""
    if PipelineStage.ICA_APPLICATION in state["skip_stages"]:
        return "skip"
    return "apply"


def should_retry_ica(state: EEGPipelineState) -> Literal["retry", "proceed"]:
    """Determine if ICA should be rerun based on quality control"""
    validation = state.get("stage_qc_validation")
    if validation and not validation["passed"] and state["ica_qc_retries"] < MAX_RETRIES:
        return "retry"
    return "proceed"


def increment_retry_counter(state: EEGPipelineState, retry_type: str) -> EEGPipelineState:
    """Increment the appropriate retry counter"""
    if retry_type == "notch_filter":
        state["notch_filter_retries"] += 1
        logger.info(f"Retrying notch filter (attempt {state['notch_filter_retries']}/{MAX_RETRIES})")
    elif retry_type == "optional_notch":
        state["optional_notch_retries"] += 1
        logger.info(f"Retrying optional notch filter (attempt {state['optional_notch_retries']}/{MAX_RETRIES})")
    elif retry_type == "slow_drift":
        state["slow_drift_retries"] += 1
        logger.info(f"Retrying slow drift correction (attempt {state['slow_drift_retries']}/{MAX_RETRIES})")
    elif retry_type == "ica":
        state["ica_qc_retries"] += 1
        logger.info(f"Retrying ICA (attempt {state['ica_qc_retries']}/{MAX_RETRIES})")
    return state


# Retry increment nodes
def increment_notch_retry(state: EEGPipelineState) -> EEGPipelineState:
    return increment_retry_counter(state, "notch_filter")


def increment_optional_notch_retry(state: EEGPipelineState) -> EEGPipelineState:
    return increment_retry_counter(state, "optional_notch")


def increment_slow_drift_retry(state: EEGPipelineState) -> EEGPipelineState:
    return increment_retry_counter(state, "slow_drift")


def increment_ica_retry(state: EEGPipelineState) -> EEGPipelineState:
    return increment_retry_counter(state, "ica")


# ============================================================================
# BUILD THE WORKFLOW GRAPH
# ============================================================================

def build_eeg_pipeline_graph() -> StateGraph:
    """
    Build the complete EEG preprocessing pipeline graph with all agents,
    conditional routing, and retry logic.
    """
    workflow = StateGraph(EEGPipelineState)
    
    # Add all agent nodes
    workflow.add_node("initial_qc", initial_qc_agent)
    workflow.add_node("notch_filtering", notch_filtering_agent)
    workflow.add_node("notch_validation", notch_validation_node)
    workflow.add_node("increment_notch_retry", increment_notch_retry)
    workflow.add_node("bad_channel_detection", bad_channel_identifier_agent)
    workflow.add_node("optional_notch_filtering", optional_notch_filtering_agent)
    workflow.add_node("optional_notch_validation", optional_notch_validation_node)
    workflow.add_node("increment_optional_notch_retry", increment_optional_notch_retry)
    workflow.add_node("slow_drift_detection", slow_drift_detector_agent)
    workflow.add_node("slow_drift_correction", slow_drift_corrector_agent)
    workflow.add_node("slow_drift_validation", slow_drift_validation_node)
    workflow.add_node("increment_slow_drift_retry", increment_slow_drift_retry)
    workflow.add_node("ica_application", ica_application_agent)
    workflow.add_node("bad_ica_detection", bad_ica_detector_agent)
    workflow.add_node("stage_qc", stage_qc_agent)
    workflow.add_node("increment_ica_retry", increment_ica_retry)
    workflow.add_node("interpolation", interpolation_agent)
    workflow.add_node("final_qc", final_qc_agent)
    
    # Set entry point
    workflow.set_entry_point("initial_qc")
    
    # ========================================================================
    # DEFINE EDGES (Sequential flow with conditional routing)
    # ========================================================================
    
    # Initial QC -> Notch Filtering (conditional)
    workflow.add_conditional_edges(
        "initial_qc",
        should_skip_notch_filtering,
        {
            "apply": "notch_filtering",
            "skip": "bad_channel_detection"
        }
    )
    
    # Notch Filtering -> Validation
    workflow.add_edge("notch_filtering", "notch_validation")
    
    # Notch Validation -> Retry or Proceed (retry loop)
    workflow.add_conditional_edges(
        "notch_validation",
        should_retry_notch_filtering,
        {
            "retry": "increment_notch_retry",
            "proceed": "bad_channel_detection"
        }
    )
    
    # Increment retry -> back to filtering
    workflow.add_edge("increment_notch_retry", "notch_filtering")
    
    # Bad Channel Detection -> Optional Notch Filtering (always)
    workflow.add_edge("bad_channel_detection", "optional_notch_filtering")
    
    # Optional Notch Filtering -> Validation
    workflow.add_edge("optional_notch_filtering", "optional_notch_validation")
    
    # Optional Notch Validation -> Retry or Proceed (retry loop)
    workflow.add_conditional_edges(
        "optional_notch_validation",
        should_retry_optional_notch,
        {
            "retry": "increment_optional_notch_retry",
            "proceed": "slow_drift_detection"
        }
    )
    
    # Increment retry -> back to optional filtering
    workflow.add_edge("increment_optional_notch_retry", "optional_notch_filtering")
    
    # Slow Drift Detection -> Correction or ICA (conditional)
    workflow.add_conditional_edges(
        "slow_drift_detection",
        should_correct_slow_drift,
        {
            "correct": "slow_drift_correction",
            "skip": "ica_application"
        }
    )
    
    # Slow Drift Correction -> Validation
    workflow.add_edge("slow_drift_correction", "slow_drift_validation")
    
    # Slow Drift Validation -> Retry or Proceed (retry loop)
    workflow.add_conditional_edges(
        "slow_drift_validation",
        should_retry_slow_drift,
        {
            "retry": "increment_slow_drift_retry",
            "proceed": "ica_application"
        }
    )
    
    # Increment retry -> back to correction
    workflow.add_edge("increment_slow_drift_retry", "slow_drift_correction")
    
    # ICA Application -> Bad ICA Detection or Interpolation (conditional)
    workflow.add_conditional_edges(
        "ica_application",
        should_skip_ica,
        {
            "apply": "bad_ica_detection",
            "skip": "interpolation"
        }
    )
    
    # Bad ICA Detection -> Stage QC
    workflow.add_edge("bad_ica_detection", "stage_qc")
    
    # Stage QC -> Retry ICA or Proceed (retry loop)
    workflow.add_conditional_edges(
        "stage_qc",
        should_retry_ica,
        {
            "retry": "increment_ica_retry",
            "proceed": "interpolation"
        }
    )
    
    # Increment retry -> back to ICA application
    workflow.add_edge("increment_ica_retry", "ica_application")
    
    # Interpolation -> Final QC
    workflow.add_edge("interpolation", "final_qc")
    
    # Final QC -> END
    workflow.add_edge("final_qc", END)
    
    return workflow


def compile_pipeline():
    """Compile the workflow graph into an executable pipeline"""
    workflow = build_eeg_pipeline_graph()
    return workflow.compile()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_pipeline(output_path: str = "pipeline_graph.png"):
    """
    Generate a visualization of the pipeline graph.
    Requires graphviz to be installed.
    """
    try:
        workflow = build_eeg_pipeline_graph()
        compiled = workflow.compile()
        
        # Generate visualization
        graph_image = compiled.get_graph().draw_mermaid_png()
        
        with open(output_path, "wb") as f:
            f.write(graph_image)
        
        logger.info(f"Pipeline visualization saved to {output_path}")
        return graph_image
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        logger.info("Install graphviz and pygraphviz to enable visualization")
        return None


def get_pipeline_summary() -> Dict[str, Any]:
    """Get a summary of the pipeline structure"""
    return {
        "stages": [stage.value for stage in PipelineStage],
        "max_retries": MAX_RETRIES,
        "retry_stages": [
            "notch_filtering",
            "optional_notch_filtering",
            "slow_drift_correction",
            "ica_application"
        ],
        "conditional_stages": [
            "notch_filtering (can be skipped by initial QC)",
            "slow_drift_correction (based on detection probability)",
            "ica_application (can be skipped by initial QC)"
        ],
        "implemented_agents": [
            "bad_channel_identifier",
            "slow_drift_detector",
            "bad_ica_detector"
        ],
        "placeholder_agents": [
            "initial_qc",
            "notch_filtering",
            "validation (all types)",
            "optional_notch_filtering",
            "slow_drift_corrector",
            "ica_application",
            "stage_qc",
            "interpolation",
            "final_qc"
        ]
    }


if __name__ == "__main__":
    # Example: compile and visualize pipeline
    pipeline = compile_pipeline()
    print("Pipeline compiled successfully!")
    
    summary = get_pipeline_summary()
    print("\n=== Pipeline Summary ===")
    for key, value in summary.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")
