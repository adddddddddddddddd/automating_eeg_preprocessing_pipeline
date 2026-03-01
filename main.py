

"""
Main orchestrator for the EEG preprocessing pipeline.
Entry point for processing EEG data through the multimodal agent pipeline.
"""
import logging
import json
from pathlib import Path
from typing import Optional
import argparse

from pipeline_state import create_initial_state, EEGPipelineState
from pipeline_graph import compile_pipeline, get_pipeline_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    subject_id: str,
    raw_eeg_file_path: str,
    output_dir: Optional[str] = None
) -> EEGPipelineState:
    """
    Run the complete EEG preprocessing pipeline for a single subject.
    
    Args:
        subject_id: Subject identifier (e.g., "sub-001")
        raw_eeg_file_path: Path to the raw EEG file
        output_dir: Optional output directory for processed data
        
    Returns:
        Final pipeline state with all processing results
    """
    logger.info(f"Starting EEG preprocessing pipeline for {subject_id}")
    logger.info(f"Input file: {raw_eeg_file_path}")
    
    # Validate input
    if not Path(raw_eeg_file_path).exists():
        raise FileNotFoundError(f"EEG file not found: {raw_eeg_file_path}")
    
    # Create initial state
    initial_state = create_initial_state(subject_id, raw_eeg_file_path)
    
    # Set output directory
    if output_dir:
        initial_state["processed_eeg_file_path"] = str(Path(output_dir) / f"{subject_id}_preprocessed.fif")
    
    # Compile and run pipeline
    logger.info("Compiling pipeline graph...")
    pipeline = compile_pipeline()
    
    logger.info("Running pipeline...")
    final_state = pipeline.invoke(initial_state)
    
    # Log results
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Success: {final_state['pipeline_success']}")
    logger.info(f"Stages completed: {len(final_state['processing_history'])}")
    
    if final_state['bad_channels']:
        logger.info(f"Bad channels removed: {final_state['bad_channels']}")
    
    if final_state['ica_components_to_remove']:
        logger.info(f"ICA components removed: {final_state['ica_components_to_remove']}")
    
    if final_state['errors']:
        logger.warning(f"Errors encountered: {final_state['errors']}")
    
    return final_state


def save_pipeline_report(state: EEGPipelineState, output_path: str):
    """
    Save a detailed report of the pipeline processing.
    
    Args:
        state: Final pipeline state
        output_path: Path to save the JSON report
    """
    report = {
        "subject_id": state["subject_id"],
        "input_file": state["raw_eeg_file_path"],
        "output_file": state["processed_eeg_file_path"],
        "pipeline_success": state["pipeline_success"],
        "processing_history": state["processing_history"],
        "bad_channels": state["bad_channels"],
        "ica_components_removed": state["ica_components_to_remove"],
        "slow_drift_probability": state["slow_drift_probability"],
        "applied_filters": state["applied_filters"],
        "validations": {
            "notch_filter": state["notch_filter_validation"],
            "optional_notch": state["optional_notch_validation"],
            "slow_drift": state["slow_drift_validation"],
            "stage_qc": state["stage_qc_validation"],
            "final_qc": state["final_qc_validation"],
        },
        "retries": {
            "notch_filter": state["notch_filter_retries"],
            "optional_notch": state["optional_notch_retries"],
            "slow_drift": state["slow_drift_retries"],
            "ica_qc": state["ica_qc_retries"],
        },
        "errors": state["errors"],
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Pipeline report saved to {output_path}")


def batch_process_subjects(
    dataset_path: str,
    subject_ids: list[str],
    output_dir: str
):
    """
    Process multiple subjects through the pipeline.
    
    Args:
        dataset_path: Base path to the dataset
        subject_ids: List of subject IDs to process
        output_dir: Output directory for processed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for subject_id in subject_ids:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {subject_id}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            # Construct input path (adjust based on your dataset structure)
            raw_file = Path(dataset_path) / subject_id / "eeg" / f"{subject_id}_task-rest_eeg.fif"
            
            if not raw_file.exists():
                logger.error(f"File not found for {subject_id}: {raw_file}")
                continue
            
            # Run pipeline
            final_state = run_pipeline(
                subject_id=subject_id,
                raw_eeg_file_path=str(raw_file),
                output_dir=str(output_path)
            )
            
            # Save report
            report_path = output_path / f"{subject_id}_report.json"
            save_pipeline_report(final_state, str(report_path))
            
            results.append({
                "subject_id": subject_id,
                "success": final_state["pipeline_success"],
                "errors": final_state["errors"]
            })
            
        except Exception as e:
            logger.error(f"Failed to process {subject_id}: {e}", exc_info=True)
            results.append({
                "subject_id": subject_id,
                "success": False,
                "errors": [str(e)]
            })
    
    # Save batch summary
    summary_path = output_path / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_subjects": len(subject_ids),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }, f, indent=2)
    
    logger.info(f"\nBatch processing complete. Summary saved to {summary_path}")


def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="EEG Preprocessing Pipeline with Multimodal LLM Agents"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Single subject processing
    single_parser = subparsers.add_parser("process", help="Process a single subject")
    single_parser.add_argument("subject_id", help="Subject ID (e.g., sub-001)")
    single_parser.add_argument("input_file", help="Path to raw EEG file")
    single_parser.add_argument("--output-dir", help="Output directory", default="./output")
    single_parser.add_argument("--save-report", action="store_true", help="Save detailed report")
    
    # Batch processing
    batch_parser = subparsers.add_parser("batch", help="Process multiple subjects")
    batch_parser.add_argument("dataset_path", help="Path to dataset directory")
    batch_parser.add_argument("--subjects", nargs="+", help="Subject IDs to process")
    batch_parser.add_argument("--output-dir", help="Output directory", default="./output")
    
    # Show pipeline summary
    summary_parser = subparsers.add_parser("info", help="Show pipeline information")
    
    args = parser.parse_args()
    
    if args.command == "process":
        # Process single subject
        final_state = run_pipeline(
            subject_id=args.subject_id,
            raw_eeg_file_path=args.input_file,
            output_dir=args.output_dir
        )
        
        if args.save_report:
            report_path = Path(args.output_dir) / f"{args.subject_id}_report.json"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            save_pipeline_report(final_state, str(report_path))
    
    elif args.command == "batch":
        # Batch process subjects
        if not args.subjects:
            logger.error("No subjects specified. Use --subjects sub-001 sub-002 ...")
            return
        
        batch_process_subjects(
            dataset_path=args.dataset_path,
            subject_ids=args.subjects,
            output_dir=args.output_dir
        )
    
    elif args.command == "info":
        # Show pipeline information
        summary = get_pipeline_summary()
        print("\n" + "=" * 80)
        print("EEG PREPROCESSING PIPELINE SUMMARY")
        print("=" * 80)
        for key, value in summary.items():
            print(f"\n{key.replace('_', ' ').upper()}:")
            if isinstance(value, list):
                for item in value:
                    print(f"  • {item}")
            else:
                print(f"  {value}")
        print("\n" + "=" * 80)
    
    else:
        # No command specified, show help
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py process sub-001 /path/to/raw_eeg.fif --output-dir ./output")
        print("  python main.py batch /path/to/dataset --subjects sub-001 sub-002 sub-003")
        print("  python main.py info")


if __name__ == "__main__":
    main()

