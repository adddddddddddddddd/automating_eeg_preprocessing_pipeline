# EEG Preprocessing Pipeline with Multimodal LLM Agents

## Overview

This project implements an automated EEG preprocessing pipeline using LangGraph to orchestrate multiple multimodal reasoning agents (Mistral AI models). The pipeline processes EEG data through various stages with intelligent decision-making, validation loops, and retry mechanisms.

## Architecture

### Pipeline Flow

```
Initial QC Agent
    ↓
[Conditional] Notch Filtering Agent ← ─ ─ ┐
    ↓                                     │
Validation Agent                          │ (retry loop,
    ↓                                     │  max 3 times)
[Retry Check] ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
    ↓
Bad Channel Identifier Agent (✓ Implemented)
    ↓
Optional Notch Filtering Agent ← ─ ─ ┐
    ↓                                 │
Validation Agent                      │ (retry loop,
    ↓                                 │  max 3 times)
[Retry Check] ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
    ↓
Slow Drift Detector Agent (✓ Implemented)
    ↓
[Conditional] Slow Drift Corrector ← ─ ─ ┐
    ↓                                     │
Validation Agent                          │ (retry loop,
    ↓                                     │  max 3 times)
[Retry Check] ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
    ↓
[Conditional] ICA Application Agent ← ─ ─ ┐
    ↓                                      │
Bad ICA Detector Agent (✓ Implemented)    │
    ↓                                      │ (retry loop,
Stage QC Agent                             │  max 3 times)
    ↓                                      │
[Retry Check] ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
    ↓
Interpolation Agent
    ↓
Final QC Agent
```

### Key Features

- **Sequential Processing**: Agents execute in a defined order with conditional branches
- **Retry Mechanisms**: Failed validation triggers retry (max 3 attempts per stage)
- **Conditional Execution**: Stages can be skipped based on Initial QC assessment
- **State Management**: Complete pipeline state passed between agents
- **Visual Analysis**: Agents analyze EEG plots via image URLs
- **Structured Outputs**: Pydantic models ensure consistent agent responses

## File Structure

```
├── main.py                          # Main orchestrator and CLI
├── pipeline_state.py                # State schema definitions
├── pipeline_graph.py                # LangGraph workflow definition
├── agents.py                        # All agent implementations
├── bad_channel_detector.py          # Original bad channel agent (standalone)
├── slow_drift_analysis_agent.py    # Original slow drift agent (standalone)
├── ica_discrimination_agent.py     # Original ICA agent (standalone)
└── pyproject.toml                   # Dependencies
```

## Installation

1. Install dependencies:
```bash
uv pip install langgraph langchain-core
# or
pip install langgraph langchain-core
```

2. Set up environment variables:
```bash
# Create .env file
echo "MISTRAL_API_KEY=your_api_key_here" > .env
```

## Usage

### Show Pipeline Information

```bash
python main.py info
```

### Process Single Subject

```bash
python main.py process sub-001 /path/to/raw_eeg.fif --output-dir ./output --save-report
```

### Batch Process Multiple Subjects

```bash
python main.py batch ./datasets/ds004504 \
    --subjects sub-001 sub-002 sub-003 \
    --output-dir ./output
```

### Programmatic Usage

```python
from main import run_pipeline
from pipeline_state import create_initial_state

# Run pipeline
final_state = run_pipeline(
    subject_id="sub-001",
    raw_eeg_file_path="./datasets/ds004504/sub-001/eeg/sub-001_task-rest_eeg.fif",
    output_dir="./output"
)

# Access results
print(f"Bad channels: {final_state['bad_channels']}")
print(f"ICA components removed: {final_state['ica_components_to_remove']}")
print(f"Success: {final_state['pipeline_success']}")
```

## Agent Details

### Implemented Agents (✓)

These agents use Mistral's multimodal reasoning models to analyze EEG plots:

#### 1. Bad Channel Identifier
- **Model**: `magistral-small-2509`
- **Input**: EEG channel plot URL
- **Output**: List of channels to remove (flat/dead channels)
- **Logic**: Identifies channels with flat lines or near-zero activity

#### 2. Slow Drift Detector
- **Model**: `magistral-medium-2509`
- **Input**: EEG plot URL
- **Output**: Probability (0-1) of slow drift presence
- **Logic**: Detects gradual baseline shifts across channels

#### 3. Bad ICA Component Detector
- **Model**: `magistral-small-2509`
- **Input**: ICA components plot URL
- **Output**: List of ICA components to remove
- **Logic**: Identifies eye movements, heart artifacts, muscle artifacts, line noise

### Placeholder Agents (TBC)

These agents have placeholder implementations with documented TODOs:

#### 1. Initial QC Agent
- Determines which pipeline stages are necessary
- Populates `skip_stages` list
- Currently: proceeds with all stages

#### 2. Notch Filtering Agent
- Analyzes power spectrum for line noise
- Selects filter parameters (50Hz or 60Hz)
- Applies notch filter
- Currently: placeholder with dummy filter params

#### 3. Validation Agents (Multiple)
- Hybrid approach: statistical metrics + LLM visual analysis
- Validates: notch filtering, optional notch, slow drift correction
- Returns pass/fail with justification
- Currently: always returns pass

#### 4. Optional Notch Filtering Agent
- Reassesses power spectrum after bad channel removal
- Independent from initial planner decision
- Currently: placeholder

#### 5. Slow Drift Corrector Agent
- Applies high-pass filtering or detrending
- Triggered when detection probability > 0.5
- Currently: placeholder

#### 6. ICA Application Agent
- Selects ICA algorithm (FastICA, Infomax, Picard)
- Sets number of components
- Generates component plots
- Currently: placeholder

#### 7. Stage QC Agent
- Validates ICA artifact removal quality
- Can trigger ICA re-run (max 3 retries)
- Currently: always passes

#### 8. Interpolation Agent
- Interpolates removed bad channels
- Selects method based on signal characteristics
- Methods: spherical spline, nearest neighbor, etc.
- Currently: placeholder

#### 9. Final QC Agent
- Comprehensive check of entire pipeline
- Generates final report
- Currently: always passes

## Pipeline State

The `EEGPipelineState` TypedDict contains:

```python
{
    "subject_id": str,
    "raw_eeg_file_path": str,
    "current_stage": str,
    
    # Plot URLs for analysis
    "raw_eeg_plot_url": Optional[str],
    "current_eeg_plot_url": Optional[str],
    "ica_components_plot_url": Optional[str],
    "power_spectrum_plot_url": Optional[str],
    
    # Results
    "bad_channels": List[str],
    "ica_components_to_remove": List[int],
    "slow_drift_probability": Optional[float],
    "applied_filters": List[Dict],
    
    # Validations
    "notch_filter_validation": Optional[ValidationResult],
    "optional_notch_validation": Optional[ValidationResult],
    "slow_drift_validation": Optional[ValidationResult],
    "stage_qc_validation": Optional[ValidationResult],
    "final_qc_validation": Optional[ValidationResult],
    
    # Retry tracking
    "notch_filter_retries": int,
    "optional_notch_retries": int,
    "slow_drift_retries": int,
    "ica_qc_retries": int,
    
    # Status
    "errors": List[str],
    "pipeline_completed": bool,
    "pipeline_success": bool,
}
```

## Conditional Logic

### Stage Skipping
- Initial QC can skip: notch filtering, ICA application
- Slow drift correction only runs if probability > 0.5

### Retry Loops
- Max 3 retries per validation stage
- Counter incremented on each retry
- Proceeds to next stage after max retries (even if validation fails)

### Routing Decisions
All routing functions in `pipeline_graph.py`:
- `should_skip_notch_filtering()`
- `should_retry_notch_filtering()`
- `should_retry_optional_notch()`
- `should_correct_slow_drift()`
- `should_retry_slow_drift()`
- `should_skip_ica()`
- `should_retry_ica()`

## Extending the Pipeline

### Implementing a TBC Agent

1. **Locate the placeholder** in `agents.py`
2. **Implement the logic**:
   - Generate required plots (MNE-Python)
   - Upload plot to image hosting or use base64
   - Create Mistral prompt with domain knowledge
   - Call `client.chat.parse()` with appropriate model
   - Update state with results

3. **Example**:
```python
def notch_filtering_agent(state: EEGPipelineState) -> EEGPipelineState:
    # Load EEG data
    raw = mne.io.read_raw(state["raw_eeg_file_path"])
    
    # Generate power spectrum plot
    fig = raw.plot_psd()
    plot_url = upload_plot_to_service(fig)
    
    # LLM analysis
    prompt = "Analyze this power spectrum and determine if notch filtering is needed..."
    messages = create_reasoning_messages(prompt, plot_url)
    
    response = client.chat.parse(
        model="magistral-small-2509",
        messages=messages,
        prompt_mode="reasoning",
        response_format=NotchFilterDecision
    )
    
    # Apply filter if needed
    if response.choices[0].message.parsed.apply_filter:
        raw.notch_filter(response.choices[0].message.parsed.frequency)
        raw.save(updated_file_path)
    
    # Update state
    state["applied_filters"].append({...})
    state["current_eeg_plot_url"] = plot_url
    
    return state
```

### Adding New Agents

1. Define agent function in `agents.py`
2. Add node to workflow in `pipeline_graph.py`:
   ```python
   workflow.add_node("new_agent", new_agent_function)
   ```
3. Connect edges:
   ```python
   workflow.add_edge("previous_agent", "new_agent")
   workflow.add_edge("new_agent", "next_agent")
   ```
4. Update `PipelineStage` enum in `pipeline_state.py`

## Output

### Processing Report (JSON)
```json
{
  "subject_id": "sub-001",
  "pipeline_success": true,
  "bad_channels": ["Fp1", "F7"],
  "ica_components_removed": [0, 1, 5],
  "slow_drift_probability": 0.75,
  "applied_filters": [...],
  "validations": {...},
  "retries": {...},
  "errors": []
}
```

### Batch Summary
```json
{
  "total_subjects": 10,
  "successful": 9,
  "failed": 1,
  "results": [...]
}
```

## Development Roadmap

### Phase 1: Core Implementation (Current)
- ✅ LangGraph workflow structure
- ✅ State management
- ✅ Three core agents (bad channel, slow drift, ICA)
- ✅ Retry mechanisms
- ✅ CLI interface

### Phase 2: Complete Agent Implementation
- ⬜ Initial QC with multimodal analysis
- ⬜ Notch filtering with parameter selection
- ⬜ Validation agents (hybrid approach)
- ⬜ Slow drift correction
- ⬜ ICA application with algorithm selection
- ⬜ Interpolation with adaptive parameters

### Phase 3: Optimization
- ⬜ Plot generation integration (MNE-Python)
- ⬜ Image hosting/management
- ⬜ Prompt engineering optimization
- ⬜ Few-shot examples for agents
- ⬜ Performance benchmarking

### Phase 4: Advanced Features
- ⬜ Parallel batch processing
- ⬜ Pipeline visualization dashboard
- ⬜ Human-in-the-loop validation
- ⬜ Model fine-tuning on EEG domain
- ⬜ Cost optimization

## Troubleshooting

### Missing API Key
```bash
export MISTRAL_API_KEY="your_key_here"
# or add to .env file
```

### LangGraph Import Error
```bash
pip install langgraph langchain-core
```

### Missing Plot URLs
Ensure you implement plot generation in TBC agents:
```python
state["current_eeg_plot_url"] = generate_and_upload_eeg_plot(raw)
```

## Contributing

When implementing TBC agents:
1. Follow the existing agent interface
2. Update state appropriately
3. Add logging statements
4. Handle exceptions gracefully
5. Document parameters and decisions

## License

[Your License]

## Contact

[Your Contact Information]
