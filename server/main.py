from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json
import pickle
from mne_bids import read_raw_bids, BIDSPath
from datetime import datetime
from typing import Dict

import requests

from models import Run, Dataset, DatasetTasks
from database import engine
from sqlmodel import Session, select

from pathlib import Path
import os

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils import (
    upload_image_to_catbox,
    ICAAnalysis,
    EEGSlowDriftAnalysis,
    EEGPipelineState,
    create_reasoning_messages,
    InitialQCResult,
    initial_qc_agent,
    bandpass_filter,
    BandpassFilterSettings,
    bandpass_filtering_agent,
    BadChannelAnalysis,
    bad_channel_identifier_agent,
    annotate_bad_channels,
    notch_filter,
    notch_filtering_agent,
    apply_slow_drift_correction,
    slow_drift_analysis_agent,
    resampling,
    prepare_ica_copy,
    apply_ica,
    ica_discrimination_agent,
    apply_ica_correction,
    interpolate_bad_channels,
    bids_path,
    ALZEIMER_EXPERIMENT_CONTEXT,
    attach_websocket_to_logger,
    detach_websocket_handler,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections per run
active_connections: Dict[str, WebSocket] = {}


class PipelineExecutor:
    """Handles pipeline execution with real-time updates"""

    def __init__(self, run_id: int, websocket: WebSocket):
        self.run_id = run_id
        self.websocket = websocket

    async def send_step_update(self, step_data: dict):
        """Send step update to WebSocket client"""
        await self.websocket.send_json(
            {"type": "step_update", "run_id": self.run_id, **step_data}
        )

    async def send_log(self, message: str, level: str = "info"):
        """Send log message to UI"""
        await self.websocket.send_json(
            {
                "type": "log",
                "run_id": self.run_id,
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def send_graph(self, step_id: int, graph_type: str, graph_data: dict):
        """Send graph data to UI"""
        await self.websocket.send_json(
            {
                "type": "graph",
                "run_id": self.run_id,
                "step_id": step_id,
                "graph_type": graph_type,
                "data": graph_data,
            }
        )

    # async def create_and_save_step(self, step_name: str, raw_data) -> int:
    #     """Create and save a pipeline step"""
    #     with Session(engine) as session:
    #         step = Step(
    #             run_id=self.run_id,
    #             name=step_name,
    #             status="running",
    #             started_at=datetime.now().isoformat(),
    #             raw=pickle.dumps(raw_data)  # Serialize MNE object
    #         )
    #         session.add(step)
    #         session.commit()
    #         session.refresh(step)
    #         return step.id

    # async def update_step_status(self, step_id: int, status: str):
    #     """Update step status"""
    #     with Session(engine) as session:
    #         step = session.get(Step, step_id)
    #         step.status = status
    #         step.ended_at = datetime.now().isoformat()
    #         session.add(step)
    #         session.commit()

    # async def save_step_graphs(self, step_id: int, raw_data, ica=None, step_number):
    #     """Generate and save graphs for a step"""
    #     with Session(engine) as session:

    #         # Save timeseries
    #         data, times = raw_data.get_data(return_times=True)
    #         data = data[:, ::10]  # Downsample
    #         times = times[::10]

    #         await self.send_graph(step_id, "timeseries", {
    #             "channels": raw_data.ch_names,
    #             "data": data.tolist(),
    #             "times": times.tolist()
    #         })

    #         # Save PSD
    #         psd = raw_data.compute_psd()
    #         freqs = psd.freqs.tolist()
    #         psd_values = psd.get_data().tolist()

    #         step.psd = pickle.dumps([freqs, psd_values])

    #         await self.send_graph(step_id, "psd", {
    #             "freqs": freqs,
    #             "psd_values": psd_values
    #         })

    #         # Save sensor topography
    #         montage = raw_data.get_montage()
    #         if montage:
    #             pos = montage.get_positions()
    #             sensors = []
    #             for ch_name in raw_data.ch_names:
    #                 if ch_name in pos['ch_pos']:
    #                     loc = pos['ch_pos'][ch_name]
    #                     sensors.append({"name": ch_name, "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2])})

    #             step.sensor_topography = pickle.dumps(sensors)

    #             await self.send_graph(step_id, "sensors", {"sensors": sensors})

    #         # Save ICA components if available
    #         if ica:
    #             step.ica_components = pickle.dumps(ica.get_components().tolist())
    #             await self.send_graph(step_id, "ica", {
    #                 "components": ica.get_components().tolist()
    #             })

    #         session.add(step)
    #         session.commit()

    async def execute(self, bids_path: BIDSPath):
        """Execute the EEG preprocessing pipeline"""
        try:
            # Update run status
            with Session(engine) as session:
                run = session.get(Run, self.run_id)
                run.pipeline_status = "running"
                run.start_time = datetime.now().isoformat()
                session.add(run)
                session.commit()

            # Load raw data
            await self.send_log("Loading raw EEG data...")
            raw = read_raw_bids(bids_path=bids_path)
            raw.pick("all").load_data()

            # Get subject_id from run
            with Session(engine) as session:
                run = session.get(Run, self.run_id)
                subject_id = run.subject_id

            # Initialize pipeline state
            state = EEGPipelineState(
                subject_id=subject_id,
                current_stage="initial_qc",
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

            # Step 1: Initial QC
            step_id = await self.create_and_save_step("Initial QC", state["input_raw"])
            await self.send_step_update(
                {
                    "step_id": step_id,
                    "step_name": "Initial QC",
                    "step_number": 1,
                    "total_steps": 6,
                    "status": "running",
                }
            )

            # state = initial_qc_agent(state)
            await self.send_log(
                f"Initial QC completed. Stages to skip: {state['skip_stage']}"
            )
            await self.save_step_graphs(step_id, state["input_raw"])
            await self.update_step_status(step_id, "completed")
            await self.send_step_update({"step_id": step_id, "status": "completed"})

            # Step 2: Bandpass Filtering
            if "bandpass_filtering" not in state["skip_stage"]:
                step_id = await self.create_and_save_step(
                    "Bandpass Filtering", state["input_raw"]
                )
                await self.send_step_update(
                    {
                        "step_id": step_id,
                        "step_name": "Bandpass Filtering",
                        "step_number": 2,
                        "total_steps": 6,
                        "status": "running",
                    }
                )

                # state = bandpass_filtering_agent(state)
                await self.send_log(
                    f"Bandpass filtering completed. {state['justification'].get('bandpass_filtering', '')}"
                )
                await self.save_step_graphs(step_id, state["output_raw"])
                await self.update_step_status(step_id, "completed")
                await self.send_step_update({"step_id": step_id, "status": "completed"})
            else:
                await self.send_log("Skipping bandpass filtering", "warning")

            # Step 3: Bad Channel Identification
            if "bad_channel_identification" not in state["skip_stage"]:
                step_id = await self.create_and_save_step(
                    "Bad Channel Identification", state["input_raw"]
                )
                await self.send_step_update(
                    {
                        "step_id": step_id,
                        "step_name": "Bad Channel Identification",
                        "step_number": 3,
                        "total_steps": 6,
                        "status": "running",
                    }
                )

                # state = bad_channel_identifier_agent(state)
                await self.send_log(f"Bad channels identified: {state['bad_channels']}")
                state["output_raw"] = annotate_bad_channels(
                    state["input_raw"], state["bad_channels"]
                )
                await self.save_step_graphs(step_id, state["output_raw"])
                await self.update_step_status(step_id, "completed")
                await self.send_step_update({"step_id": step_id, "status": "completed"})

            # Step 4: Notch Filtering
            if "notch_filtering" not in state["skip_stage"]:
                step_id = await self.create_and_save_step(
                    "Notch Filtering", state["input_raw"]
                )
                await self.send_step_update(
                    {
                        "step_id": step_id,
                        "step_name": "Notch Filtering",
                        "step_number": 4,
                        "total_steps": 6,
                        "status": "running",
                    }
                )

                # state = notch_filtering_agent(state)
                await self.send_log(f"Notch filtering completed")
                await self.save_step_graphs(step_id, state["output_raw"])
                await self.update_step_status(step_id, "completed")
                await self.send_step_update({"step_id": step_id, "status": "completed"})

            # Step 5: ICA
            if "ica" not in state["skip_stage"]:
                step_id = await self.create_and_save_step(
                    "ICA Correction", state["input_raw"]
                )
                await self.send_step_update(
                    {
                        "step_id": step_id,
                        "step_name": "ICA Correction",
                        "step_number": 5,
                        "total_steps": 6,
                        "status": "running",
                    }
                )

                state = resampling(state)
                state = prepare_ica_copy(state)
                ica = apply_ica(state)
                # state = ica_discrimination_agent(state, ica)
                state = apply_ica_correction(state, ica)

                await self.send_log(
                    f"ICA correction completed. {state.get('ica_justification', '')}"
                )
                await self.save_step_graphs(step_id, state["output_raw"], ica=ica)
                await self.update_step_status(step_id, "completed")
                await self.send_step_update({"step_id": step_id, "status": "completed"})

            # Step 6: Final interpolation and QC
            step_id = await self.create_and_save_step("Final QC", state["output_raw"])
            await self.send_step_update(
                {
                    "step_id": step_id,
                    "step_name": "Final QC",
                    "step_number": 6,
                    "total_steps": 6,
                    "status": "running",
                }
            )

            if state["bad_channels"]:
                state["output_raw"] = interpolate_bad_channels(state["output_raw"])

            await self.save_step_graphs(step_id, state["output_raw"])
            await self.update_step_status(step_id, "completed")
            await self.send_step_update({"step_id": step_id, "status": "completed"})

            # Update run status
            with Session(engine) as session:
                run = session.get(Run, self.run_id)
                run.pipeline_status = "completed"
                run.end_time = datetime.now().isoformat()
                session.add(run)
                session.commit()

            await self.send_log("Pipeline completed successfully!", "success")

        except Exception as e:
            await self.send_log(f"Pipeline failed: {str(e)}", "error")
            with Session(engine) as session:
                run = session.get(Run, self.run_id)
                run.pipeline_status = "failed"
                run.end_time = datetime.now().isoformat()
                session.add(run)
                session.commit()
            raise


@app.websocket("/ws/run")
async def run_websocket(
    websocket: WebSocket,
    user_id: int,
    eeg_file_id: int,
    bids_dataset_id: int,
    subject_id: str,
):
    """WebSocket endpoint that creates run and executes pipeline immediately"""
    await websocket.accept()

    try:
        # Create run
        with Session(engine) as session:
            run = Run(
                user_id=user_id,
                eeg_file_id=eeg_file_id,
                bids_dataset_id=bids_dataset_id,
                subject_id=subject_id,
                start_time=datetime.now().isoformat(),
                pipeline_status="pending",
                last_opened_at=datetime.now().isoformat(),
            )
            session.add(run)
            session.commit()
            session.refresh(run)

            # Get BIDS path
            dataset = session.get(Dataset, bids_dataset_id)
            if not dataset:
                await websocket.send_json(
                    {"type": "error", "message": "BIDS dataset not found"}
                )
                await websocket.close()
                return

        # Send run created confirmation
        await websocket.send_json({"type": "run_created", "run_id": run.id})

        # Execute pipeline immediately
        executor = PipelineExecutor(run.id, websocket)
        await executor.execute(bids_path)

        # Send completion
        await websocket.send_json({"type": "pipeline_complete", "run_id": run.id})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


# API Endpoints for retrieving step data

# @app.get("/run/{run_id}/step/{step_number}/timeseries")
# def get_timeseries(run_id: int, step_id: int):

#     raw = pickle.loads(step.raw)
#     data, times = raw.get_data(return_times=True)

#     # Downsample for web
#     data = data[:, ::10]
#     times = times[::10]

#     return {
#         "channels": raw.ch_names,
#         "data": data.tolist(),
#         "times": times.tolist()
#     }

# @app.get("/run/{run_id}/step/{step_id}/psd")
# def get_psd(run_id: int, step_id: int):
#     with Session(engine) as session:
#         step = session.get(Step, step_id)
#         if not step or not step.psd:
#             raise HTTPException(status_code=404, detail="Step or PSD data not found")

#     freqs, psd_values = pickle.loads(step.psd)

#     return {
#         "freqs": freqs,
#         "psd_values": psd_values
#     }

# @app.get("/run/{run_id}/step/{step_id}/sensors")
# def get_sensors(run_id: int, step_id: int):
#     with Session(engine) as session:
#         step = session.get(Step, step_id)
#         if not step or not step.sensor_topography:
#             raise HTTPException(status_code=404, detail="Step or sensor data not found")

#     sensors = pickle.loads(step.sensor_topography)
#     return {
#         "sensors": sensors
#     }

# @app.get("/run/{run_id}/step/{step_id}/ica_components")
# def get_ica_components(run_id: int, step_id: int):

#     with Session(engine) as session:
#         step = session.get(Step, step_id)
#         if not step or not step.ica_components:
#             raise HTTPException(status_code=404, detail="Step or ICA data not found")

#     ica_components = pickle.loads(step.ica_components)
#     return {
#         "ica_components": ica_components
#     }


@app.get("/run")
def get_runs():
    with Session(engine) as session:
        runs = session.exec(select(Run)).all()
        return runs


@app.websocket("/ws/run/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int):
    await websocket.accept()
    
    # Ajouter le WebSocket au logger
    ws_handler = attach_websocket_to_logger(websocket, run_id)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected to run {run_id}",
            "run_id": run_id
        })
        
        # Garder la connexion ouverte
        while True:
            try:
                # Recevoir les messages du client (keepalive)
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Nettoyer le handler
        detach_websocket_handler(ws_handler)
        await websocket.close()


@app.post("/run")
def create_run(dataset_name: str, subject_id: str, task_name: str, background_tasks: BackgroundTasks):
    with Session(engine) as session:
        dataset = session.exec(
            select(Dataset).where(Dataset.name == dataset_name)
        ).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        run = Run(
            dataset_id=int(dataset.id),
            status="pending",
            created_at=datetime.now().isoformat(),
            last_opened_at=datetime.now().isoformat(),
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),  # to be updated when pipeline completes
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    # Lancer le pipeline en tâche de fond
    background_tasks.add_task(execute_pipeline, run_id, dataset_name, subject_id, task_name)
    return {"run_id": run_id}


def execute_pipeline(
    run_id: int, dataset_name: str, subject_id: str, task_name: str
):
    """Exécute le pipeline en tâche de fond"""
    try:
        with Session(engine) as session:
            dataset = session.exec(
                select(Dataset).where(Dataset.name == dataset_name)
            ).first()

            bids_path = BIDSPath(
                subject=subject_id,
                task=task_name,
                root=f"../datasets/{dataset.name}",
                datatype="eeg",
            )

        # Charger les données
        logger.info(f"Loading raw EEG data for run {run_id}")
        raw = read_raw_bids(bids_path=bids_path)
        raw.pick("all").load_data()

        # Initialiser l'état
        logger.info(f"Initializing pipeline state for run {run_id}")
        state = EEGPipelineState(
            subject_id=subject_id,
            current_stage="initial_qc",
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

        # Exécuter les étapes du pipeline

        # Step 1: Initial QC
        logger.info(f"Starting Initial QC for run {run_id}")
        state = initial_qc_agent(state)
        logger.info(f"Initial QC completed. Stages to skip: {state['skip_stage']}")

        # Step 2: Bandpass Filtering
        if "bandpass_filtering" not in state["skip_stage"]:
            logger.info(f"Starting Bandpass Filtering for run {run_id}")
            state = bandpass_filtering_agent(state)
            logger.info(
                f"Bandpass filtering completed. {state['justification'].get('bandpass_filtering', '')}"
            )
        else:
            logger.warning(f"Skipping bandpass filtering for run {run_id}")

        # Step 3: Bad Channel Identification
        if "bad_channel_identification" not in state["skip_stage"]:
            logger.info(f"Starting Bad Channel Identification for run {run_id}")
            state = bad_channel_identifier_agent(state)
            logger.info(f"Bad channels identified: {state['bad_channels']}")
            state["output_raw"] = annotate_bad_channels(
                state["input_raw"], state["bad_channels"]
            )
        else:
            logger.warning(f"Skipping bad channel identification for run {run_id}")

        # Step 4: Notch Filtering
        if "notch_filtering" not in state["skip_stage"]:
            logger.info(f"Starting Notch Filtering for run {run_id}")
            state = notch_filtering_agent(state)
            logger.info(f"Notch filtering completed")
        else:
            logger.warning(f"Skipping notch filtering for run {run_id}")

        # Step 5: Slow Drift Analysis
        if "slow_drift_correction" not in state["skip_stage"]:
            logger.info(f"Starting Slow Drift Analysis for run {run_id}")
            state = slow_drift_analysis_agent(state)
            if state.get("slow_drift_probability"):
                logger.info(f"Slow drift detected, applying correction")
                state = apply_slow_drift_correction(state)
            else:
                logger.info(f"No slow drift detected")
        else:
            logger.warning(f"Skipping slow drift analysis for run {run_id}")

        # Step 6: ICA Correction
        if "ica" not in state["skip_stage"]:
            logger.info(f"Starting ICA Correction for run {run_id}")
            state = resampling(state)
            logger.info(f"Resampling completed")

            state = prepare_ica_copy(state)
            logger.info(f"ICA preparation completed")

            ica = apply_ica(state)
            logger.info(f"ICA fitting completed")

            state = ica_discrimination_agent(state, ica)
            logger.info(
                f"ICA discrimination completed. Components to remove: {state.get('ica_channels_to_remove', [])}"
            )

            state = apply_ica_correction(state, ica)
            logger.info(f"ICA correction applied")
        else:
            logger.warning(f"Skipping ICA correction for run {run_id}")

        # Step 7: Final interpolation
        if state["bad_channels"]:
            logger.info(f"Interpolating bad channels: {state['bad_channels']}")
            state["output_raw"] = interpolate_bad_channels(state["output_raw"])
            logger.info(f"Bad channel interpolation completed")

        logger.info(f"Pipeline completed successfully for run {run_id}")

        with Session(engine) as session:
            run = session.get(Run, run_id)
            run.status = "completed"
            run.completed_at = datetime.now().isoformat()
            session.add(run)
            session.commit()

    except Exception as e:
        logger.error(f"Pipeline failed for run {run_id}: {str(e)}")
        with Session(engine) as session:
            run = session.get(Run, run_id)
            run.status = "failed"
            session.add(run)
            session.commit()
