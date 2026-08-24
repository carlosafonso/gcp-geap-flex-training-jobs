#!/usr/bin/env python3
"""
GEAP Flex Training Launcher (Dynamic Workload Scheduler / Flex-Start)
====================================================================
Launches training jobs concurrently across multiple candidate Google Cloud regions
and machine types using GEAP Dynamic Workload Scheduler (Flex-Start). As soon as
any (region, machine_type) combination provisions the requested capacity and
transitions to RUNNING (or succeeds), all other jobs across all other regions
and machine types are cancelled.
"""

import argparse
import concurrent.futures
import datetime
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job
from google.cloud.aiplatform_v1.types import job_state as gca_job_state

logger = logging.getLogger("geap_flex_train")

# Maximum concurrent worker threads for API operations
MAX_CONCURRENT_SUBMISSIONS = 4

# Standard GCP GPU machine families and their default accelerator specs
KNOWN_ACCELERATORS = {
    # A2 High-GPU (A100 40GB)
    "a2-highgpu-1g": ("NVIDIA_TESLA_A100", 1),
    "a2-highgpu-2g": ("NVIDIA_TESLA_A100", 2),
    "a2-highgpu-4g": ("NVIDIA_TESLA_A100", 4),
    "a2-highgpu-8g": ("NVIDIA_TESLA_A100", 8),
    # A2 Ultra-GPU (A100 80GB)
    "a2-ultragpu-1g": ("NVIDIA_A100_80GB", 1),
    "a2-ultragpu-2g": ("NVIDIA_A100_80GB", 2),
    "a2-ultragpu-4g": ("NVIDIA_A100_80GB", 4),
    "a2-ultragpu-8g": ("NVIDIA_A100_80GB", 8),
    # A3 (H100 / H200)
    "a3-highgpu-8g": ("NVIDIA_H100_80GB", 8),
    "a3-megagpu-8g": ("NVIDIA_H100_MEGA_80GB", 8),
    "a3-ultragpu-8g": ("NVIDIA_H200_141GB", 8),
    # G2 (L4)
    "g2-standard-4": ("NVIDIA_L4", 1),
    "g2-standard-8": ("NVIDIA_L4", 1),
    "g2-standard-12": ("NVIDIA_L4", 1),
    "g2-standard-16": ("NVIDIA_L4", 1),
    "g2-standard-24": ("NVIDIA_L4", 2),
    "g2-standard-32": ("NVIDIA_L4", 1),
    "g2-standard-48": ("NVIDIA_L4", 4),
    "g2-standard-96": ("NVIDIA_L4", 8),
    # G4 (RTX PRO 6000 Blackwell)
    "g4-standard-48": ("NVIDIA_RTX_PRO_6000", 1),
    "g4-standard-96": ("NVIDIA_RTX_PRO_6000", 2),
    "g4-standard-192": ("NVIDIA_RTX_PRO_6000", 4),
    "g4-standard-384": ("NVIDIA_RTX_PRO_6000", 8),
}

TERMINAL_STATES = {
    gca_job_state.JobState.JOB_STATE_SUCCEEDED,
    gca_job_state.JobState.JOB_STATE_FAILED,
    gca_job_state.JobState.JOB_STATE_CANCELLED,
    gca_job_state.JobState.JOB_STATE_EXPIRED,
    gca_job_state.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
}

STARTED_STATES = {
    gca_job_state.JobState.JOB_STATE_RUNNING,
    gca_job_state.JobState.JOB_STATE_SUCCEEDED,
    gca_job_state.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
}


def infer_accelerator(machine_type: str) -> Tuple[str, int]:
    """Looks up accelerator type and count for standard GCP GPU machine families."""
    m = machine_type.lower().strip()
    if m in KNOWN_ACCELERATORS:
        return KNOWN_ACCELERATORS[m]
    raise ValueError(
        f"Unknown or unsupported machine type '{machine_type}'. "
        f"Supported GPU machine families: {', '.join(sorted(KNOWN_ACCELERATORS.keys()))}. "
        "Or specify explicitly via 'machine:gpu:count'."
    )


_UNSET = object()


class MachineConfig:
    """Represents a compute tier configuration (machine type + accelerators)."""

    def __init__(
        self,
        machine_type: str,
        accelerator_type: Optional[str] = _UNSET,
        accelerator_count: int = 1,
    ):
        self.machine_type = machine_type.strip()
        if accelerator_type is not _UNSET:
            self.accelerator_type = accelerator_type.strip() if accelerator_type else None
            self.accelerator_count = accelerator_count
        else:
            self.accelerator_type, self.accelerator_count = infer_accelerator(self.machine_type)

    @property
    def label(self) -> str:
        if self.accelerator_type:
            return f"{self.machine_type} ({self.accelerator_type} x{self.accelerator_count})"
        return f"{self.machine_type} (CPU)"

    def __repr__(self) -> str:
        return f"MachineConfig({self.machine_type}, gpu={self.accelerator_type}x{self.accelerator_count})"


def parse_machine_configs(
    machine_types_str: str,
) -> List[MachineConfig]:
    """
    Parses comma-separated machine types or colon-formatted specs.
    Examples:
        'a2-highgpu-1g,g2-standard-48'
        'n1-standard-4:NVIDIA_TESLA_T4:1,g2-standard-4'
    """
    if not machine_types_str:
        return []

    configs = []
    for item in machine_types_str.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":")
        if len(parts) == 3:
            m_type, acc_type, acc_cnt = parts[0].strip(), parts[1].strip(), int(parts[2].strip())
            configs.append(MachineConfig(m_type, acc_type, acc_cnt))
        elif len(parts) == 2:
            m_type, acc_type = parts[0].strip(), parts[1].strip()
            configs.append(MachineConfig(m_type, acc_type, 1))
        else:
            m_type = parts[0].strip()
            configs.append(MachineConfig(m_type))

    return configs


_VERIFIED_BUCKETS = set()


def get_regional_staging_bucket(project: str, region: str, template: Optional[str] = None) -> str:
    """
    Resolves and ensures the existence of a regional staging bucket for GEAP jobs.
    If template contains '{region}' or '{project}', it is interpolated.
    Otherwise defaults to 'gs://{project}-geap-staging-{region}' and creates it if needed.
    """
    if template:
        if "{region}" in template or "{project}" in template:
            return template.format(region=region, project=project)
        return template

    bucket_name = f"{project}-geap-staging-{region}"
    bucket_uri = f"gs://{bucket_name}"

    if bucket_name in _VERIFIED_BUCKETS:
        return bucket_uri

    try:
        from google.cloud import storage
        client = storage.Client(project=project)
        try:
            client.get_bucket(bucket_name)
        except Exception:
            # Check if existing vertex-staging bucket exists
            alt_name = f"{project}-vertex-staging-{region}"
            try:
                client.get_bucket(alt_name)
                _VERIFIED_BUCKETS.add(alt_name)
                return f"gs://{alt_name}"
            except Exception:
                pass

            # Create the geap staging bucket in the target region
            client.create_bucket(bucket_name, location=region)
            logger.info("Created regional staging bucket: %s (location: %s)", bucket_uri, region)
    except Exception:
        pass

    _VERIFIED_BUCKETS.add(bucket_name)
    return bucket_uri


def create_job_spec(
    display_name: str,
    machine_config: MachineConfig,
    image_uri: Optional[str] = None,
) -> List[Dict]:
    """Builds GEAP worker pool specification for a given MachineConfig."""
    machine_spec: Dict[str, any] = {"machine_type": machine_config.machine_type}
    if machine_config.accelerator_type:
        machine_spec["accelerator_type"] = machine_config.accelerator_type
        machine_spec["accelerator_count"] = machine_config.accelerator_count
        if not image_uri:
            image_uri = "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest"
    else:
        if not image_uri:
            image_uri = "python:3.10-slim"

    training_script = (
        "import time, sys, os, socket\n"
        "print('GEAP Flex Training job started', flush=True)\n"
        "print(f'Host: {socket.gethostname()}', flush=True)\n"
        "print(f'Python: {sys.version}', flush=True)\n"
        "print('Simulating workload (30s)...', flush=True)\n"
        "for step in range(1, 6):\n"
        "    time.sleep(6)\n"
        "    print(f'Progress: step {step}/5', flush=True)\n"
        "print('Training completed successfully', flush=True)\n"
    )

    worker_pool_specs = [
        {
            "machine_spec": machine_spec,
            "replica_count": 1,
            "container_spec": {
                "image_uri": image_uri,
                "command": ["python3", "-c"],
                "args": [training_script],
            },
        }
    ]
    return worker_pool_specs


def format_state(state: gca_job_state.JobState) -> str:
    """Returns clean string representation of GEAP JobState."""
    return state.name.replace("JOB_STATE_", "")


class FlexTrainingRunner:
    """Manages concurrent GEAP training jobs across regions and machine types using DWS Flex-Start."""

    def __init__(
        self,
        project: str,
        regions: List[str],
        machine_configs: List[MachineConfig],
        staging_bucket_template: Optional[str] = None,
        base_job_name: str = "geap-flex-train",
        poll_interval: float = 5.0,
        max_wait_duration: int = 604800,
    ):
        self.project = project
        self.regions = regions
        self.machine_configs = machine_configs
        self.staging_bucket_template = staging_bucket_template
        self.base_job_name = base_job_name
        self.poll_interval = poll_interval
        self.max_wait_duration = max_wait_duration

        self.jobs: Dict[Tuple[str, str], aiplatform.CustomJob] = {}
        self.job_errors: Dict[Tuple[str, str], str] = {}
        self.selected_target: Optional[Tuple[str, MachineConfig]] = None
        self._interrupted = False

    def submit_job(
        self, region: str, machine_config: MachineConfig
    ) -> Tuple[Tuple[str, str], Optional[aiplatform.CustomJob], Optional[str]]:
        """Submits a single CustomJob for a (region, machine_config) pair using DWS Flex-Start."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_key = (region, machine_config.machine_type)
        display_name = f"{self.base_job_name}-{region}-{machine_config.machine_type}-{timestamp}"
        staging_bucket = get_regional_staging_bucket(self.project, region, self.staging_bucket_template)

        worker_pool_specs = create_job_spec(
            display_name=display_name,
            machine_config=machine_config,
        )

        try:
            job = aiplatform.CustomJob(
                display_name=display_name,
                worker_pool_specs=worker_pool_specs,
                project=self.project,
                location=region,
                staging_bucket=staging_bucket,
            )

            job.submit(
                scheduling_strategy=gca_custom_job.Scheduling.Strategy.FLEX_START,
                max_wait_duration=self.max_wait_duration,
            )
            return target_key, job, None
        except Exception as e:
            return target_key, None, str(e)

    def cancel_job(self, target_key: Tuple[str, str], job: aiplatform.CustomJob) -> None:
        """Sends a cancellation request to a specific job with automatic retry."""
        region, m_type = target_key
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info("[%s | %s] Cancelling job %s (attempt %d/%d)...", region, m_type, job.name, attempt, max_attempts)
                job.cancel()
                logger.info("[%s | %s] Cancel request confirmed", region, m_type)
                return
            except Exception as e:
                if attempt == max_attempts:
                    logger.error("[%s | %s] Failed to cancel job %s after %d attempts: %s", region, m_type, job.name, max_attempts, e)
                else:
                    time.sleep(1.0)

    def cancel_all_jobs(self, exclude_target: Optional[Tuple[str, str]] = None) -> None:
        """Concurrently cancels all active jobs except an optional excluded target."""
        targets = [
            (target_key, job)
            for target_key, job in self.jobs.items()
            if target_key != exclude_target and job is not None
        ]
        if not targets:
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(targets), MAX_CONCURRENT_SUBMISSIONS)) as executor:
            futures = [executor.submit(self.cancel_job, target_key, job) for target_key, job in targets]
            concurrent.futures.wait(futures)

    def handle_interrupt(self, signum, frame):
        """Signal handler to cancel all jobs if interrupted (Ctrl+C)."""
        if self._interrupted:
            return
        self._interrupted = True
        logger.warning("Received interrupt signal. Cancelling all submitted jobs...")
        self.cancel_all_jobs()
        logger.info("All jobs cancelled. Exiting.")
        sys.exit(1)

    def run(self) -> int:
        """Submits jobs concurrently, monitors until one starts, and cancels the others."""
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

        candidate_pairs = [
            (region, m_cfg)
            for region in self.regions
            for m_cfg in self.machine_configs
        ]

        logger.info(
            "Starting GEAP flex training: project=%s, regions=%s, machines=%s (%d combinations)",
            self.project,
            ", ".join(self.regions),
            ", ".join(cfg.label for cfg in self.machine_configs),
            len(candidate_pairs),
        )

        logger.info("Submitting %d DWS Flex-Start candidate jobs...", len(candidate_pairs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidate_pairs), MAX_CONCURRENT_SUBMISSIONS)) as executor:
            futures = [
                executor.submit(self.submit_job, region, m_cfg)
                for region, m_cfg in candidate_pairs
            ]
            for future in concurrent.futures.as_completed(futures):
                target_key, job, error = future.result()
                region, m_type = target_key
                if job:
                    self.jobs[target_key] = job
                    console_url = f"https://console.cloud.google.com/agent-platform/locations/{region}/training/{job.name}?project={self.project}"
                    logger.info("[%s | %s] Job submitted: %s (Console: %s)", region, m_type, job.name, console_url)
                else:
                    self.job_errors[target_key] = error or "Unknown error"
                    logger.error("[%s | %s] Submission failed: %s", region, m_type, error)

        if not self.jobs:
            logger.error("All job submissions failed across all combinations.")
            return 1

        logger.info("Polling job states every %.1fs...", self.poll_interval)
        start_time = time.time()
        cfg_map = {cfg.machine_type: cfg for cfg in self.machine_configs}

        while not self.selected_target:
            time.sleep(self.poll_interval)
            elapsed = int(time.time() - start_time)
            status_parts = []
            active_count = 0

            for target_key, job in list(self.jobs.items()):
                region, m_type = target_key
                try:
                    state = job.state
                except Exception:
                    state = gca_job_state.JobState.JOB_STATE_UNSPECIFIED

                status_parts.append(f"{region}:{m_type}={format_state(state)}")

                if state in STARTED_STATES:
                    self.selected_target = (region, cfg_map[m_type])
                    break
                elif state not in TERMINAL_STATES:
                    active_count += 1

            logger.info("[+%ds] %s", elapsed, " | ".join(status_parts))

            if self.selected_target:
                break

            if active_count == 0:
                logger.error("All jobs finished or failed without any job starting.")
                return 1

        selected_region, selected_cfg = self.selected_target
        selected_key = (selected_region, selected_cfg.machine_type)
        selected_job = self.jobs[selected_key]

        logger.info(
            "Job started: [%s | %s] is now RUNNING (job ID: %s)",
            selected_region,
            selected_cfg.label,
            selected_job.name,
        )

        logger.info("Cancelling remaining %d jobs...", len(self.jobs) - 1)
        self.cancel_all_jobs(exclude_target=selected_key)
        logger.info("All other jobs cancelled.")

        logger.info("Job is running in [%s | %s]. Exiting.", selected_region, selected_cfg.machine_type)
        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch a GEAP custom training job concurrently across multiple candidate regions and machine types using DWS Flex-Start."
    )
    parser.add_argument(
        "--regions",
        type=str,
        required=True,
        help="Comma-separated list of candidate GCP regions (e.g. us-central1,europe-west4)",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Google Cloud Project ID",
    )
    parser.add_argument(
        "--machine-types",
        "--machine-type",
        type=str,
        required=True,
        help="Comma-separated list of machine types (e.g. a2-highgpu-1g,g2-standard-48) or colon specs",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="geap-flex-train",
        help="Base display name for the custom job (default: geap-flex-train)",
    )
    parser.add_argument(
        "--staging-bucket",
        type=str,
        default=None,
        help="GCS Staging Bucket template (e.g. gs://<PROJECT_ID>-geap-staging-{region}). Auto-resolved per region if not provided.",
    )
    parser.add_argument(
        "--max-wait-duration",
        type=int,
        default=604800,
        help="Maximum duration to wait for DWS Flex-Start resource provisioning in seconds (default: 604800 / 7 days).",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    project = args.project.strip() if args.project else ""
    if not project:
        logger.error("A valid Google Cloud Project ID must be specified via --project.")
        sys.exit(1)

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    if not regions:
        logger.error("At least one region must be specified.")
        sys.exit(1)

    try:
        machine_configs = parse_machine_configs(
            machine_types_str=args.machine_types,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not machine_configs:
        logger.error("At least one machine type must be specified.")
        sys.exit(1)

    runner = FlexTrainingRunner(
        project=project,
        regions=regions,
        machine_configs=machine_configs,
        staging_bucket_template=args.staging_bucket,
        base_job_name=args.job_name,
        max_wait_duration=args.max_wait_duration,
    )

    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
