"""
Unit tests for launch_flex_train.py (Multi-region and Multi-machine type DWS Flex-Start)
"""

from unittest import mock
import os
import pytest
from google.cloud.aiplatform_v1.types import custom_job as gca_custom_job
from google.cloud.aiplatform_v1.types import job_state as gca_job_state

from launch_flex_train import (
    FlexTrainingRunner,
    MachineConfig,
    create_job_spec,
    format_state,
    get_regional_staging_bucket,
    infer_accelerator,
    parse_machine_configs,
    parse_args,
)


def test_infer_accelerator():
    acc_type, acc_cnt = infer_accelerator("a2-highgpu-1g")
    assert acc_type == "NVIDIA_TESLA_A100"
    assert acc_cnt == 1

    acc_type, acc_cnt = infer_accelerator("a2-ultragpu-4g")
    assert acc_type == "NVIDIA_A100_80GB"
    assert acc_cnt == 4

    acc_type, acc_cnt = infer_accelerator("g2-standard-48")
    assert acc_type == "NVIDIA_L4"
    assert acc_cnt == 4

    acc_type, acc_cnt = infer_accelerator("a3-highgpu-8g")
    assert acc_type == "NVIDIA_H100_80GB"
    assert acc_cnt == 8

    acc_type, acc_cnt = infer_accelerator("a3-megagpu-8g")
    assert acc_type == "NVIDIA_H100_MEGA_80GB"
    assert acc_cnt == 8

    acc_type, acc_cnt = infer_accelerator("a3-ultragpu-8g")
    assert acc_type == "NVIDIA_H200_141GB"
    assert acc_cnt == 8

    acc_type, acc_cnt = infer_accelerator("g4-standard-48")
    assert acc_type == "NVIDIA_RTX_PRO_6000"
    assert acc_cnt == 1

    acc_type, acc_cnt = infer_accelerator("g4-standard-96")
    assert acc_type == "NVIDIA_RTX_PRO_6000"
    assert acc_cnt == 2

    acc_type, acc_cnt = infer_accelerator("g4-standard-192")
    assert acc_type == "NVIDIA_RTX_PRO_6000"
    assert acc_cnt == 4

    acc_type, acc_cnt = infer_accelerator("g4-standard-384")
    assert acc_type == "NVIDIA_RTX_PRO_6000"
    assert acc_cnt == 8

    # Unknown or non-GPU machine type raises ValueError
    with pytest.raises(ValueError, match="Unknown or unsupported machine type"):
        infer_accelerator("n1-standard-4")

    with pytest.raises(ValueError, match="Unknown or unsupported machine type"):
        infer_accelerator("invalid-machine")


def test_parse_machine_configs():
    # Auto-inferred
    configs = parse_machine_configs("a2-highgpu-1g,g2-standard-48")
    assert len(configs) == 2
    assert configs[0].machine_type == "a2-highgpu-1g"
    assert configs[0].accelerator_type == "NVIDIA_TESLA_A100"
    assert configs[0].accelerator_count == 1
    assert configs[1].machine_type == "g2-standard-48"
    assert configs[1].accelerator_type == "NVIDIA_L4"
    assert configs[1].accelerator_count == 4

    # Explicit colon formats: 3-part (machine:gpu:count) and 2-part (machine:gpu)
    configs2 = parse_machine_configs("n1-standard-4:NVIDIA_TESLA_T4:1,n1-standard-8:NVIDIA_TESLA_V100:2,n1-standard-16:NVIDIA_TESLA_V100")
    assert len(configs2) == 3
    assert configs2[0].machine_type == "n1-standard-4"
    assert configs2[0].accelerator_type == "NVIDIA_TESLA_T4"
    assert configs2[0].accelerator_count == 1
    assert configs2[1].machine_type == "n1-standard-8"
    assert configs2[1].accelerator_type == "NVIDIA_TESLA_V100"
    assert configs2[1].accelerator_count == 2
    assert configs2[2].machine_type == "n1-standard-16"
    assert configs2[2].accelerator_type == "NVIDIA_TESLA_V100"
    assert configs2[2].accelerator_count == 1

    # Unknown machine type without colon spec raises ValueError
    with pytest.raises(ValueError, match="Unknown or unsupported machine type"):
        parse_machine_configs("n1-standard-4")


def test_create_job_spec_cpu():
    cfg = MachineConfig(machine_type="n1-standard-4", accelerator_type=None, accelerator_count=0)
    specs = create_job_spec(display_name="test-job", machine_config=cfg)
    assert len(specs) == 1
    assert specs[0]["machine_spec"]["machine_type"] == "n1-standard-4"
    assert "accelerator_type" not in specs[0]["machine_spec"]
    assert specs[0]["container_spec"]["image_uri"] == "python:3.10-slim"


def test_create_job_spec_gpu():
    cfg = MachineConfig(machine_type="a2-highgpu-1g")
    specs = create_job_spec(display_name="test-job", machine_config=cfg)
    assert len(specs) == 1
    assert specs[0]["machine_spec"]["machine_type"] == "a2-highgpu-1g"
    assert specs[0]["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_A100"
    assert specs[0]["machine_spec"]["accelerator_count"] == 1
    assert "tf-gpu" in specs[0]["container_spec"]["image_uri"]


def test_get_regional_staging_bucket():
    b1 = get_regional_staging_bucket("my-project", "europe-west4")
    assert b1 == "gs://my-project-geap-staging-europe-west4"

    b2 = get_regional_staging_bucket("my-project", "us-central1", "gs://custom-{region}")
    assert b2 == "gs://custom-us-central1"


def test_parse_machine_configs_empty():
    assert parse_machine_configs("") == []
    assert parse_machine_configs("   ") == []


def test_format_state():
    assert "RUNNING" in format_state(gca_job_state.JobState.JOB_STATE_RUNNING)
    assert "QUEUED" in format_state(gca_job_state.JobState.JOB_STATE_QUEUED)
    assert "FAILED" in format_state(gca_job_state.JobState.JOB_STATE_FAILED)


def test_flex_runner_multi_region_multi_machine():
    # 2 regions x 2 machine types = 4 candidate jobs
    regions = ["us-central1", "europe-west4"]
    machine_configs = [
        MachineConfig("a2-highgpu-1g"),
        MachineConfig("g2-standard-48"),
    ]
    project = "test-project"

    runner = FlexTrainingRunner(
        project=project,
        regions=regions,
        machine_configs=machine_configs,
        poll_interval=0.01,
    )

    mock_jobs = {
        ("us-central1", "a2-highgpu-1g"): mock.MagicMock(name="job-us-a2"),
        ("us-central1", "g2-standard-48"): mock.MagicMock(name="job-us-g2"),
        ("europe-west4", "a2-highgpu-1g"): mock.MagicMock(name="job-eu-a2"),
        ("europe-west4", "g2-standard-48"): mock.MagicMock(name="job-eu-g2"),
    }

    for key, job in mock_jobs.items():
        job.name = f"job-{key[0]}-{key[1]}"

    # Progression:
    # europe-west4:g2-standard-48 becomes RUNNING
    mock_jobs[("us-central1", "a2-highgpu-1g")].state = gca_job_state.JobState.JOB_STATE_PENDING
    mock_jobs[("us-central1", "g2-standard-48")].state = gca_job_state.JobState.JOB_STATE_PENDING
    mock_jobs[("europe-west4", "a2-highgpu-1g")].state = gca_job_state.JobState.JOB_STATE_PENDING
    mock_jobs[("europe-west4", "g2-standard-48")].state = gca_job_state.JobState.JOB_STATE_RUNNING

    with mock.patch("google.cloud.aiplatform.CustomJob") as MockCustomJob:
        def custom_job_factory(display_name, worker_pool_specs, project, location, **kwargs):
            m_type = worker_pool_specs[0]["machine_spec"]["machine_type"]
            return mock_jobs[(location, m_type)]

        MockCustomJob.side_effect = custom_job_factory

        exit_code = runner.run()

        assert exit_code == 0
        assert runner.selected_target[0] == "europe-west4"
        assert runner.selected_target[1].machine_type == "g2-standard-48"

        # Verify all 4 jobs were submitted with DWS FLEX_START
        for job in mock_jobs.values():
            job.submit.assert_called_with(
                scheduling_strategy=gca_custom_job.Scheduling.Strategy.FLEX_START,
                max_wait_duration=604800,
            )

        # Verify the 3 other jobs were cancelled
        mock_jobs[("us-central1", "a2-highgpu-1g")].cancel.assert_called_once()
        mock_jobs[("us-central1", "g2-standard-48")].cancel.assert_called_once()
        mock_jobs[("europe-west4", "a2-highgpu-1g")].cancel.assert_called_once()

        # Verify the running job was NOT cancelled
        mock_jobs[("europe-west4", "g2-standard-48")].cancel.assert_not_called()


def test_flex_runner_partial_submission_failure():
    regions = ["us-central1"]
    machine_configs = [
        MachineConfig("a2-highgpu-1g"),
        MachineConfig("g2-standard-48"),
    ]
    project = "test-project"

    runner = FlexTrainingRunner(
        project=project,
        regions=regions,
        machine_configs=machine_configs,
        poll_interval=0.01,
    )

    mock_job_g2 = mock.MagicMock(name="job-g2")
    mock_job_g2.name = "99999-g2"
    mock_job_g2.state = gca_job_state.JobState.JOB_STATE_RUNNING

    def custom_job_factory(display_name, worker_pool_specs, project, location, **kwargs):
        m_type = worker_pool_specs[0]["machine_spec"]["machine_type"]
        if m_type == "a2-highgpu-1g":
            raise RuntimeError("Quota exceeded for a2-highgpu-1g")
        return mock_job_g2

    with mock.patch("google.cloud.aiplatform.CustomJob", side_effect=custom_job_factory):
        exit_code = runner.run()

        assert exit_code == 0
        assert runner.selected_target[0] == "us-central1"
        assert runner.selected_target[1].machine_type == "g2-standard-48"
        assert ("us-central1", "a2-highgpu-1g") in runner.job_errors
        assert mock_job_g2.cancel.call_count == 0


def test_flex_runner_all_submissions_failed():
    regions = ["us-central1"]
    machine_configs = [MachineConfig("a2-highgpu-1g")]
    project = "test-project"

    runner = FlexTrainingRunner(
        project=project,
        regions=regions,
        machine_configs=machine_configs,
        poll_interval=0.01,
    )

    with mock.patch("google.cloud.aiplatform.CustomJob", side_effect=RuntimeError("GCP API error")):
        exit_code = runner.run()
        assert exit_code == 1
        assert runner.selected_target is None


def test_flex_runner_interrupt_cancels_all():
    regions = ["us-central1"]
    machine_configs = [
        MachineConfig("a2-highgpu-1g"),
        MachineConfig("g2-standard-48"),
    ]
    project = "test-project"

    runner = FlexTrainingRunner(
        project=project,
        regions=regions,
        machine_configs=machine_configs,
    )

    mock_job_1 = mock.MagicMock()
    mock_job_2 = mock.MagicMock()
    runner.jobs = {
        ("us-central1", "a2-highgpu-1g"): mock_job_1,
        ("us-central1", "g2-standard-48"): mock_job_2,
    }

    with pytest.raises(SystemExit):
        runner.handle_interrupt(None, None)

    mock_job_1.cancel.assert_called_once()
    mock_job_2.cancel.assert_called_once()


def test_parse_args():
    # When all required args are passed
    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1,europe-west4", "--project", "my-project", "--machine-types", "a2-highgpu-1g,g2-standard-48"]):
        args = parse_args()
        assert args.regions == "us-central1,europe-west4"
        assert args.project == "my-project"
        assert args.machine_types == "a2-highgpu-1g,g2-standard-48"
        assert not hasattr(args, "detach")
        assert not hasattr(args, "poll_interval")
        assert not hasattr(args, "gpu")
        assert not hasattr(args, "gpu_count")

    # With --machine-type alias
    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--project", "my-project", "--machine-type", "g2-standard-4"]):
        args = parse_args()
        assert args.machine_types == "g2-standard-4"

    # When --regions is omitted
    with mock.patch("sys.argv", ["launch_flex_train.py", "--project", "my-project", "--machine-types", "n1-standard-4"]):
        with pytest.raises(SystemExit):
            parse_args()

    # When --project is omitted
    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--machine-types", "n1-standard-4"]):
        with pytest.raises(SystemExit):
            parse_args()

    # When --machine-types is omitted
    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--project", "my-project"]):
        with pytest.raises(SystemExit):
            parse_args()

    # Verify --detach, --poll-interval, --gpu, and --gpu-count are rejected as unrecognized arguments
    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--project", "my-proj", "--machine-types", "n1-standard-4", "--detach"]):
        with pytest.raises(SystemExit):
            parse_args()

    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--project", "my-proj", "--machine-types", "n1-standard-4", "--poll-interval", "10"]):
        with pytest.raises(SystemExit):
            parse_args()

    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--project", "my-proj", "--machine-types", "n1-standard-4", "--gpu", "NVIDIA_TESLA_T4"]):
        with pytest.raises(SystemExit):
            parse_args()

    with mock.patch("sys.argv", ["launch_flex_train.py", "--regions", "us-central1", "--project", "my-proj", "--machine-types", "n1-standard-4", "--gpu-count", "2"]):
        with pytest.raises(SystemExit):
            parse_args()
