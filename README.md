# Google Cloud GEAP Flex Training Jobs

Submit a Google Cloud GEAP (Gemini Enterprise Agent Platform, formerly known as Vertex AI) training job across multiple Google Cloud regions and machine types concurrently using Dynamic Workload Scheduler Flex-Start. The first combination that allocates capacity and starts running is kept; all other jobs are cancelled automatically.

This helps when GPU capacity is constrained and you are flexible on both region and compute tier (for example, able to run on either an A100 or an L4).

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Authenticate with Google Cloud:

```bash
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
```

## Usage

### Customizing the training job

Modify the [`create_job_spec()`](https://github.com/cloud-gtm/gcp-geap-flex-training-jobs/blob/main/launch_flex_train.py#L196-L235) function in [`launch_flex_train.py`](https://github.com/cloud-gtm/gcp-geap-flex-training-jobs/blob/main/launch_flex_train.py) to configure your actual workload. Things you might want to change:

- **Container image:** Set your custom Docker image URI (`image_uri`).
- **Entrypoint & command:** Change the execution command (`command`).
- **Training script & args:** Pass your script path or training arguments (`args`).
- **Environment variables:** Add any required variables or API keys (`env`).
- **Replicas:** Adjust `replica_count` or worker pools for distributed training.

### Basic run

Submit a training job across candidate regions and machine types:

```bash
python launch_flex_train.py \
    --project my-project-id \
    --regions europe-west4,us-central1 \
    --machine-types n1-standard-4
```

### Multi-region and multi-machine execution

Submit across multiple regions and candidate GPU machine types at once:

```bash
python launch_flex_train.py \
    --project my-project-id \
    --regions europe-west4,us-central1,asia-southeast1 \
    --machine-types a2-highgpu-1g,g2-standard-48
```

Accelerators are auto-inferred for standard GCP GPU families.

You can also specify custom accelerator pairings explicitly:

```bash
python launch_flex_train.py \
    --project my-project-id \
    --regions europe-west4,us-central1 \
    --machine-types "n1-standard-4:NVIDIA_TESLA_T4:1,g2-standard-4"
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--project` | *(required)* | Google Cloud Project ID |
| `--regions` | *(required)* | Comma-separated list of candidate GCP regions |
| `--machine-types` | *(required)* | Comma-separated machine types or `machine:gpu:count` specs |
| `--job-name` | `geap-flex-train` | Display name prefix for CustomJob |
| `--staging-bucket` | `gs://<project>-geap-staging-<region>` | GCS bucket template for staging |
| `--max-wait-duration` | `604800` (7 days) | Maximum wait duration in seconds for DWS Flex-Start allocation |

## Tests

```bash
pytest test_launch_flex_train.py
```
