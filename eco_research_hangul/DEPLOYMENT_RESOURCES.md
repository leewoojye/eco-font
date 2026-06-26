# Deployment Resource Review

Scope: `eco_research_hangul` API/demo inference environment.

The current folder is a CLI-first research pipeline. It loads PyTorch,
Tesseract OCR, font rendering, OpenCV, and optional torchvision/VGG style
scoring. Resource settings should therefore be split between the lightweight
API/demo profile and full research runs.

## Corrected API/Demo Settings

Use this for the existing lightweight API job profile:

```yaml
cpu: 2vCPU
memory: 4Gi
timeout: 10min
concurrency: 1
max_instances: 1
cpu_idle: false
```

Rationale:

- `cpu: 2vCPU` is acceptable for single-request demo inference.
- `memory: 2Gi` is too tight as a default. A one-character lightweight run was
  measured at about 0.93GiB max RSS before container/runtime headroom, and
  heavier configs can load VGG/torchvision. Use `4Gi` for the demo default.
- `timeout: 10min` is acceptable only for the lightweight API profile using
  settings like `sample_steps: 1`, `diffusion_candidates: 0`, and
  `use_vgg_style: false`.
- `concurrency: 1` is correct. Tesseract and PyTorch inference are CPU and
  memory heavy enough that concurrent requests can easily create latency spikes
  or OOM risk on 2vCPU.
- `max_instances: 1` is fine for a demo. It limits cost and keeps generated
  output state simple.
- `cpu_idle: false` is appropriate for CPU-bound inference because the process
  should keep CPU allocation during request handling. The tradeoff is higher
  cost while the instance is warm.

Cloud Run-style equivalent:

```yaml
template:
  timeout: 600s
  scaling:
    max_instance_count: 1
  max_instance_request_concurrency: 1
  containers:
    - resources:
        limits:
          cpu: "2"
          memory: "4Gi"
        cpu_idle: false
```

## Settings That Are Not Covered By The Demo Profile

The above settings are not enough for normal research/training runs such as:

- `configs/guided.yaml`
- `configs/guided_jua.yaml`
- `configs/guided_cherokee.yaml`
- training with `eco-research-hangul train`

Those configs use heavier defaults such as `sample_steps: 48`,
`diffusion_candidates: 2`, and often `use_vgg_style: true`. For those runs,
prefer at least:

```yaml
cpu: 4vCPU
memory: 8Gi
timeout: 30min
concurrency: 1
max_instances: 1
cpu_idle: false
```

For model training, GPU-backed execution is preferable. CPU-only training is
usable for smoke tests, but it is not a practical production training profile.

## Local Measurement

Measured command:

```bash
../.venv/bin/eco-research-hangul guided-infer --config configs/api_job_e1e902f27c1142a793a5505d5afcca37.yaml
```

Measured result on this machine:

```text
elapsed_seconds=6.38
max_rss_gib=0.93
```

This confirms that the lightweight API profile can fit under 10 minutes, but it
does not justify using `2Gi` as the stable deployment default.
