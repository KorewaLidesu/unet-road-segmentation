"""MLflow plumbing for RFM-UNet training on Kaggle.

Design goals:

* **Always works offline.** With no configuration at all, everything lands in a
  local file store under the output directory, which Kaggle keeps as notebook
  output. ``mlflow ui --backend-store-uri ./mlruns`` reads it afterwards.
* **Remote without code changes.** If ``MLFLOW_TRACKING_URI`` is set as an env
  var *or* as a Kaggle Secret, that wins and credentials are pulled from the
  same two places.
* **Survives a killed session.** The run id is persisted next to the
  checkpoints, so a resumed Kaggle session appends to the same MLflow run
  instead of starting a second one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

# Secrets that are useful to lift out of Kaggle Secrets into the environment,
# where mlflow picks them up on its own.
SECRET_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_TOKEN",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)


def _rank() -> int:
    for var in ("LOCAL_RANK", "RANK", "SLURM_PROCID"):
        if var in os.environ:
            try:
                return int(os.environ[var])
            except ValueError:
                pass
    return 0


def is_rank_zero() -> bool:
    return _rank() == 0


def load_kaggle_secrets(names: Iterable[str] = SECRET_ENV_VARS) -> list[str]:
    """Copy Kaggle Secrets into ``os.environ`` unless already set there.

    Returns the names that were found. Silently does nothing off Kaggle, or when
    the notebook has no secrets attached -- the local file store is the fallback.
    """
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        return []

    try:
        client = UserSecretsClient()
    except Exception:
        return []

    found = []
    for name in names:
        if os.environ.get(name):
            continue
        try:
            value = client.get_secret(name)
        except Exception:
            continue  # secret not attached to this notebook
        if value:
            os.environ[name] = value
            found.append(name)
    return found


def resolve_tracking_uri(default_dir: str | os.PathLike) -> str:
    """Remote URI if one is configured, otherwise a local file store."""
    load_kaggle_secrets(("MLFLOW_TRACKING_URI",))
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    local = Path(default_dir).expanduser().resolve()
    local.mkdir(parents=True, exist_ok=True)
    return local.as_uri()


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #

_SCALARS = (bool, int, float, str, type(None))


def _is_loggable(value: Any) -> bool:
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(v, _SCALARS) for v in value)
    return False


def config_params(config, skip: Iterable[str] = ()) -> dict[str, Any]:
    """Scalar entries of a py2cfg config, safe to send as MLflow params.

    Everything heavy (``net``, ``loss``, dataloaders, transforms, modules) is
    dropped; lists such as ``classes`` are stringified.
    """
    skip = set(skip)
    params: dict[str, Any] = {}
    for key, value in sorted(dict(config).items()):
        if key.startswith("_") or key in skip:
            continue
        if not _is_loggable(value):
            continue
        params[key] = ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else value
    return params


def _git_sha(repo_root: str | os.PathLike = ".") -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def env_params() -> dict[str, Any]:
    """Facts about the machine that explain a run's speed and its numbers."""
    import torch

    params: dict[str, Any] = {
        "env/python": sys.version.split()[0],
        "env/torch": torch.__version__,
        "env/cuda": torch.version.cuda,
        "env/gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        params["env/gpu"] = torch.cuda.get_device_name(0)
        params["env/gpu_capability"] = f"{major}.{minor}"
        params["env/gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1
        )

    # Which selective-scan path the SSM blocks actually take. Worth recording on
    # every run: the torch fallback is ~2 orders of magnitude slower, so a run that
    # silently landed on it looks like a hardware problem rather than a missing wheel.
    try:
        from geoseg.models.ADAMamba import selective_scan_backend
        params["env/selective_scan"] = selective_scan_backend()
    except Exception:
        params["env/selective_scan"] = "unknown"
    try:
        import triton  # noqa: F401
        params["env/triton"] = triton.__version__
    except Exception:
        params["env/triton"] = "unavailable"

    sha = _git_sha(Path(__file__).resolve().parent.parent)
    if sha:
        params["env/git_sha"] = sha
    upstream = Path(__file__).resolve().parent.parent / ".rfmunet-upstream-sha"
    if upstream.exists():
        params["env/rfmunet_upstream_sha"] = upstream.read_text().strip()[:12]
    return params


def model_params(net) -> dict[str, Any]:
    total = sum(p.numel() for p in net.parameters())
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {
        "model/params_total_m": round(total / 1e6, 3),
        "model/params_trainable_m": round(trainable / 1e6, 3),
    }


# --------------------------------------------------------------------------- #
# run id persistence
# --------------------------------------------------------------------------- #

def _run_id_path(state_dir: str | os.PathLike) -> Path:
    return Path(state_dir) / "mlflow_run.json"


def _read_existing_run_id(state_dir, tracking_uri: str, experiment_name: str) -> str | None:
    """Return a previous run id from this output dir, if it still exists."""
    path = _run_id_path(state_dir)
    if not path.exists():
        return None
    try:
        saved = json.loads(path.read_text())
    except Exception:
        return None
    run_id = saved.get("run_id")
    if not run_id or saved.get("tracking_uri") != tracking_uri:
        return None
    if saved.get("experiment_name") != experiment_name:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        run = mlflow.get_run(run_id)
    except Exception:
        return None
    # A finished run must not be reopened; a killed Kaggle session leaves it RUNNING.
    if run.info.lifecycle_stage != "active" or run.info.status == "FINISHED":
        return None
    return run_id


def saved_run_id(state_dir) -> tuple[str | None, str | None]:
    """``(run_id, tracking_uri)`` recorded by a previous training run, if any.

    Unlike the resume path this ignores run status, so evaluation can attach its
    metrics to a finished training run.
    """
    path = _run_id_path(state_dir)
    if not path.exists():
        return None, None
    try:
        saved = json.loads(path.read_text())
    except Exception:
        return None, None
    return saved.get("run_id"), saved.get("tracking_uri")


def _save_run_id(state_dir, run_id: str, tracking_uri: str, experiment_name: str) -> None:
    path = _run_id_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
    }, indent=2))


# --------------------------------------------------------------------------- #
# logger construction
# --------------------------------------------------------------------------- #

def build_logger(
    experiment_name: str,
    run_name: str,
    default_store_dir: str | os.PathLike,
    state_dir: str | os.PathLike | None = None,
    tags: dict[str, Any] | None = None,
    resume: bool = True,
    log_model: bool = False,
):
    """A ``MLFlowLogger`` pointed at the resolved tracking URI.

    ``state_dir`` (usually the checkpoint dir) is where the run id is cached so
    that a resumed session continues the same run.
    """
    from pytorch_lightning.loggers import MLFlowLogger

    load_kaggle_secrets()
    tracking_uri = resolve_tracking_uri(default_store_dir)

    run_id = None
    if resume and state_dir is not None:
        run_id = _read_existing_run_id(state_dir, tracking_uri, experiment_name)

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags,
        run_id=run_id,
        log_model=log_model,
    )

    # Touching .run_id creates the run; do it now so the id can be cached and
    # printed before the first (possibly long) epoch.
    actual_run_id = logger.run_id
    if state_dir is not None and is_rank_zero():
        _save_run_id(state_dir, actual_run_id, tracking_uri, experiment_name)

    if run_id:
        print(f"[mlflow] resuming run {actual_run_id}")
    else:
        print(f"[mlflow] new run {actual_run_id}")
    print(f"[mlflow] tracking_uri={tracking_uri} experiment={experiment_name}")
    return logger


def log_hyperparams(logger, params: dict) -> None:
    """Log params, tolerating a resumed run that already has some of them.

    MLflow rejects a param whose value changed, and on resume a few genuinely do
    (Kaggle may hand out a P100 instead of a T4). Those go in as tags instead,
    which are overwritable, so a session change never aborts training.
    """
    try:
        logger.log_hyperparams(params)
        return
    except Exception as exc:
        print(f"[mlflow] params already set on this run, recording as tags: {exc}")
    if not is_rank_zero():
        return
    for key, value in params.items():
        try:
            logger.experiment.set_tag(logger.run_id, f"param.{key}", value)
        except Exception:
            pass


def enable_system_metrics() -> bool:
    """Log CPU/GPU/RAM utilisation alongside the training metrics."""
    try:
        import mlflow
        mlflow.enable_system_metrics_logging()
        return True
    except Exception as exc:  # pynvml missing, old mlflow, ...
        print(f"[mlflow] system metrics unavailable: {exc}")
        return False


def log_artifact(logger, local_path: str | os.PathLike, artifact_path: str | None = None) -> None:
    """Best-effort artifact upload; a logging hiccup must not kill training."""
    if not is_rank_zero():
        return
    local_path = Path(local_path)
    if not local_path.exists():
        return
    try:
        logger.experiment.log_artifact(logger.run_id, str(local_path), artifact_path)
    except Exception as exc:
        print(f"[mlflow] could not log artifact {local_path}: {exc}")


def log_dict(logger, payload: dict, artifact_file: str) -> None:
    if not is_rank_zero():
        return
    try:
        logger.experiment.log_dict(logger.run_id, payload, artifact_file)
    except Exception as exc:
        print(f"[mlflow] could not log dict {artifact_file}: {exc}")


def log_image(logger, image, artifact_file: str) -> None:
    """``image`` is an HxWx3 uint8 numpy array."""
    if not is_rank_zero():
        return
    try:
        logger.experiment.log_image(logger.run_id, image, artifact_file)
    except Exception as exc:
        print(f"[mlflow] could not log image {artifact_file}: {exc}")


def set_terminated(logger, status: str = "FINISHED") -> None:
    if not is_rank_zero():
        return
    try:
        logger.experiment.set_terminated(logger.run_id, status)
    except Exception as exc:
        print(f"[mlflow] could not set run status: {exc}")
