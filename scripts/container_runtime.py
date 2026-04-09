"""
Container runtime detection and image management for tranquillyzer.

Resolves docker / apptainer / singularity availability, pulls/caches
images on demand, and builds the shell-quoted invocation string used to
exec a tool inside the container.
"""

import logging
import os
import re
import shlex
import shutil
import subprocess
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default featureCounts image (Docker Hub).
DEFAULT_FEATURECOUNTS_IMAGE = "varishenlab/featurecounts:subread2.0.6_py3.10.12"

_AUTO_ORDER = ("apptainer", "singularity", "docker")


def detect_runtime(preferred: str = "auto") -> str:
    """Return one of 'apptainer', 'singularity', 'docker', 'native'.

    'native' means run the tool directly from $PATH (no container).
    """
    preferred = (preferred or "auto").lower()
    if preferred == "native":
        return "native"
    if preferred != "auto":
        if shutil.which(preferred) is None:
            raise FileNotFoundError(f"Requested container runtime '{preferred}' not found in PATH")
        return preferred
    for rt in _AUTO_ORDER:
        if shutil.which(rt) is not None:
            return rt
    return "native"


def _sif_filename(image: str) -> str:
    """Convert a docker-style image ref to a SIF filename."""
    name = re.sub(r"^docker://", "", image)
    name = name.replace("/", "_").replace(":", "_")
    return f"{name}.sif"


def _default_image_cache() -> str:
    """Default image cache: <package_dir>/container_images/."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(pkg_dir, "container_images")


def ensure_image(
    runtime: str,
    image: str,
    image_cache: Optional[str] = None,
) -> str:
    """Ensure the image is available locally; return the local reference.

    For apptainer/singularity, pulls a SIF into the cache and returns its path.
    For docker, runs `docker pull` and returns the bare image tag.
    For native, returns the image string unchanged (caller ignores it).
    """
    if runtime == "native":
        return image

    if runtime in ("apptainer", "singularity"):
        cache = image_cache or _default_image_cache()
        os.makedirs(cache, exist_ok=True)
        sif_path = os.path.join(cache, _sif_filename(image))
        if os.path.exists(sif_path) and os.path.getsize(sif_path) > 0:
            logger.info("Using cached image: %s", sif_path)
            return sif_path

        pull_uri = image if image.startswith("docker://") else f"docker://{image}"
        logger.info("Pulling image %s -> %s", pull_uri, sif_path)
        subprocess.run([runtime, "pull", sif_path, pull_uri], check=True)
        return sif_path

    if runtime == "docker":
        logger.info("Pulling docker image: %s", image)
        subprocess.run(["docker", "pull", image], check=True)
        return image

    raise ValueError(f"Unknown runtime: {runtime}")


def _normalize_binds(paths: Iterable[str]) -> List[str]:
    """Dedupe bind specs.

    Plain paths (no ':') are absolutized; src:dst entries are preserved verbatim.
    Returns deduped, order-preserving list.
    """
    seen = set()
    out: List[str] = []
    for p in paths:
        if not p:
            continue
        spec = p if ":" in p else os.path.abspath(p)
        if spec not in seen:
            seen.add(spec)
            out.append(spec)
    return out


def build_invocation(
    runtime: str,
    image_ref: str,
    tool: str,
    bind_paths: Iterable[str] = (),
) -> str:
    """Return a shell-quoted command string that runs `tool` inside the container."""
    if runtime == "native":
        return tool

    binds = _normalize_binds(bind_paths)

    if runtime in ("apptainer", "singularity"):
        parts = [runtime, "exec"]
        if binds:
            parts += ["--bind", ",".join(binds)]
        parts += [image_ref, tool]
        return " ".join(shlex.quote(p) for p in parts)

    if runtime == "docker":
        parts = ["docker", "run", "--rm"]
        for b in binds:
            if ":" in b:
                parts += ["-v", b]
            else:
                parts += ["-v", f"{b}:{b}"]
        parts += [image_ref, tool]
        return " ".join(shlex.quote(p) for p in parts)

    raise ValueError(f"Unknown runtime: {runtime}")


def resolve(
    tool: str,
    image: str,
    runtime: str = "auto",
    image_cache: Optional[str] = None,
    bind_paths: Iterable[str] = (),
) -> Tuple[str, str]:
    """One-shot helper: detect runtime, pull image, build invocation.

    Returns (runtime, invocation_string).
    """
    rt = detect_runtime(runtime)
    image_ref = ensure_image(rt, image, image_cache=image_cache)
    return rt, build_invocation(rt, image_ref, tool, bind_paths=bind_paths)
