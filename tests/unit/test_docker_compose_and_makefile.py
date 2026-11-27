"""Offline unit tests validating Docker Compose GPU override and Makefile targets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_BASE_PATH = REPO_ROOT / "docker-compose.yml"
COMPOSE_GPU_PATH = REPO_ROOT / "docker-compose.gpu.yml"
MAKEFILE_PATH = REPO_ROOT / "Makefile"


def _load_yaml(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"Expected YAML file to exist: {path}"
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_base_docker_compose_remains_portable_cpu_only_default() -> None:
    compose = _load_yaml(COMPOSE_BASE_PATH)
    services = compose.get("services")
    assert isinstance(services, dict)
    assert "ollama" in services

    ollama = services["ollama"]
    assert isinstance(ollama, dict)
    assert ollama.get("image") == "ollama/ollama:latest"
    assert ollama.get("container_name") == "brasao-ai-ollama"
    assert "deploy" not in ollama


def test_gpu_docker_compose_override_reserves_nvidia_gpu_for_ollama() -> None:
    override = _load_yaml(COMPOSE_GPU_PATH)
    services = override.get("services")
    assert isinstance(services, dict)
    assert set(services.keys()) == {"ollama"}

    ollama = services["ollama"]
    assert isinstance(ollama, dict)
    devices = (
        ollama.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices")
    )
    assert isinstance(devices, list)
    assert len(devices) >= 1

    nvidia_devices = [
        device
        for device in devices
        if isinstance(device, dict) and device.get("driver") == "nvidia"
    ]
    assert len(nvidia_devices) == 1
    gpu_device = nvidia_devices[0]
    assert gpu_device.get("count") in ("all", 1)
    capabilities = gpu_device.get("capabilities")
    assert isinstance(capabilities, list)
    assert "gpu" in capabilities


def test_makefile_exposes_ollama_up_gpu_target_and_phony() -> None:
    assert MAKEFILE_PATH.is_file()
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    phony_match = re.search(r"^\.PHONY:\s*(.+)$", content, flags=re.MULTILINE)
    assert phony_match is not None
    phony_targets = phony_match.group(1).split()
    assert "ollama-up" in phony_targets
    assert "ollama-up-gpu" in phony_targets

    target_match = re.search(
        r"^ollama-up-gpu:\s*\n\t(.+)$",
        content,
        flags=re.MULTILINE,
    )
    assert target_match is not None
    recipe = target_match.group(1).strip()
    assert "docker compose" in recipe
    assert "-f docker-compose.yml" in recipe
    assert "-f docker-compose.gpu.yml" in recipe
    assert "up -d ollama" in recipe
