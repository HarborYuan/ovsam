"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file in the temp directory."""
    temp_path = temp_dir / "test_file.txt"
    temp_path.write_text("test content")
    yield temp_path


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for tests."""
    return {
        "model": {
            "type": "test_model",
            "pretrained": False,
            "num_classes": 10,
        },
        "dataset": {
            "type": "test_dataset",
            "data_root": "/tmp/test_data",
            "batch_size": 4,
        },
        "optimizer": {
            "type": "SGD",
            "lr": 0.001,
            "momentum": 0.9,
        },
        "scheduler": {
            "type": "StepLR",
            "step_size": 10,
            "gamma": 0.1,
        },
        "training": {
            "num_epochs": 10,
            "checkpoint_interval": 5,
            "log_interval": 100,
        },
    }


@pytest.fixture
def sample_image_data() -> Dict[str, Any]:
    """Provide sample image data for testing."""
    return {
        "image": [[0, 128, 255], [64, 192, 128], [255, 0, 128]],
        "height": 3,
        "width": 3,
        "channels": 3,
        "format": "RGB",
    }


@pytest.fixture
def sample_annotation() -> Dict[str, Any]:
    """Provide sample annotation data for testing."""
    return {
        "id": 1,
        "image_id": 100,
        "category_id": 1,
        "bbox": [10, 20, 50, 60],  # x, y, width, height
        "area": 3000,
        "segmentation": [[10, 20, 60, 20, 60, 80, 10, 80]],
        "iscrowd": 0,
    }


@pytest.fixture
def mock_dataset_info() -> Dict[str, Any]:
    """Provide mock dataset information."""
    return {
        "name": "test_dataset",
        "num_classes": 80,
        "class_names": [f"class_{i}" for i in range(80)],
        "images": 1000,
        "annotations": 5000,
    }


@pytest.fixture
def sample_model_output() -> Dict[str, Any]:
    """Provide sample model output for testing."""
    return {
        "predictions": [
            {"bbox": [10, 20, 50, 60], "score": 0.95, "label": 1},
            {"bbox": [100, 150, 80, 90], "score": 0.87, "label": 2},
        ],
        "masks": None,
        "features": None,
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return logs during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog


@pytest.fixture
def mock_checkpoint_path(temp_dir: Path) -> Path:
    """Create a mock checkpoint file path."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_checkpoint.pth"
    checkpoint_path.touch()
    return checkpoint_path


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Provide sample metrics for testing."""
    return {
        "loss": 0.1234,
        "accuracy": 0.9876,
        "precision": 0.9234,
        "recall": 0.8976,
        "f1_score": 0.9103,
        "mAP": 0.8765,
    }


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit/integration markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)