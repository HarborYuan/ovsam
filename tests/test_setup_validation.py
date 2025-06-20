"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest


class TestSetupValidation:
    """Test class to validate the testing setup."""

    def test_pytest_import(self):
        """Test that pytest can be imported."""
        assert pytest is not None
        assert hasattr(pytest, "mark")
        assert hasattr(pytest, "fixture")

    def test_project_structure(self):
        """Test that the project structure is as expected."""
        workspace_path = Path("/workspace")
        assert workspace_path.exists()
        assert (workspace_path / "tests").exists()
        assert (workspace_path / "tests" / "__init__.py").exists()
        assert (workspace_path / "tests" / "unit").exists()
        assert (workspace_path / "tests" / "integration").exists()

    def test_conftest_fixtures(self, temp_dir, mock_config):
        """Test that conftest fixtures are available."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "dataset" in mock_config

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        import time
        time.sleep(0.1)
        assert True

    def test_coverage_import(self):
        """Test that coverage tools can be imported."""
        try:
            import coverage
            assert coverage is not None
        except ImportError:
            pytest.skip("Coverage not yet installed")

    def test_pytest_mock_import(self):
        """Test that pytest-mock can be imported."""
        try:
            import pytest_mock
            assert pytest_mock is not None
        except ImportError:
            pytest.skip("pytest-mock not yet installed")

    def test_python_path(self):
        """Test that the project modules are importable."""
        assert "/workspace" in sys.path or any("/workspace" in p for p in sys.path)

    def test_sample_fixtures(self, sample_image_data, sample_annotation, mock_dataset_info):
        """Test that sample data fixtures work correctly."""
        assert sample_image_data["height"] == 3
        assert sample_image_data["width"] == 3
        assert sample_annotation["bbox"] == [10, 20, 50, 60]
        assert mock_dataset_info["num_classes"] == 80

    def test_temp_file_fixture(self, temp_file):
        """Test that temp file fixture creates and cleans up properly."""
        assert temp_file.exists()
        assert temp_file.read_text() == "test content"

    def test_mock_checkpoint(self, mock_checkpoint_path):
        """Test that mock checkpoint fixture works."""
        assert mock_checkpoint_path.exists()
        assert mock_checkpoint_path.suffix == ".pth"

    def test_metrics_fixture(self, sample_metrics):
        """Test that metrics fixture provides expected data."""
        assert "loss" in sample_metrics
        assert "accuracy" in sample_metrics
        assert 0 <= sample_metrics["accuracy"] <= 1


@pytest.mark.parametrize("marker", ["unit", "integration", "slow"])
def test_markers_registered(marker):
    """Test that custom markers are properly registered."""
    # Check if markers are defined in pyproject.toml
    pyproject_path = Path("/workspace/pyproject.toml")
    content = pyproject_path.read_text()
    assert f'"{marker}:' in content


def test_pyproject_exists():
    """Test that pyproject.toml exists and is configured."""
    pyproject_path = Path("/workspace/pyproject.toml")
    assert pyproject_path.exists()
    content = pyproject_path.read_text()
    assert "[tool.poetry]" in content
    assert "[tool.pytest.ini_options]" in content
    assert "[tool.coverage.run]" in content