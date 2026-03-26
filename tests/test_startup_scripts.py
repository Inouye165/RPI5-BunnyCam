"""Regression checks for the local startup and shutdown workflow."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_startup_results_template_has_required_sections():
    """The startup markdown should keep the required operational sections."""
    content = (REPO_ROOT / "STARTUP_RESULTS.md").read_text(encoding="utf-8")

    assert "## Lifecycle Commands" in content
    assert "## Required Components And Versions" in content
    assert "## Run History" in content
    assert "## LLS Session Notes" in content
    assert "Start command: .\\start_local.ps1" in content
    assert "Stop command: .\\stop_local.ps1" in content


def test_start_and_stop_scripts_exist():
    """Both one-command lifecycle entrypoints should be present."""
    assert (REPO_ROOT / "start_local.ps1").is_file()
    assert (REPO_ROOT / "stop_local.ps1").is_file()
    assert (REPO_ROOT / "bin" / "start_local.ps1").is_file()
    assert (REPO_ROOT / "bin" / "stop_local.ps1").is_file()


def test_startup_scripts_record_runtime_state_and_lls_metadata():
    """The launcher scripts should support deterministic shutdown and LLS notes."""
    start_script = (REPO_ROOT / "bin" / "start_local.ps1").read_text(encoding="utf-8")
    stop_script = (REPO_ROOT / "bin" / "stop_local.ps1").read_text(encoding="utf-8")

    assert "bunnycam-runtime.json" in start_script
    assert "bunnycam-runtime.json" in stop_script
    assert "Add-LlsNoteEntry" in start_script
    assert "Add-LlsNoteEntry" in stop_script
    assert "[string]$Actor = 'manual'" in start_script
    assert "[string]$Actor = 'manual'" in stop_script
