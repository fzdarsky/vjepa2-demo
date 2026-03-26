import subprocess
import sys


def test_main_module_shows_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "app", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "serve" in result.stdout
    assert "infer" in result.stdout
    assert "download" in result.stdout
