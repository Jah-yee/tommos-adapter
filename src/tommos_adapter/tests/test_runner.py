import re

from tommos_adapter import TommosRunner


def test_name():
    assert TommosRunner().package_name == "Tommos"


def test_call_dry_run():
    runner = TommosRunner()
    cmd = runner._call("hyst", dry_run=True)
    assert cmd == "tommos loop hyst"


def test_call():
    """Test calculator call without needed input files.

    The simulation will fail and we can check return code and reason for the error.
    """
    runner = TommosRunner()
    res = runner._call("nonexistent-system")
    assert res.returncode == 1
    assert "FileNotFoundError" in res.stderr.decode("utf-8")
    assert re.search(
        "No such file or directory: 'nonexistent-system.npz'",
        res.stderr.decode("utf-8"),
    )
