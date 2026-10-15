"""Test hysteresis driver."""

from unittest.mock import Mock

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


def test_allowed_attributes():
    assert ta.HysteresisDriver()._allowed_attributes == ["max_iter", "tol_fun", "eps_a"]
    driver = ta.HysteresisDriver(max_iter=1000, tol_fun=1e-8, eps_a="auto")
    assert dict(driver) == {"max_iter": 1000, "tol_fun": 1e-8, "eps_a": "auto"}
    with pytest.raises(AttributeError, match="Invalid attribute key"):
        ta.HysteresisDriver(unexpected_argument=1)


@pytest.mark.parametrize(
    "Bmin,Bmax,n,hx,hy,hz,hstart,hfinal,hstep",
    [
        (
            (-1, 0, 0),
            (2, 0, 0),
            6,
            np.float64(1),
            np.float64(0),
            np.float64(0),
            np.float64(-1),
            np.float64(2),
            np.float64(0.6),
        ),
        (
            (1, 0, 0),
            (-3, 0, 0),
            5,
            np.float64(-1),
            np.float64(0),
            np.float64(0),
            np.float64(-1),
            np.float64(3),
            np.float64(1),
        ),
        (
            (0, 0, -8),
            (0, 0, 2),
            21,
            np.float64(0),
            np.float64(0),
            np.float64(1),
            np.float64(-8),
            np.float64(2),
            np.float64(0.5),
        ),
        (
            (2, 0, 0),
            (0, 0, 0),
            3,
            np.float64(-1),
            np.float64(0),
            np.float64(0),
            np.float64(-2),
            np.float64(0),
            np.float64(1),
        ),
    ],
)
@pytest.mark.parametrize("processor", ["_checkargs", "schedule_kwargs_setup", "drive_kwargs_setup"])
def test_process_hysteresis_arguments(processor, Bmin, Bmax, n, hx, hy, hz, hstart, hfinal, hstep):
    """Test that field arguments are processed correctly."""
    kwargs = {
        "Hmin": (Bmin[0] / mm.consts.mu0, Bmin[1] / mm.consts.mu0, Bmin[2] / mm.consts.mu0),
        "Hmax": (Bmax[0] / mm.consts.mu0, Bmax[1] / mm.consts.mu0, Bmax[2] / mm.consts.mu0),
        "n": n,
    }
    hd = ta.HysteresisDriver()
    getattr(hd, processor)(kwargs)
    assert kwargs == {
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "hstart": hstart,
        "hfinal": hfinal,
        "hstep": hstep,
        "n": n,
    }


@pytest.mark.parametrize(
    "kwargs, error_message",
    [
        ({}, "Hmin, Hmax, and n must all be provided"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, 0, 0)}, "Hmin, Hmax, and n must all be provided"),
        ({"Hmin": (-1, 0), "Hmax": (1, 0, 0), "n": 3}, "Hmin must contain three numeric values"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, np.nan, 0), "n": 3}, "Hmax must contain only finite values"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, 0, 0), "n": 3.0}, "n must be an integer"),
        ({"Hmin": (-1, 0, 0), "Hmax": (1, 0, 0), "n": 1}, "n must be at least 2"),
        ({"Hmin": (1, 0, 0), "Hmax": (1, 0, 0), "n": 3}, "Hmin and Hmax must differ"),
        (
            {"Hmin": (1, 0, 0), "Hmax": (0, 1, 0), "n": 3},
            "Hmin and Hmax must be collinear with the origin",
        ),
    ],
)
def test_invalid_hysteresis_arguments(kwargs, error_message):
    """Invalid hysteresis arguments must produce helpful errors."""
    with pytest.raises(ValueError, match=error_message):
        ta.HysteresisDriver()._checkargs(kwargs)


def test_write_input_files_after_checking(state_zero, tmp_path, monkeypatch):
    """Check inputs after processing the field arguments."""
    system = mm.System(name="test_write_input_files_after_checking")
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system.m = state
    system.energy = mm.Demag()
    hd = ta.HysteresisDriver(max_iter=1000, tol_fun=1e-8, eps_a="auto")
    kwargs = {
        "Hmin": (1 / mm.consts.mu0, 0, 0),
        "Hmax": (-2 / mm.consts.mu0, 0, 0),
        "n": 11,
    }
    monkeypatch.chdir(tmp_path)
    hd._checkargs(kwargs)
    hd._write_input_files(system, **kwargs)
    assert (tmp_path / "test_write_input_files_after_checking.npz").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.p2").is_file()
    assert (tmp_path / "test_write_input_files_after_checking.krn").read_text() == "0 0 0 0 1 0\n"
    p2_text = (tmp_path / "test_write_input_files_after_checking.p2").read_text()
    assert "size = 1.0000000000000001e-09" in p2_text
    assert "mx = 0" in p2_text
    assert "my = 0" in p2_text
    assert "mz = 1" in p2_text
    assert "hx = -1" in p2_text
    assert "hstart = -1" in p2_text
    assert "hfinal = 2" in p2_text
    assert "hstep = 0.29999999999999999" in p2_text
    assert "max_iter = 1000" in p2_text
    assert "tol_fun = 1e-08" in p2_text
    assert "eps_a = auto" in p2_text


def test_schedule_command():
    """Test driver schedule command."""
    system = mm.System(name="test_schedule_command")
    hd = ta.HysteresisDriver()
    tr = ta.TommosRunner()
    assert hd._schedule_commands(system, tr) == ["tommos loop test_schedule_command"]


def test_call_passes_progress_information():
    """Pass the number and location of TOMMOS snapshots to the runner."""
    system = mm.System(name="test_progress")
    runner = Mock()

    ta.HysteresisDriver()._call(system, runner, verbose=2, n=7)

    runner.call.assert_called_once_with(
        argstr="test_progress",
        verbose=2,
        total=7,
        glob_name="hyst_test_progress/*.vtu",
    )


def _write_tommos_output(state, output_file, magnetization):
    """Write a minimal VTU file matching TOMMOS output."""
    output_state = state.copy()
    output_state.points /= ta.scripts.DEFAULT_MESH_UNIT
    output_state.clear_data()
    output_state.point_data["m"] = np.broadcast_to(magnetization, (state.n_points, 3))
    output_state.point_data["U"] = np.zeros(state.n_points)
    output_state.cell_data["mat_id"] = state.cell_data["region_id"]
    output_state.save(output_file)


def test_read_data_updates_only_magnetization(state_zero, tmp_path, monkeypatch):
    """Read the last TOMMOS state while retaining user grid metadata."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    state.point_data["custom_point_data"] = np.arange(state.n_points)
    state.cell_data["region"] = np.arange(state.n_cells)
    system = mm.System(name="read_data", m=state)
    original_state = system.m
    original_region_id = state.cell_data["region_id"].copy()
    original_Ms = state.cell_data["Ms"].copy()
    original_region = state.cell_data["region"].copy()
    original_point_data = state.point_data["custom_point_data"].copy()

    output_directory = tmp_path / "hyst_read_data"
    output_directory.mkdir()
    _write_tommos_output(state, output_directory / "state_cfg00010_B+1.0000e+00T.vtu", (0, 1, 0))
    _write_tommos_output(state, output_directory / "state_cfg00002_B-1.0000e+00T.vtu", (1, 0, 0))
    monkeypatch.chdir(tmp_path)

    ta.HysteresisDriver()._read_data(system)

    assert system.m is original_state
    np.testing.assert_array_equal(system.m.point_data["m"], np.tile((0, 1, 0), (state.n_points, 1)))
    np.testing.assert_array_equal(system.m.point_data["custom_point_data"], original_point_data)
    np.testing.assert_array_equal(system.m.cell_data["region_id"], original_region_id)
    np.testing.assert_array_equal(system.m.cell_data["Ms"], original_Ms)
    np.testing.assert_array_equal(system.m.cell_data["region"], original_region)


def test_read_data_requires_output(state_zero, tmp_path, monkeypatch):
    """Missing TOMMOS snapshots produce a helpful error without changing the system."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system = mm.System(name="missing_output", m=state)
    magnetization_before_read = state.point_data["m"].copy()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="No TOMMOS VTU output found"):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


@pytest.mark.parametrize("invalid_output", ["missing_m", "wrong_shape", "different_mesh"])
def test_read_data_rejects_invalid_output(state_zero, tmp_path, monkeypatch, invalid_output):
    """Invalid TOMMOS snapshots do not partially update the system."""
    state = state_zero.copy()
    state.point_data["m"][:, 2] = 1
    system = mm.System(name="invalid_output", m=state)
    magnetization_before_read = state.point_data["m"].copy()

    output_state = state.copy()
    output_state.points /= ta.scripts.DEFAULT_MESH_UNIT
    if invalid_output == "missing_m":
        output_state.point_data.remove("m")
        error_message = "does not contain point data 'm'"
    elif invalid_output == "wrong_shape":
        output_state.point_data["m"] = np.ones((state.n_points, 2))
        error_message = "output magnetization has shape"
    else:
        output_state.points[0, 0] += 1
        error_message = "does not match system.m"

    output_directory = tmp_path / "hyst_invalid_output"
    output_directory.mkdir()
    output_state.save(output_directory / "state_cfg00000_B+0.0000e+00T.vtu")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match=error_message):
        ta.HysteresisDriver()._read_data(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_read)


def test_check_system_missing_energies(state_zero):
    """Test that the system is not well defined if energies are not defined."""
    system = mm.System(name="test_invalid_magnetization")
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    system.m = state
    hd = ta.HysteresisDriver()
    with pytest.raises(RuntimeError, match="System's energy is not defined"):
        hd._check_system(system)


def test_check_system_missing_magnetization():
    """A system without magnetization is invalid."""
    system = mm.System(name="test_invalid_magnetization")
    system.energy = mm.Exchange(A=1e-12)

    with pytest.raises(RuntimeError, match="Undefined magnetization state"):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_requires_unstructured_grid():
    """TOMMOS does not accept a discretisedfield magnetization."""
    system = mm.examples.macrospin()

    with pytest.raises(TypeError, match=r"pyvista\.UnstructuredGrid"):
        ta.HysteresisDriver()._check_system(system)


@pytest.mark.parametrize(
    "association, name, error_message",
    [
        ("point_data", "m", "magnetization array 'm' is missing"),
        ("cell_data", "region_id", "material array 'region_id' is missing"),
        ("cell_data", "Ms", "material array 'Ms' is missing"),
    ],
)
def test_check_system_requires_grid_data(state_zero, association, name, error_message):
    """All arrays needed to create TOMMOS inputs must be present."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    getattr(state, association).remove(name)
    system = mm.System(name="test_missing_grid_data", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match=error_message):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_magnetization_shape(state_zero):
    """Magnetization must contain one three-dimensional vector per point."""
    state = state_zero.copy()
    state.point_data["m"] = np.ones((state.n_points, 2))
    system = mm.System(name="test_invalid_shape", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match="must have shape"):
        ta.HysteresisDriver()._check_system(system)


def test_check_system_finite_magnetization(state_zero):
    """Magnetization values must be finite."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    state.point_data["m"][0, 0] = np.nan
    system = mm.System(name="test_non_finite_m", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match="must contain only finite values"):
        ta.HysteresisDriver()._check_system(system)


@pytest.mark.parametrize("value", [0, 2])
def test_check_system_normalized_magnetization(state_zero, value):
    """Magnetization must be normalized and is not modified during validation."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = value
    magnetization_before_check = state.point_data["m"].copy()
    system = mm.System(name="test_non_normalized_m", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    with pytest.raises(RuntimeError, match="must contain unit vectors"):
        ta.HysteresisDriver()._check_system(system)
    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_check)


def test_check_system_does_not_modify_valid_system(state_zero):
    """Validation accepts pointwise-normalized magnetization without changing it."""
    state = state_zero.copy()
    state.point_data["m"][:, 0] = 1
    magnetization_before_check = state.point_data["m"].copy()
    system = mm.System(name="test_valid_system", energy=mm.Exchange(A=1e-12) + mm.Demag(), m=state)

    ta.HysteresisDriver()._check_system(system)

    np.testing.assert_array_equal(system.m.point_data["m"], magnetization_before_check)


def test_x():
    hd = ta.HysteresisDriver()
    assert hd._x == "B_hysteresis"
