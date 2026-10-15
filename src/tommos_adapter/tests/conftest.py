"""Test configurations."""

import shlex
import subprocess

import micromagneticmodel as mm
import numpy as np
import pytest
import pyvista as pv


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Define a temporary data directory."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def mesh(data_dir):
    """Define a cubic mesh of side length 20 and mesh size 2."""
    cmd = "tommos mesh --geom box --extent 20,20,20 --h 2 --out-name test"
    subprocess.run(shlex.split(cmd), cwd=data_dir)
    return {k: data_dir / f"test.{k}" for k in ["vtu", "npz"]}


@pytest.fixture(scope="session")
def state_zero(mesh):
    """Define basic micromagnetic state.

    The default magnetization is zero everywhere and the user needs to specify a physical
    magnetization in the tests. The saturation magnetization corresponds to
    a spontaneous polarization of 1 T everywhere.
    """
    state = pv.read(mesh["vtu"])
    state.points *= 1e-9
    id_array = state.get_array("mat_id")
    state.cell_data["region_id"] = id_array
    state.cell_data.remove("mat_id")
    m = np.zeros((state.n_points, 3), dtype=np.float32)
    state.point_data["m"] = m
    state.cell_data["Ms"] = np.full(state.n_cells, 1 / mm.consts.mu0)
    return state
