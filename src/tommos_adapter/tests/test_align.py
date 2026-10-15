"""Test hysteresis loop."""

import micromagneticmodel as mm
import numpy as np
import pytest

import tommos_adapter as ta


@pytest.mark.parametrize("sign", [+1, -1])
def test_align_external_field(state_zero, tmp_path, sign):
    """Test alignment with external field."""
    # Intrinsic properties
    Js = 1.76
    A = 1e-12
    K = 4e5
    u_theta = 0
    u_phi = 0

    # Read mesh and define magnetization
    state = state_zero.copy()
    state.point_data["m"][:, [1, 2]] = 1 / np.sqrt(2)
    state.cell_data["Js"] *= Js

    # Define micromagnetic system
    system = mm.System(name="test_relaxation")
    system.m = state

    # Define energies
    u = (np.sin(u_theta) * np.cos(u_phi), np.sin(u_theta) * np.sin(u_phi), np.cos(u_theta))
    system.energy = mm.Exchange(A=A) + mm.UniaxialAnisotropy(K=K, u=u)

    # Drive system
    Hmin = (-sign / mm.consts.mu0, 0, 0)
    Hmax = (sign / mm.consts.mu0, 0, 0)
    n = 5
    hd = ta.HysteresisDriver()
    hd.drive(system, dirname=tmp_path, Hmin=Hmin, Hmax=Hmax, n=n)

    # Read final magnetization
    mag_array = system.m.get_array("m")
    average_magnetization = np.mean(mag_array, axis=0)
    np.testing.assert_allclose(average_magnetization[0], sign)
