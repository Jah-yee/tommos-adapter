"""Collection of scripts to translate a system into `tommos` input files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import micromagneticmodel as mm
import numpy as np
import pyvista as pv
from jinja2 import Environment, PackageLoader, select_autoescape

if TYPE_CHECKING:
    import jinja2
    import micromagneticmodel

    import tommos_adapter


def write_input_files(
    driver: tommos_adapter.tommos_driver.HysteresisDriver, system: micromagneticmodel.System, **kwargs: Any
) -> None:
    """Write input files.

    - Mesh in `npz` format.
    - Intrinsic properties in `krn` format.
    - Simulation and hysteresis parameters in `p2` format.

    Args:
        driver: Micromagnetic driver.
        system: Micromagnetic system.
        **kwargs: keyword arguments passed to the writing function.
    """
    save_mesh_npz(system)
    write_krn(system)
    write_p2(system, **kwargs)


def _grid(system: micromagneticmodel.System) -> pv.UnstructuredGrid:
    """Return the TOMMOS mesh after validating its type."""
    if not isinstance(system.m, pv.UnstructuredGrid):
        raise TypeError("TOMMOS requires system.m to be a pyvista.UnstructuredGrid.")
    return system.m


def save_mesh_npz(system: micromagneticmodel.System) -> None:
    """Save mesh in `npz` format.

    Args:
        system: Micromagnetic system
    """
    grid = _grid(system)
    knt = grid.points
    ijk = np.c_[
        grid.cell_connectivity.reshape(-1, 4),
        grid.get_array("id"),
    ]
    np.savez(f"{system.name}.npz", knt=knt, ijk=ijk)


def _get_jinja_environment() -> jinja2.Environment:
    """Load jinja environment.

    Returns:
        Loaded jinja environment.
    """
    return Environment(
        loader=PackageLoader("tommos_adapter"),
        autoescape=select_autoescape(),
        trim_blocks=True,
    )


def write_krn(system: micromagneticmodel.System) -> None:
    """Save intrinsic properties in `krn` format.

    Args:
        system: Micromagnetic system.
    """
    grid = _grid(system)
    properties = {"Js": 0.0, "A": 0.0, "K": 0.0, "theta": 0.0, "phi": 0.0}
    if "Js" in grid.cell_data:
        properties["Js"] = np.unique(grid["Js"]).item()  # only works for singlegrain
        # TODO: generalize for multigrain
    for energy in system.energy:
        if isinstance(energy, mm.Exchange):
            properties["A"] = energy.A
        if isinstance(energy, mm.UniaxialAnisotropy):
            properties["K"] = energy.K
            u_x, u_y, u_z = energy.u  # TODO: assumes normal. Is this always true?
            properties["theta"] = np.arccos(u_z)
            if (r_xy := np.sqrt(u_x * u_x + u_y * u_y)) > 1e-20:
                properties["phi"] = np.sign(u_y) * np.arccos(u_x / r_xy)
    env = _get_jinja_environment()
    template = env.get_template("krn.jinja")
    with open(f"{system.name}.krn", "w") as f:
        f.write(template.render({"properties": properties}))


_P2_STRUCTURE = {
    "mesh": ["size"],
    "initial state": ["mx", "my", "mz", "ini"],
    "field": ["hx", "hy", "hz", "hstart", "hfinal", "hstep", "mstep", "bias_type", "bias_strength"],
    "minimizer": [
        "method",
        "max_iter",
        "tol_fun",
        "eps_a",
        "tau0",
        "pc_iters",
        "pc_auto",
        "pc_force_eta",
        "pc_force_alpha",
        "pc_stagnation_nu",
        "tn_iters",
    ],
    "poisson": ["cg_maxiter", "cg_tol", "reg"],
}


def write_p2(system: micromagneticmodel.System, **kwargs: dict[str, Any]) -> None:
    """Save simulation parameters in `p2` format.

    Args:
        system: Micromagnetic system.
        **kwargs: keyword arguments passed to the writing function.
    """
    grid = _grid(system)

    # Evaluate average magnetization
    # TODO: only uniform magnetization is supported
    average_magnetization = grid.point_data["m"].sum(axis=0) / grid.n_points

    # Create parameter dictionary
    parameters = {
        **kwargs,
        "mx": average_magnetization[0],
        "my": average_magnetization[1],
        "mz": average_magnetization[2],
    }

    # Load template
    env = _get_jinja_environment()
    template = env.get_template("p2.jinja")

    # Write `p2` input files using the jinja template
    with open(f"{system.name}.p2", "w") as f:
        f.write(template.render(parameters=parameters, structure=_P2_STRUCTURE))
        # TODO: decide if we want to use the default parameters instead of ignoring them
