"""Define drivers for tommos.

The only available driver is a `HysteresisDriver` as the only capability of `tommos` is to run hysteresis loops.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import micromagneticmodel as mm
import numpy as np
import pyvista as pv

from tommos_adapter import scripts
from tommos_adapter.tommos_runner import TommosRunner

if TYPE_CHECKING:
    import micromagneticmodel

    import tommos_adapter


class HysteresisDriver(mm.adapter_base.ExternalDriver):
    """Driver to run a hysteresis loop.

    The field sweep is defined when calling :meth:`drive` or :meth:`schedule`
    using ``Hmin``, ``Hmax``, and ``n``. Field vectors are specified in A/m and
    must lie on a common axis through the origin.

    Examples:
        1. Defining a driver.

        >>> import tommos_adapter as ta
        ...
        >>> hd = ta.HysteresisDriver()

        2. Passing an argument which is not allowed.

        >>> import tommos_adapter as ta
        ...
        >>> md = ta.HysteresisDriver(myarg=1)
        Traceback (most recent call last):
           ...
        AttributeError: ...

        3. Getting the list of allowed attributes.

        >>> import tommos_adapter as ta
        ...
        >>> hd = ta.HysteresisDriver()
        >>> hd._allowed_attributes
        [...]

    """

    _allowed_attributes = ["max_iter", "tol_fun", "eps_a"]

    def _checkargs(self, kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments to TOMMOS parameters."""
        required_arguments = {"Hmin", "Hmax", "n"}
        if not required_arguments.issubset(kwargs):
            raise ValueError("Hmin, Hmax, and n must all be provided.")

        fields = {}
        for name in ("Hmin", "Hmax"):
            try:
                field = np.asarray(kwargs[name], dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must contain three numeric values.") from error
            if field.shape != (3,):
                raise ValueError(f"{name} must contain three numeric values.")
            if not np.all(np.isfinite(field)):
                raise ValueError(f"{name} must contain only finite values.")
            fields[name] = field

        n = kwargs["n"]
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(f"n must be an integer, not {type(n).__name__}.")
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}.")

        Hmin = fields["Hmin"]
        Hmax = fields["Hmax"]
        field_difference = Hmax - Hmin
        difference_norm = np.linalg.norm(field_difference)
        if difference_norm == 0:
            raise ValueError("Hmin and Hmax must differ.")

        Hmin_norm = np.linalg.norm(Hmin)
        Hmax_norm = np.linalg.norm(Hmax)
        if Hmin_norm > 0 and Hmax_norm > 0:
            relative_cross_product = np.linalg.norm(np.cross(Hmin, Hmax)) / (Hmin_norm * Hmax_norm)
            if not np.isclose(relative_cross_product, 0, atol=1e-12):
                raise ValueError("Hmin and Hmax must be collinear with the origin.")

        h = field_difference / difference_norm
        hstart = float(np.vdot(h, Hmin) * mm.consts.mu0)
        hfinal = float(np.vdot(h, Hmax) * mm.consts.mu0)
        kwargs.update(
            hx=float(h[0]),
            hy=float(h[1]),
            hz=float(h[2]),
            hstart=hstart,
            hfinal=hfinal,
            hstep=(hfinal - hstart) / (n - 1),
        )

        # TOMMOS uses the converted field parameters, but n is retained for progress reporting.
        for key in ["Hmin", "Hmax"]:
            kwargs.pop(key)

        # The adapter currently represents one collinear field sweep from Hmin to Hmax.

    def _write_input_files(self, system: micromagneticmodel.System, **kwargs: Any) -> None:
        """Write input files."""
        mesh = scripts.mesh_data(system, mesh_unit=scripts.DEFAULT_MESH_UNIT)
        np.savez(f"{system.name}.npz", **mesh)
        Path(f"{system.name}.krn").write_text(scripts.krn_script(system), encoding="utf-8")

        p2_arguments = {name: kwargs[name] for name in ("hx", "hy", "hz", "hstart", "hfinal", "hstep")}
        p2_arguments.update(dict(self))
        Path(f"{system.name}.p2").write_text(
            scripts.p2_script(
                system,
                mesh_unit=scripts.DEFAULT_MESH_UNIT,
                **p2_arguments,
            ),
            encoding="utf-8",
        )

    def _call(
        self,
        system: micromagneticmodel.System,
        runner: tommos_adapter.tommos_runner.TommosRunner,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """Call runner to launch the simulation."""
        if runner is None:
            runner = TommosRunner()
        runner.call(
            argstr=system.name,
            verbose=verbose,
            total=kwargs.get("n"),
            glob_name=f"hyst_{system.name}/*.vtu",
        )

    def _schedule_commands(
        self, system: micromagneticmodel.System, runner: tommos_adapter.tommos_runner.TommosRunner
    ) -> list[str]:
        """Get the TOMMOS command to add to a schedule script.

        Args:
            system: Micromagnetic system.
            runner: Runner object defining the calculator.

        Returns:
            Command that runs the TOMMOS hysteresis simulation.
        """
        if runner is None:
            runner = tommos_adapter.TommosRunner()
        return [runner._call(argstr=system.name, dry_run=True)]

    def _read_data(self, system: micromagneticmodel.System) -> None:
        """Read generated data.

        After this function is called, the state `system.m` is updated.

        Args:
            system: Micromagnetic system.
        """
        output_directory = Path(f"hyst_{system.name}")
        indexed_output_files = []
        for output_file in output_directory.glob("state_cfg*.vtu"):
            match = re.fullmatch(r"state_cfg(\d+)_.*\.vtu", output_file.name)
            if match is not None:
                indexed_output_files.append((int(match.group(1)), output_file))

        if not indexed_output_files:
            raise RuntimeError(f"No TOMMOS VTU output found in '{output_directory}'.")

        last_output_file = max(indexed_output_files, key=lambda item: item[0])[1]
        try:
            output_state = pv.read(last_output_file)
        except Exception as error:
            raise RuntimeError(f"Cannot read TOMMOS output '{last_output_file}'.") from error

        if not isinstance(output_state, pv.UnstructuredGrid):
            raise RuntimeError(f"TOMMOS output '{last_output_file}' is not an UnstructuredGrid.")
        if "m" not in output_state.point_data:
            raise RuntimeError(f"TOMMOS output '{last_output_file}' does not contain point data 'm'.")

        magnetization = np.asarray(output_state.point_data["m"])
        expected_shape = (system.m.n_points, 3)
        if magnetization.shape != expected_shape:
            raise RuntimeError(
                f"TOMMOS output magnetization has shape {magnetization.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(magnetization)):
            raise RuntimeError("TOMMOS output magnetization contains non-finite values.")

        matching_mesh = (
            output_state.n_points == system.m.n_points
            and output_state.n_cells == system.m.n_cells
            and np.allclose(
                output_state.points * scripts.DEFAULT_MESH_UNIT,
                system.m.points,
                rtol=1e-7,
                atol=scripts.DEFAULT_MESH_UNIT * 1e-5,
            )
            and np.array_equal(output_state.celltypes, system.m.celltypes)
            and np.array_equal(output_state.cell_connectivity, system.m.cell_connectivity)
        )
        if not matching_mesh:
            raise RuntimeError(f"Mesh in TOMMOS output '{last_output_file}' does not match system.m.")

        system.m.point_data["m"] = magnetization.copy()

        # TODO: update table information
        # system.table = table_from_file("output.csv", x=self._x)

    def schedule_kwargs_setup(self, schedule_kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments for a scheduled run.

        Args:
            schedule_kwargs: Arguments containing the starting field ``Hmin``
                in A/m, final field ``Hmax`` in A/m, and number of field
                points ``n``.

        Raises:
            ValueError: If arguments are missing or do not define a valid
                collinear field sweep.
        """
        self._checkargs(schedule_kwargs)

    def drive_kwargs_setup(self, drive_kwargs: dict[str, Any]) -> None:
        """Validate and convert hysteresis arguments for a direct run.

        Args:
            drive_kwargs: Arguments containing the starting field ``Hmin`` in
                A/m, final field ``Hmax`` in A/m, and number of field points
                ``n``.

        Raises:
            ValueError: If arguments are missing or do not define a valid
                collinear field sweep.
        """
        self._checkargs(drive_kwargs)

    def _check_system(self, system: micromagneticmodel.System) -> None:
        """Check that the system is well defined.

        Args:
            system: Micromagnetic system.

        Raises:
            TypeError: If ``system.m`` is not a PyVista unstructured grid.
            RuntimeError: If energy, magnetization, or required grid data are
                missing or invalid.
        """
        if len(system.energy) == 0:
            raise RuntimeError("System's energy is not defined")
        if system.m is None:
            raise RuntimeError(f"Undefined magnetization state in given system {system}.")
        if not isinstance(system.m, pv.UnstructuredGrid):
            raise TypeError("TOMMOS requires system.m to be a pyvista.UnstructuredGrid.")
        if "m" not in system.m.point_data:
            raise RuntimeError("The magnetization array 'm' is missing from system.m.point_data.")
        for name in ("region_id", "Ms"):
            if name not in system.m.cell_data:
                raise RuntimeError(f"The material array '{name}' is missing from system.m.cell_data.")

        magnetization = np.asarray(system.m.point_data["m"])
        expected_shape = (system.m.n_points, 3)
        if magnetization.shape != expected_shape:
            raise RuntimeError(f"The magnetization array 'm' must have shape {expected_shape}.")
        if not np.all(np.isfinite(magnetization)):
            raise RuntimeError("The magnetization array 'm' must contain only finite values.")

        magnetization_norm = np.linalg.norm(magnetization, axis=1)
        if not np.allclose(magnetization_norm, 1, rtol=1e-6, atol=1e-6):
            raise RuntimeError("The magnetization array 'm' must contain unit vectors at every mesh point.")
        if not np.allclose(magnetization, magnetization[0], rtol=1e-12, atol=1e-14):
            raise RuntimeError(
                "TOMMOS adapter currently supports only a uniform initial magnetization; "
                "writing a restart VTU file is not implemented yet."
            )

        try:
            scripts.mesh_data(system, mesh_unit=scripts.DEFAULT_MESH_UNIT)
            scripts.krn_data(system)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"System cannot be translated to TOMMOS input: {error}") from error

    @property
    def _x(self) -> str:
        return "B_hysteresis"
