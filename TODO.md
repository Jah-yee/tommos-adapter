# tommos-adapter: to-do list

Findings from the review of 2026-10-16. Items are numbered so they can be
referred to; tick them off as they are done. "Measured" means it was reproduced
with the real `tommos` from this repository's pixi environment, not inferred
from reading code.

File references are to this repository unless prefixed with `tommos/`, which
means `reference/tommos/src/tommos/`.

---

## 1. Airbox: `Demag()` currently computes no demag

### 1.1 Background

tommos solves the magnetostatic potential `U` with the Dirichlet condition
`U = 0` on the outer surface of the mesh it is given
(`tommos/loop.py:927`, `find_outer_boundary_mask`). Open-boundary demag
therefore needs vacuum around the magnet: either an air region in the mesh or
the shell tommos can generate with `--add-shell`. The adapter does neither, and
neither do the example notebook and the test fixtures.

Measured, 20 nm cube, `h = 2 nm`, Js = 1 T, A = 1e-10 J/m, K = 0, uniform m,
B = 0. Analytic demag energy density is Kd/3 = 132 629 J/m³.

| mesh                         | nodes / cells  | E at B = 0 (J/m³) | wall time |
|------------------------------|----------------|-------------------|-----------|
| bare cube (what we run now)  | 3284 / 15109   | 4.7e-10           | 5.9 s     |
| `tommos loop --add-shell`    | 6258 / 37027   | 123 346           | 6.8 s     |

The airbox is a numerical requirement of tommos, not part of the user's model.
In ubermag `Demag()` means open-boundary demag (oommfc and mumax3c users never
mesh vacuum). The adapter should therefore add the shell automatically and hide
it from the user.

Facts about the current `--add-shell`, all measured:

- Output VTUs contain the full mesh, shell included, so `_read_data`
  (`tommos_driver.py:204-217`) rejects them as not matching `system.m`.
- Shell generation welds duplicate nodes (1556 in the cube), which **renumbers
  all nodes**. The first N output points are not the body points (largest
  difference 124 mesh units). A coordinate lookup does recover the body: all
  3284 body nodes matched one-to-one, matched nodes moved by at most 6.6e-7
  mesh units, and the body cells come back first, in input order.
- With a shell, the restart check (`tommos/loop.py:910`) compares the restart
  VTU's point count against the mesh *including* the shell. So a restart file
  must contain shell nodes, which the adapter cannot know before tommos runs.
- The shell gets material id `max(mat_id) + 1` and no `.krn` row
  (`tommos/loop.py:146`, `load_materials_krn`), so it carries no exchange or
  anisotropy. The adapter's `.krn` therefore stays as it is.
- Defaults differ between the two entry points: `tommos loop --add-shell` uses
  `--shell-type box` (`tommos/loop.py:377`) and `tommos add-shell` uses `hull`
  (`tommos/add_shell.py:1042`). On the cube they produce 6258 vs 6266 nodes.
  Standalone `tommos add-shell` is deterministic (three identical runs).
- `--center` (ray origin for shell expansion) defaults to `0,0,0` in mesh
  units in both entry points. Whether that matters for a body not centred on
  the origin is **not verified**.

### 1.2 Target behaviour

- `HysteresisDriver().drive(system, ...)` with `Demag()` in the energy and a
  bare magnet mesh gives correct open-boundary demag without the user doing
  anything.
- `system.m` keeps exactly the user's mesh: same points, cells, order and
  arrays. Only `point_data["m"]` changes.
- The shell is configurable, recorded with the drive, and can be switched off
  by users who mesh their own air.
- Restarts (item 4.1) work with the shell enabled.

### 1.3 Changes requested in tommos

Write these up for the tommos developers. The flag name below is a proposal.

- [ ] 1.3.1 **Body-only output.** New CLI flag for `tommos loop`, e.g.
  `--output-body-only`, used together with `--add-shell`. All
  `state_cfg*.vtu` snapshots then contain only the input mesh:
  - points: the nodes of the input `.npz`, in input order and at exactly the
    input coordinates (no welding drift);
  - cells: the input cells, in input order;
  - `cell_data["mat_id"]`: the input material ids;
  - `point_data["m"]` and `point_data["U"]` restricted to those nodes.
- [ ] 1.3.2 **Body-only restart input.** With `--add-shell` (at least when the
  flag from 1.3.1 is set), `[initial state] ini = k` must accept a restart VTU
  that has the input mesh's point count. tommos fills the shell nodes itself;
  any unit vector will do because air carries no moment.
- [ ] 1.3.3 **CLI flag rather than `.p2` key.** `load_params_p2`
  (`tommos/loop.py:188-296`) silently ignores unknown keys, so an older tommos
  would ignore a `.p2` key and write shell-sized output. An unknown CLI flag
  makes argparse fail loudly, which doubles as a feature check.
- [ ] 1.3.4 **Document what `hysteresis.csv` reports with a shell.** Volume
  averages already exclude air (`V_mag`, `tommos/loop.py:861`). State how the
  energy `E` is normalised and whether it includes stray-field energy in the
  shell, since the adapter's table (item 4.3) will expose it.
- [ ] 1.3.5 Ask whether `--center` should default to the body centroid rather
  than the origin. This needs the check in 1.7.3 first.
- [ ] 1.3.6 Ask the tommos developers to align the shell-type defaults of the
  two entry points, or document the difference.

### 1.4 Adapter: driver attributes and validation

- [ ] 1.4.1 Add driver attributes, named in the adapter's own scheme, e.g.:
  - `airbox` (bool, default `True`);
  - `airbox_KL` (float > 1, total outer scale relative to the body; tommos
    default 10);
  - `airbox_K` (float > 1, layer growth factor; tommos default 1.5);
  - `airbox_layers` (int >= 1, optional);
  - `airbox_hmax` (float > 0, **in metres**; the adapter divides by
    `DEFAULT_MESH_UNIT` because tommos expects mesh units);
  - `airbox_shell_type` (`"box"`, `"hull"` or `"triangles"`).

  Leave a parameter out of the command line when the user hasn't set it, so
  tommos's own defaults apply.
- [ ] 1.4.2 Validate all of these before the drive directory is created, in
  `_check_system` or the kwargs setup (see also 2.4), with messages that name
  the attribute and the allowed values.
- [ ] 1.4.3 For `airbox=False`, require that the user's mesh already encloses
  the magnet in air: every node on the outer surface (faces belonging to
  exactly one tetrahedron) must touch only cells with `Ms == 0`. Otherwise
  raise, explaining that tommos sets `U = 0` on the outer surface and that
  demag would be wrong. If `Demag()` is required, an `airbox=False` run on a
  bare mesh can never be correct, so it must not run silently.
- [ ] 1.4.4 Keep `Demag()` mandatory (`scripts.py:426-427`); tommos always
  computes demag.
- [ ] 1.4.5 Check the installed tommos version, or probe for the 1.3.1 flag,
  and raise a clear error if it is too old.

### 1.5 Adapter: command line, runner, read-back

- [ ] 1.5.1 Build the tommos argument string in **one** driver method used by
  both `_call` and `_schedule_commands`. Today both pass only `system.name`
  (`tommos_driver.py:144`, `164`); scheduled jobs must get the same flags.
  The string is
  `<name> --add-shell --output-body-only [--KL ..] [--K ..] [--layers ..] [--hmax ..] [--shell-type ..]`,
  or `<name>` alone for `airbox=False`.
- [ ] 1.5.2 Keep `_read_data`'s strict mesh-identity check
  (`tommos_driver.py:204-217`). With body-only output it passes, and it then
  guards against a tommos that ignored the request.
- [ ] 1.5.3 Record the airbox settings with the drive. `info.json` only
  receives drive kwargs, not driver attributes (`_write_info_json` in
  micromagneticmodel), so either override the info writing or check that
  tommos's `params.log` lists the shell parameters.
- [ ] 1.5.4 Progress reporting (`total=n`, glob `hyst_<name>/*.vtu`) is
  unchanged. Check it once with a shell.

### 1.6 Air regions meshed by the user (independent of the shell)

tommos assembles exchange from `A` per element, regardless of `Js`
(`tommos/loop.py:1129`). Measured: a scalar `Exchange(A=1e-11)` and
`UniaxialAnisotropy(K=1e5)` are written into the `.krn` row of an `Ms = 0`
region (`0 0 100000 0 0 1e-11`).

- [ ] 1.6.1 In `krn_data` (`scripts.py:149-207`), write `A = 0` and `K1 = 0`
  for regions with `Ms == 0` when the parameter is a scalar. If a dict gives a
  non-zero value to an `Ms == 0` region, raise.
- [ ] 1.6.2 Allow `m = 0` at nodes that belong only to `Ms == 0` cells
  (`tommos_driver.py:282-284` and `scripts.py:126-128` reject it today,
  measured). That is the usual ubermag convention. Write any unit vector for
  those nodes to tommos.
- [ ] 1.6.3 Decide what `_read_data` writes back at those nodes: tommos's
  values, or zero for consistency with 1.6.2.

### 1.7 Tests, docs, open questions

- [ ] 1.7.1 Demag regression test: uniformly magnetised cube at B = 0, demag
  energy density against Kd/3. The tolerance has to come from 1.7.2;
  0.93 × analytic was measured with default shell settings at `h = 2 nm`.
  Needs the table reader (4.3), or parse `mammos_hysteresis.csv` in the test.
- [ ] 1.7.2 Separate discretisation error from airbox error: rerun the cube
  with a finer `h` and larger `KL` and record how the energy converges. This
  sets the defaults and the test tolerance.
- [ ] 1.7.3 Check an off-centre body (mesh at 0..20 nm, not centred) with the
  default `--center 0,0,0`, compared with a centred one.
- [ ] 1.7.4 Test that the `airbox=False` check (1.4.3) rejects a bare mesh and
  accepts a mesh with an enclosing air region.
- [ ] 1.7.5 Test that the schedule script contains the same flags as the
  direct run.
- [ ] 1.7.6 Test with a non-convex and with a two-body mesh: does shell
  generation succeed, and do TetGen failures reach the user as readable
  errors?
- [ ] 1.7.7 Update `examples/example.ipynb`: the airbox is added by default,
  how to tune it, its cost (about 2.5× elements on the cube), and the
  `airbox=False` contract.
- [ ] 1.7.8 tommos aborts when its memory estimate exceeds available RAM
  (`--ignore-mem-warning`). With roughly 2.5× elements this triggers sooner;
  make sure the error message reaches the user (see 2.9 about swapped
  stdout/stderr).

### 1.8 Interim, only if correct demag is needed before 1.3 lands

- [ ] 1.8.1 Run `tommos add-shell --in <name>_body.npz --out-npz <name>.npz`
  in the drive directory before `tommos loop`. Build the body-to-full node map
  by nearest-coordinate lookup (measured feasible, see 1.1). Use the map to
  strip output in `_read_data` and to pad restart VTUs with shell nodes. Keep
  the full mesh in the drive directory. Remove this path once 1.3 is
  available.

---

## 2. Correctness and validation bugs

- [ ] 2.1 `schedule()` without an explicit runner raises `NameError`
  (`tommos_driver.py:163`, measured). `tommos_adapter` is imported only under
  `TYPE_CHECKING`; use `TommosRunner()` as `_call` does. Add a test for
  `runner=None`.
- [ ] 2.2 Same keyword arguments as oommfc, different experiment. In oommfc
  `HysteresisDriver.drive(Hmin, Hmax, n)` is a symmetric loop
  Hmin → Hmax → Hmin and `Hsteps` is accepted. Here it is one branch
  (`tommos_driver.py:98-108`). Use tommos's `[field] loop = true`
  (`tommos/hysteresis_loop.py:117-139`) for parity, and either support `Hsteps`
  or reject it explicitly. Today `Hsteps` alone gives the misleading "Hmin,
  Hmax, and n must all be provided" (measured).
- [ ] 2.3 The field axis can come out flipped. The direction is
  `(Hmax - Hmin)/|...|`, so a sweep +1 T → −3 T along x is passed to tommos as
  direction −x with fields −1 → +3. VTU file names (`B+3.0000e+00T` for a
  physical −3 T) and `B_ext_T` / `J_par_T` in `hysteresis.csv` use the
  flipped axis. Fix the axis from the field vectors (e.g. the direction of
  whichever of Hmax/Hmin is non-zero) and allow negative `hstep`; tommos
  accepts either sign. Update `test_process_hysteresis_arguments`
  accordingly: that is a deliberate behaviour change, not weakening a test.
- [ ] 2.4 Driver attributes (`max_iter`, `tol_fun`, `eps_a`) are validated
  only inside `p2_script`, after the drive directory exists. Measured:
  `HysteresisDriver(max_iter=-1).drive(...)` leaves an empty `drive-0/`.
  Validate in `_check_system`.
- [ ] 2.5 Reject unknown drive kwargs (measured: `Hsteps` and `t` silently
  accepted next to a valid sweep).
- [ ] 2.6 Reject `system.T != 0` (measured: accepted, tommos has no
  temperature).
- [ ] 2.7 Reject `Demag(asymptotic_radius=...)` (measured: accepted and
  ignored).
- [ ] 2.8 Accept numpy integers for `n` (`tommos_driver.py:79`) and
  `max_iter` (`scripts.py:273`). Use `numbers.Integral` and exclude `bool`
  (measured: `np.int64(3)` is rejected).
- [ ] 2.9 Give a specific error for inter-region exchange keys
  (`"r1:r2"`) instead of the generic "mismatching regions" (measured).
- [ ] 2.10 Warn on minimiser non-convergence. tommos exits 0 when it hits
  `max_iter`; the adapter could check `gnorm` in `hysteresis.csv`.

## 3. Runner

- [ ] 3.1 Make the executable configurable. `tommos_runner.py:327` hard-codes
  `tommos` from PATH, and a Jupyter kernel without the environment activated
  gets a raw `FileNotFoundError`. Look it up with `shutil.which` and give a
  clear error, and allow a command prefix (e.g. `pixi run -e cuda tommos`).
- [ ] 3.2 Allow extra CLI flags (needed for 1.5.1). The tommos README requires
  `--no-cpp-mkl --cpu-spmv-backend scipy` on macOS, and `pyproject.toml` lists
  `osx-arm64` and `osx-64`. `--ignore-mem-warning` and `--num-devices` should
  also be reachable.
- [ ] 3.3 Fix the `_call` return annotation (it returns `str` for
  `dry_run`, `tommos_runner.py:316`) and the typo "calculator" (line 18).

## 4. Missing features

- [ ] 4.1 Non-uniform initial state, which repeated drives depend on.
  Measured: after one hysteresis drive `m` deviates from uniform by about 5e-5
  and the second `drive()` is rejected. Write `state_cfg00000_*.vtu` into the
  drive directory and set `[initial state] ini = 0`; tommos then continues
  numbering at 1, which `_read_data` handles. With the airbox this needs
  1.3.2.
- [ ] 4.2 Float32 read-back: tommos writes `m` as float32
  (`tommos/hysteresis_loop.py:534`). Decide whether to cast to float64 on
  read, and check that the unit-norm tolerance in `_check_system` still holds
  on the next drive.
- [ ] 4.3 Set `system.table` from `hysteresis.csv` (TODO at
  `tommos_driver.py:221`) with a `B_hysteresis` column so that `_x` means
  something. Column names and units should follow oommfc where possible.
- [ ] 4.4 micromagneticdata plugins: a `CalculatorDrive` and `read_table`,
  registered as entry points under the key `tommos_adapter` (what `info.json`
  records). Follow mock-adapter.
- [ ] 4.5 `MinDriver`: tommos's sweep yields a single field point when
  `hstart == hfinal` (`tommos/hysteresis_loop.py:117-139`; read from code,
  not run). A uniform `Zeeman(H=...)` would supply the field.
- [ ] 4.6 Expose `mstep`, `mfinal`, `method` and `snapshot_every` if users
  need them. If `snapshot_every = 0`, `_read_data` finds nothing; reject or
  handle that.
- [ ] 4.7 Document the provisional grid contract in user docs: `region_id`,
  `Ms` in A/m per cell, unit `m` per node, optional
  `region_ids`/`region_names`, default region names `region_<id>` for dict
  parameters. Today it is only in the `scripts.py` module docstring.

## 5. Packaging

- [ ] 5.1 Direct-URL dependencies need a `git+` prefix
  (`pyproject.toml:27,29`). Measured: pip fails with "Cannot determine
  archive format"; pixi and uv tolerate it.
- [ ] 5.2 `__init__.py:5` imports `pytest` at import time, but pytest is not
  a runtime dependency. Import it inside `test()`.
- [ ] 5.3 The coverage omit pattern `tommos_adapter/tests/*`
  (`pyproject.toml:43`) does not match the `src/` layout.
- [ ] 5.4 Add a changelog fragment for the driver work (only `11.added.md`
  exists).

## 6. Upstream: micromagneticmodel `adapter_base`

- [ ] 6.1 `ExternalRunner.call` swaps the labels in its error message: the
  text after `stdout:` is stderr, and vice versa.
- [ ] 6.2 `info.json` fails with `TypeError: ndarray is not JSON
  serializable` for numpy kwargs such as `Hmin=np.array(...)`, after the input
  files are already written (measured). The adapter accepts that input, so
  either upstream converts, or the adapter documents tuples or lists.
- [ ] 6.3 `drive()` creates the drive directory before input writing, so any
  failure there leaves an empty directory and shifts the next drive number.
  2.4 avoids this for our own validation.

## 7. Style and unnecessary code

- [ ] 7.1 `_check_system` repeats checks that `scripts` already does: the
  magnetisation shape, finiteness and uniformity checks
  (`tommos_driver.py:275-289`) duplicate `scripts.py:115-146`. The uniform
  check exists twice (`tommos_driver.py:285`, `scripts.py:287-292`). Both
  translations are computed again in `_write_input_files`.
- [ ] 7.2 `_scalar_values` and `_vector_values` duplicate the region-mapping
  logic (`scripts.py:431-470`).
- [ ] 7.3 Dangling comment at `tommos_driver.py:114`.

## 8. Tests

- [ ] 8.1 No test depends on demag; `test_align` saturates at 1 T whether
  demag is there or not. Covered by 1.7.1.
- [ ] 8.2 `conftest.py:22` runs `tommos mesh` without `check=True`, so a
  meshing failure appears later as an unrelated error.
- [ ] 8.3 `test_runner.test_call` asserts on tommos's internal exception
  text (`FileNotFoundError ... .npz`), which ties it to tommos implementation
  details. Assert on the return code, or on the adapter's own error.
- [ ] 8.4 Add tests for every validation item in section 2 once it is fixed.
