"""Run the `tommos` micromagnetic calculator."""

import shlex
import subprocess

from micromagneticmodel import adapter_base


class TommosRunner(adapter_base.ExternalRunner):
    """Runner to execute `tommos`."""

    @property
    def package_name(self) -> str:
        """Name of the calculator."""
        return "Tommos"

    def _call(self, argstr: str, need_stderr: bool = False, dry_run: bool = False) -> subprocess.CompletedProcess:
        """Call the external calcualtor in a subprocess.

        Args:
            argstr: String argument to add to the process call.
            need_stderr: Whether the stderr is needed.
            dry_run: Whether to only return the command to be executed.

        Returns:
            Result of the subprocess call.
        """
        command = shlex.split(f"tommos loop {argstr}")

        # `need_stderr` is ignored in this implementation, and stdout and stderr are always captured.
        stdout = stderr = subprocess.PIPE

        if dry_run:
            return shlex.join(command)
        return subprocess.run(command, stdout=stdout, stderr=stderr)
