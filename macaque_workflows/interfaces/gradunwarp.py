import os
import subprocess

from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.base import TraitedSpec


class _GradUnwarpInputSpec(BaseInterfaceInputSpec):
    in_file = File()
    coeffs = File()


class _GradUnwarpOutputSpec(TraitedSpec):
    out_file = File()
    out_warp = File()


class GradUnwarp(SimpleInterface):

    input_spec = _GradUnwarpInputSpec
    output_spec = _GradUnwarpOutputSpec

    def _run_interface(self, runtime):

        in_file = self.inputs.in_file
        coeffs = self.inputs.coeffs
        out_file = os.path.basename(in_file).replace(".nii.gz", "_gdc.nii.gz")

        cmd = f"gradient_unwarp.py {in_file} {out_file} siemens -g {coeffs} -n"
        subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)

        self._results["out_file"] = os.path.join(runtime.cwd, out_file)
        self._results["out_warp"] = os.path.join(runtime.cwd, "fullWarp_abs.nii.gz")
        return runtime
