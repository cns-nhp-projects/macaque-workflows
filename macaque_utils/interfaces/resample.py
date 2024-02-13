import os
import subprocess

import nibabel as nib
import numpy as np
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.base import TraitedSpec
from nipype.interfaces.base import traits


def _fetch_ref_dims(in_file):
    img = nib.load(in_file)
    dims = np.diag(img.header.get_sform())[:-1]
    return tuple(dims)


class _ResampleImageInputSpec(BaseInterfaceInputSpec):
    input_image = File()
    dimension = traits.Int()
    reference_image = File()
    voxel_size = traits.Tuple()
    interp = traits.String()
    output_image = File()


class _ResampleImageOutputSpec(TraitedSpec):
    output_image = File()


class ResampleImage(SimpleInterface):

    input_spec = _ResampleImageInputSpec
    output_spec = _ResampleImageOutputSpec

    def _run_interface(self, runtime):

        input_image = self.inputs.input_image
        dimension = self.inputs.dimension
        reference_image = self.inputs.reference_image
        interp = self.inputs.interp
        output_image = self.inputs.output_image

        voxdims = _fetch_ref_dims(reference_image)

        cmd = (
            f"ResampleImage {dimension} {input_image} {output_image} "
            f"{voxdims[0]}x{voxdims[1]}x{voxdims[2]} [1,0] {interp}"
        )
        subprocess.run(cmd.split())

        self._results["output_image"] = os.path.join(runtime.cwd, output_image)
        return runtime
