"""Interface to determine the best alignment of source images to a target
image using spatial correlation
"""
import os

import nibabel as nib
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiMasker
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File
from nipype.interfaces.base import InputMultiObject
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.base import TraitedSpec
from nipype.interfaces.base import traits
from scipy.stats import pearsonr


class _SelectCoregistrationMapInputSpec(BaseInterfaceInputSpec):
    target = File()
    sources = InputMultiObject(File())
    output_list = traits.Bool(default_value=True)


class _SelectCoregistrationMapOutputSpec(TraitedSpec):
    source_index = traits.Int()
    correlations = File()


class SelectCoregistration(SimpleInterface):

    input_spec = _SelectCoregistrationMapInputSpec
    output_spec = _SelectCoregistrationMapOutputSpec

    def _run_interface(self, runtime):

        target = nib.load(self.inputs.target)
        target_mask = image.binarize_img(target)

        masker = NiftiMasker(target_mask)
        target_vox = masker.fit_transform(target).ravel()

        correlations = []
        for i, src in enumerate(self.inputs.sources):
            src_vox = masker.fit_transform(src).ravel()
            r, _ = pearsonr(target_vox, src_vox)
            correlations.append(r)

        idx = np.nanargmax(correlations)
        if self.inputs.output_list:
            idx = [idx]
        self._results["source_index"] = idx

        out = os.path.join(runtime.cwd, "corrs.csv")
        np.savetxt(out, np.array(correlations), delimiter=",")
        self._results["correlations"] = out

        return runtime
