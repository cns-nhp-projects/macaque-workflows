import os
import shutil
import subprocess

import nibabel as nib
import numpy as np
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.base import TraitedSpec
from nipype.interfaces.base import traits

pjoin = os.path.join


class _InstallNMTInputSpec(BaseInterfaceInputSpec):
    install_path = traits.String()
    overwrite = traits.Bool(default_value=False)
    use_05mm = traits.Bool(default_value=True)


class _InstallNMTOutputSpec(TraitedSpec):
    nmt_path = traits.String()


class InstallNMT(SimpleInterface):

    input_spec = _InstallNMTInputSpec
    output_spec = _InstallNMTOutputSpec

    def _run_interface(self, runtime):

        install_path = self.inputs.install_path
        overwrite = self.inputs.overwrite
        use_05mm = self.inputs.use_05mm

        if (not os.path.exists(install_path)) or overwrite:
            os.makedirs(install_path, exist_ok=True)
            cmd = (
                f"@Install_NMT -install_dir {install_path} -nmt_ver 2.0 "
                "-sym sym -overwrite"
            )
            subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)

        subdir = "NMT_v2.0_sym"
        if use_05mm:
            subdir += "_05mm"
        out_path = os.path.join(install_path, "NMT_v2.0_sym", subdir)
        self._results["nmt_path"] = out_path
        return runtime


class _HemisphereMaskInputSpec(BaseInterfaceInputSpec):
    brain_mask = File()


class _HemisphereMaskOutputSpec(TraitedSpec):
    left_mask = File()
    right_mask = File()


class HemisphereMask(SimpleInterface):

    input_spec = _HemisphereMaskInputSpec
    output_spec = _HemisphereMaskOutputSpec

    def _run_interface(self, runtime):

        brain_mask = self.inputs.brain_mask
        mask_img = nib.load(brain_mask)
        mask = mask_img.get_fdata()

        # all voxels < 64 are in the left hemisphere (neg. coordinate)
        midpoint = 64

        right_mask = mask.copy()
        right_mask[:midpoint, :, :] *= 0
        left_mask = mask.copy()
        left_mask[midpoint:, :, :] *= 0

        basename = os.path.basename(brain_mask)
        for m, name in zip([right_mask, left_mask], ["right", "left"]):
            img = nib.Nifti1Image(m.astype(np.float32), mask_img.affine)

            out = pjoin(runtime.cwd, basename.replace(".nii.gz", f"_{name}.nii.gz"))
            self._results[f"{name}_mask"] = out
            img.to_filename(out)

        return runtime


class _FetchSegmentationInputSpec(BaseInterfaceInputSpec):
    install_path = traits.String()
    overwrite = traits.Bool(default_value=False)


class _FetchSegmentationOutputSpec(TraitedSpec):
    segmentation_path = traits.String()


class FetchSegmentation(SimpleInterface):

    input_spec = _FetchSegmentationInputSpec
    output_spec = _FetchSegmentationOutputSpec

    def _run_interface(self, runtime):

        install_path = os.path.abspath(self.inputs.install_path)
        overwrite = self.inputs.overwrite

        if (not os.path.exists(install_path)) or overwrite:
            os.makedirs(install_path, exist_ok=True)

            fnames = ["segmentation", "csf", "gray_matter", "subctx_cb", "white_matter"]

            gh_url = (
                "https://raw.githubusercontent.com/danjgale/"
                "better-nmt-segmentation/main/segmentations/"
            )

            for i in fnames:
                cmd = f"wget {gh_url}/{i}.nii.gz -P {install_path}"
                subprocess.run(cmd.split(), stdout=subprocess.DEVNULL)

        self._results["segmentation_path"] = install_path

        return runtime


class _NMTDataSinkInputSpec(BaseInterfaceInputSpec):

    output_path = traits.String()

    dilated_mask = traits.File()
    dilated_brain = traits.File()
    right_mask = traits.File()
    left_mask = traits.File()
    cortex_mask = traits.File()


class _NMTDataSinkSinkOutputSpec(TraitedSpec):
    out_path = traits.String()


class NMTDataSink(SimpleInterface):

    input_spec = _NMTDataSinkInputSpec
    output_spec = _NMTDataSinkSinkOutputSpec

    def _run_interface(self, runtime):
        dest = os.path.join(self.inputs.output_path, "custom")
        os.makedirs(dest, exist_ok=True)

        filemap = {
            "dilated_mask.nii.gz": self.inputs.dilated_mask,
            "dilated_brain.nii.gz": self.inputs.dilated_brain,
            "right_mask.nii.gz": self.inputs.right_mask,
            "left_mask.nii.gz": self.inputs.left_mask,
            "cortical_mask.nii.gz": self.inputs.cortex_mask,
        }
        for k, v in filemap.items():
            shutil.copy2(v, pjoin(dest, k))

        self._results["out_path"] = dest

        return runtime
