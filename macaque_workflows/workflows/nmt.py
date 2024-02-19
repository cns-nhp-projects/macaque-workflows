"""NMTv2 preprocessing and setup

NMT workflow runs once at the start of everything. This workflow:
1. Installs the NMTv2 template and associated parcellations
2. Creates dilated brain mask
3. Creates left and right hemisphere masks

Additional custom pre-generated segmentation and visual/insula mask files are
copied to the custom directory
"""
import os
import shutil
from pathlib import Path

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from mnpipe.interfaces.nmt import HemisphereMask
from mnpipe.interfaces.nmt import InstallNMT
from mnpipe.interfaces.nmt import NMTDataSink
from nipype import config
from nipype.interfaces import fsl
from pkg_resources import resource_filename as pkgr_fn

config.update_config({"execution": {"crashfile_format": "txt"}})


def init_nmt_wf(template_dir: str, working_dir: str, wf_name="nmt_wf") -> pe.Workflow:

    wf = pe.Workflow(name=wf_name, base_dir=working_dir)

    get_nmt = pe.Node(
        InstallNMT(install_path=template_dir, use_05mm=True),
        name="get_nmt",
    )

    nmt_files = pe.Node(
        niu.SelectFiles(
            templates={
                "template": "NMT*05mm.nii.gz",
                "template_brain": "NMT*SS.nii.gz",
                "mask": "NMT*brainmask.nii.gz",
                "sarm": "supplemental_SARM/SARM_1*nii.gz",
            },
        ),
        name="standard_files",
    )
    wf.connect(get_nmt, "nmt_path", nmt_files, "base_directory")

    # --- create custom masks ---
    dilate_mask = pe.Node(fsl.DilateImage(operation="modal"), name="dilate_mask")
    wf.connect(nmt_files, "mask", dilate_mask, "in_file")

    make_hem_masks = pe.Node(HemisphereMask(), name="make_hem_masks")
    wf.connect(dilate_mask, "out_file", make_hem_masks, "brain_mask")

    # --- brain extraction ---
    # this dilated brain extraction seems to outperform the default in
    # registration procedures
    brain_extract = pe.Node(fsl.ApplyMask(), name="brain_extract")
    wf.connect(dilate_mask, "out_file", brain_extract, "mask_file")
    wf.connect(nmt_files, "template", brain_extract, "in_file")

    # --- enhance contrast of template ---
    square_image = pe.Node(fsl.UnaryMaths(operation="sqr"), name="square_image")
    wf.connect(nmt_files, "template_brain", square_image, "in_file")

    get_mean = pe.Node(fsl.ImageStats(op_string="-M"), name="get_mean")
    wf.connect(square_image, "out_file", get_mean, "in_file")

    norm_image = pe.Node(fsl.BinaryMaths(operation="div"), name="norm_image")
    wf.connect(get_mean, "out_stat", norm_image, "operand_value")
    wf.connect(square_image, "out_file", norm_image, "in_file")

    # --- create a cortex-only mask ---
    cortex_mask = pe.Node(fsl.ImageMaths(op_string="-binv"), name="cortex_mask")
    wf.connect(nmt_files, "sarm", cortex_mask, "in_file")

    # --- save to NMT directory ---
    datasink = pe.Node(NMTDataSink(), name="datasink")
    wf.connect(get_nmt, "nmt_path", datasink, "output_path")
    wf.connect(dilate_mask, "out_file", datasink, "dilated_mask")
    wf.connect(brain_extract, "out_file", datasink, "dilated_brain")
    wf.connect(cortex_mask, "out_file", datasink, "cortex_mask")
    wf.connect(make_hem_masks, "right_mask", datasink, "right_mask")
    wf.connect(make_hem_masks, "left_mask", datasink, "left_mask")

    return wf


def setup_nmt(template_dir: str, work_dir: str):
    """Run NMT download and setup

    Args:
        template_dir (str): Directory in which the NMT template folder will be
        installed
        work_dir (str): Working directory for NMT workflow

    Returns:
        pathlib.Path: Path to .5mm NMT directory
    """

    if os.path.exists(template_dir) and not os.listdir(template_dir):
        nmt_wf = init_nmt_wf(template_dir, work_dir)
        nmt_wf.run()

    nmt_dir = Path(template_dir, "NMT_v2.0_sym/NMT_v2.0_sym_05mm")

    # copy over custom files
    for i in ("segmentation", "vis_insula_mask"):
        fname = pkgr_fn("macaque_workflows.data", f"{i}.nii.gz")
        shutil.copy2(fname, Path(nmt_dir, "custom"))

    return nmt_dir
