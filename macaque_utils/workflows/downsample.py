import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import fsl

from macaque_utils.interfaces.resample import ResampleImage


def init_downsample_wf(wf_name: str = "downsample_wf"):
    """Downsamples various structural images to BOLD/DWI resolution

    Subject anatomical and NMT-space images are downsampled to the reference
    resolution so that these images can be used as fixed (target) images for
    the final coregistration and normalization. Doing so will ensure that
    the BOLD/DWI data will retain it's original resolution. Note that the
    actual transforms (e.g., ``init_coregistration wf``) are computed with the
    high resolution anatomical data to obtain most accurate results.

    No actual alignment is performed in this workflow--the input 'reference'
    file is just used to obtain the voxel dimensions.

    The T1w tissue map/segmentation is also downsampled. As well, eroded CSF
    and and WM masks are downsampled (erosion done before downsampling) for
    subsequent confound estimation (see ``init_bold_confound_wf``).

    All resampling is done with ANTs' ResampleImage, either using B-spline or
    nearest neighbours interpolation.

    Parameters
    ----------
    name : str
        Workflow name

    Returns
    -------
    from nipype.pipeline.engine.Workflow
        Downsampling workflow
    """
    wf = pe.Workflow(name=wf_name)
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=["reference", "t1_brain", "t1_mask", "t1_tissue_map", "nmt"],
        ),
        name="input_node",
    )

    downsample_t1 = pe.Node(
        ResampleImage(dimension=3, interp=3, output_image="t1_resampled.nii.gz"),
        name="downsample_t1",
    )
    wf.connect(input_node, "reference", downsample_t1, "reference_image")
    wf.connect(input_node, "t1_brain", downsample_t1, "input_image")

    downsample_nmt = pe.Node(
        ResampleImage(dimension=3, interp=3, output_image="nmt_resampled.nii.gz"),
        name="downsample_nmt",
    )
    wf.connect(input_node, "reference", downsample_nmt, "reference_image")
    wf.connect(input_node, "nmt", downsample_nmt, "input_image")

    downsample_brain_mask = pe.Node(
        ResampleImage(
            dimension=3,
            interp=1,
            output_image="brain_mask_resampled.nii.gz",
        ),
        name="downsample_brain_mask",
    )
    wf.connect(input_node, "reference", downsample_nmt, "reference_image")
    wf.connect(input_node, "t1_mask", downsample_nmt, "input_image")

    # ----------------------------
    # Segmentation and tissue maps
    # ----------------------------
    csf_mask = pe.Node(
        fsl.ImageMaths(op_string="-thr .9 -uthr 1.9 -bin"),
        name="csf_mask",
    )
    wf.connect(input_node, "t1_tissue_map", csf_mask, "in_file")

    erode_csf_mask = pe.Node(fsl.ErodeImage(), name="erode_csf_mask")
    wf.connect(csf_mask, "out_file", erode_csf_mask, "in_file")

    downsample_csf_mask = pe.Node(
        ResampleImage(dimension=3, interp=1, output_image="csf_mask_resampled.nii.gz"),
        name="downsample_csf_mask",
    )
    wf.connect(erode_csf_mask, "out_file", downsample_csf_mask, "input_image")
    wf.connect(input_node, "reference", downsample_csf_mask, "reference_image")

    wm_mask = pe.Node(fsl.ImageMaths(op_string="-thr 2.9 -bin"), name="wm_mask")
    wf.connect(input_node, "t1_tissue_map", wm_mask, "in_file")

    erode_wm_mask = pe.Node(fsl.ErodeImage(), name="erode_wm_mask")
    wf.connect(wm_mask, "out_file", erode_wm_mask, "in_file")

    downsample_wm_mask = pe.Node(
        ResampleImage(dimension=3, interp=1, output_image="wm_mask_resampled.nii.gz"),
        name="downsample_wm_mask",
    )
    wf.connect(erode_wm_mask, "out_file", downsample_wm_mask, "input_image")
    wf.connect(input_node, "reference", downsample_wm_mask, "reference_image")

    downsample_segmentation = pe.Node(
        ResampleImage(
            dimension=3,
            interp=1,
            output_image="segmentation_resampled.nii.gz",
        ),
        name="downsample_segmentation",
    )
    wf.connect(input_node, "t1_tissue_map", downsample_segmentation, "input_image")
    wf.connect(input_node, "reference", downsample_segmentation, "reference_image")

    output_node = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1_brain",
                "t1_mask",
                "wm_mask",
                "csf_mask",
                "t1_tissue_map",
                "nmt",
            ],
        ),
        name="output_node",
    )
    wf.connect(downsample_t1, "output_image", output_node, "t1_brain")
    wf.connect(downsample_nmt, "output_image", output_node, "nmt")
    wf.connect(downsample_brain_mask, "output_image", output_node, "t1_mask")
    wf.connect(downsample_csf_mask, "output_image", output_node, "csf_mask")
    wf.connect(downsample_wm_mask, "output_image", output_node, "wm_mask")
    wf.connect(downsample_segmentation, "output_image", output_node, "t1_tissue_map")

    return wf
