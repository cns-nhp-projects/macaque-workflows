import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import ants
from nipype.interfaces import fsl
from pkg_resources import resource_filename as pkgr_fn

from macaque_workflows.interfaces import qc
from macaque_workflows.interfaces.resample import ResampleImage
from macaque_workflows.interfaces.select_coreg import SelectCoregistration
from macaque_workflows.utils import check_params


def init_coregistration_wf(
    params: dict | None = None,
    wf_name: str = "coregistration_wf",
) -> pe.Workflow:
    """Estimate source to T1w coregistration transform

    The transfrom (source to T1w) estimated from this pipeline
    will be used to coregister all source data. Steps:
    1. Align the skullstripped T1w to source transform using FLIRT (rigid 6 DOF)
    2. Dilate the T1w brain mask and align to the source based on transform
       from step 1
    3. Skullstrip source image
    4. Perform final alignment using ANTs (rigid 6 DOF) of skullstripped source
       to skullstripped T1w (nb: source to T1w here works better than vice
       versa)

    QC outputs:
    1. An overlay .gif of the aligned source image

    Parameters
    ----------
    name : str
        Workflow name

    Returns
    -------
    from nipype.pipeline.engine.Workflow
        Coregistration workflow
    """
    if not params:
        params = pkgr_fn("macaque_workflows.data", "params.json")
    params = check_params(params, required={"coregistration_wf"})
    params = params["coregistration_wf"]
    print(params)

    wf = pe.Workflow(name=wf_name)
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1_img",
                "t1_brain",
                "t1_mask",
                "source_file",
            ],
        ),
        name="input_node",
    )

    # --- initial coregistrations for brain extraction ---
    t1_to_source = pe.Node(fsl.FLIRT(**params["t1_to_source"]), name="t1_to_source")
    wf.connect(input_node, "t1_img", t1_to_source, "in_file")
    wf.connect(input_node, "source_file", t1_to_source, "reference")

    t1_brain_to_source = pe.Node(
        fsl.FLIRT(**params["t1_to_source"]),
        name="t1_brain_to_source",
    )
    wf.connect(input_node, "t1_brain", t1_to_source, "in_file")
    wf.connect(input_node, "source_file", t1_to_source, "reference")

    # --- compare coregistrations and select best ---
    t1_list = pe.Node(niu.Merge(2), name="t1_list")
    wf.connect(t1_to_source, "out_file", t1_list, "in1")
    wf.connect(t1_brain_to_source, "out_file", t1_list, "in2")

    select_coreg = pe.Node(SelectCoregistration(), name="select_coreg", overwrite=True)
    wf.connect(t1_list, "out", select_coreg, "sources")
    wf.connect(input_node, "source_file", select_coreg, "target")

    xfm_list = pe.Node(niu.Merge(2), name="xfm_list")
    wf.connect(t1_to_source, "out_matrix_file", xfm_list, "in1")
    wf.connect(t1_brain_to_source, "out_matrix_file", xfm_list, "in1")

    get_best_xfm = pe.Node(niu.Select(), name="get_best_xfm")
    wf.connect(xfm_list, "out", get_best_xfm, "inlist")
    wf.connect(select_coreg, "source_index", get_best_xfm, "index")

    # --- template-based brain extraction ---
    dilate_mask = pe.Node(fsl.DilateImage(operation="modal"), name="dilate_mask")
    wf.connect(input_node, "t1_mask", dilate_mask, "in_file")

    mask_to_source = pe.Node(
        fsl.ApplyXFM(interp="nearestneighbour"),
        name="mask_to_source",
    )
    wf.connect(dilate_mask, "out_file", mask_to_source, "in_file")
    wf.connect(input_node, "source_file", mask_to_source, "reference")
    wf.connect(get_best_xfm, "out", mask_to_source, "in_matrix_file")

    brain_extract_source = pe.Node(
        fsl.ImageMaths(op_string="-mul"),
        name="brain_extract_source",
    )
    wf.connect(mask_to_source, "out_file", brain_extract_source, "in_file")
    wf.connect(input_node, "source_file", brain_extract_source, "in_file2")

    # --- final source to T1w coregistration (ANTs) ---
    source_to_t1 = pe.Node(
        ants.Registration(
            output_transform_prefix="source_to_T1w_",
            output_warped_image="source_in_T1w.nii.gz",
            **params["source_to_t1"],
        ),
        name="source_to_t1",
    )
    wf.connect(brain_extract_source, "out_file", source_to_t1, "moving_image")
    wf.connect(input_node, "t1_brain", source_to_t1, "fixed_image")

    # --- qc ---
    # do downsampling so that the QC source image is it's original resolution
    downsample_t1 = pe.Node(
        ResampleImage(dimension=3, interp=3, output_image="t1_resampled.nii.gz"),
        name="downsample_t1",
    )
    wf.connect(input_node, "t1_brain", downsample_t1, "input_image")
    wf.connect(input_node, "source_file", downsample_t1, "reference_image")

    resample_source = pe.Node(
        ants.ApplyTransforms(
            interpolation="BSpline",
        ),
        name="resample_source",
    )
    wf.connect(downsample_t1, "output_image", resample_source, "reference_image")
    wf.connect(source_to_t1, "composite_transform", resample_source, "transforms")
    wf.connect(brain_extract_source, "out_file", resample_source, "input_image")

    coreg_qc_inputs = pe.Node(niu.Merge(2), name="coreg_qc_inputs")
    wf.connect(input_node, "t1_brain", coreg_qc_inputs, "in1")
    wf.connect(resample_source, "output_image", coreg_qc_inputs, "in2")

    coreg_qc = pe.Node(qc.AlignmentQC(img_names=["T1w", "source"]), name="coreg_qc")
    wf.connect(coreg_qc_inputs, "out", coreg_qc, "input_imgs")

    output_node = pe.Node(
        niu.IdentityInterface(
            fields=[
                "init_t1_in_source",
                "t1_in_source",
                "t1_correlations",
                "source_in_t1",
                "source_mask",
                "source_to_t1_transform",
                "t1_to_source_transform",
                "coreg_qc",
            ],
        ),
        name="output_node",
    )
    wf.connect(t1_to_source, "out_file", output_node, "init_t1_in_source")
    wf.connect(source_to_t1, "inverse_warped_image", output_node, "t1_in_source")
    wf.connect(select_coreg, "correlations", output_node, "t1_correlations")
    wf.connect(resample_source, "output_image", output_node, "source_in_t1")
    wf.connect(mask_to_source, "out_file", output_node, "source_mask")
    wf.connect(
        source_to_t1,
        "composite_transform",
        output_node,
        "source_to_t1_transform",
    )
    wf.connect(
        source_to_t1,
        "inverse_composite_transform",
        output_node,
        "t1_to_source_transform",
    )
    wf.connect(coreg_qc, "out_file", output_node, "coreg_qc")

    return wf
