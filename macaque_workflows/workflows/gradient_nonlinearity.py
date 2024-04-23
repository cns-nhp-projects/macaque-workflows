from __future__ import annotations

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces import fsl

from macaque_workflows.interfaces.gradunwarp import GradUnwarp


def init_gradient_nonlinearity_wf(
    coeff_file: str,
    ref_vol: int = 0,
    warp_field_only: bool = False,
    wf_name: str = "gdc_wf",
) -> pe.Workflow:

    wf = pe.Workflow(wf_name)
    wf.config["execution"] = {"remove_unnecessary_outputs": False}

    input_node = pe.Node(niu.IdentityInterface(fields=["in_file"]), name="input_node")

    # select first volume
    reference = pe.Node(fsl.ExtractROI(t_min=ref_vol, t_size=1), name="reference")
    wf.connect(input_node, "in_file", reference, "in_file")

    # gradient_unwarp
    gradient_unwarp = pe.Node(GradUnwarp(coeffs=coeff_file), name="gradient_unwarp")
    wf.connect(reference, "roi_file", gradient_unwarp, "in_file")

    # convert warp
    warpfield = pe.Node(
        fsl.ConvertWarp(
            abswarp=True,
            out_relwarp=True,
        ),
        name="warpfield",
    )
    wf.connect(gradient_unwarp, "out_file", warpfield, "reference")
    wf.connect(gradient_unwarp, "out_warp", warpfield, "warp1")

    output_node = pe.Node(
        niu.IdentityInterface(
            fields=["out_file", "warp_field"],
            mandatory_inputs=False,
        ),
        name="output_node",
    )
    wf.connect(warpfield, "out_file", output_node, "warp1")

    if not warp_field_only:

        # apply warp w/ spline
        apply_warp = pe.Node(
            fsl.ApplyWarp(relwarp=True, interp="spline"),
            name="apply_warp",
        )
        wf.connect(warpfield, "out_file", apply_warp, "field_file")
        wf.connect(reference, "roi_file", apply_warp, "ref_file")
        wf.connect(input_node, "in_file", apply_warp, "in_file")
        wf.connect(apply_warp, "out_file", output_node, "out_file")

    return wf
