import os
import subprocess

import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn import image
from nilearn import plotting
from nilearn._utils import check_niimg
from nipype.interfaces.base import BaseInterfaceInputSpec
from nipype.interfaces.base import File
from nipype.interfaces.base import InputMultiPath
from nipype.interfaces.base import SimpleInterface
from nipype.interfaces.base import TraitedSpec
from nipype.interfaces.base import traits


def _find_cut_coords(img):
    return {
        "x": plotting.find_cut_slices(img, "x", 9),
        "y": plotting.find_cut_slices(img, "y", 9),
        "z": plotting.find_cut_slices(img, "z", 9),
    }


def _plot_mosaic(
    img,
    plot_type="anat",
    cut_coords=None,
    title=None,
    figsize=None,
    **kwargs,
):

    if plot_type == "anat":
        plot_func = plotting.plot_anat
    elif plot_type == "epi":
        plot_func = plotting.plot_epi
    else:
        raise ValueError("plot_type must be either 'anat' or 'epi'")

    if cut_coords is None:
        cut_coords = _find_cut_coords(img)

    if figsize is None:
        n_slices = max(len(i) for i in cut_coords.values())
        figsize = (1.3 * n_slices, 1.2 * 3)

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0)

    panels = []
    for ax, view in zip(axes, ["x", "y", "z"]):
        g = plot_func(
            img,
            display_mode=view,
            cut_coords=cut_coords[view],
            axes=ax,
            annotate=False,
            draw_cross=False,
            **kwargs,
        )
        g.annotate(size=6)
        if view == "x" and title is not None:
            g.title(title, size=10, color="w", alpha=0)
        panels.append(g)

    return fig, panels


def _alignment_gif(imgs, out, delay=15, morph=10, smooth_loop=True):

    input_imgs = imgs
    # add first to create smooth transition when loop restarts
    if smooth_loop:
        input_imgs.append(input_imgs[0])

    input_imgs = " ".join(input_imgs)
    cmd = f"convert -delay {delay} {input_imgs} -morph {morph} -loop 0 {out}"
    subprocess.run(cmd.split())


class _AnatSegmentationQCInputSpec(BaseInterfaceInputSpec):
    t1_img = File(exists=True, mandatory=True, desc="File path of native T1w image")
    segment_label_img = File(
        exists=True,
        mandatory=True,
        desc="File path of segmentation label image",
    )


class _AnatSegmentationQCOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="T1 segmentation QC image")


class AnatSegmentationQC(SimpleInterface):

    input_spec = _AnatSegmentationQCInputSpec
    output_spec = _AnatSegmentationQCOutputSpec

    def _run_interface(self, runtime):

        t1_img = check_niimg(self.inputs.t1_img)
        segment_img = check_niimg(self.inputs.segment_label_img)
        wm_gm_img = image.threshold_img(segment_img, 1.5)

        fig, panels = _plot_mosaic(t1_img)
        for p in panels:
            p.add_contours(wm_gm_img, levels=[1, 2], colors=["r", "b"], linewidths=0.3)

        self._results["out_file"] = os.path.join(runtime.cwd, "T1w_segmentation.png")
        fig.savefig(self._results["out_file"], dpi=200)

        return runtime


class _AnatMaskQCInputSpec(BaseInterfaceInputSpec):
    t1_img = File(exists=True, mandatory=True, desc="File path of native T1w image")
    mask_img = File(exists=True, mandatory=True, desc="File path of mask image")


class _AnatMaskQCOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="T1 segmentation QC image")


class AnatMaskQC(SimpleInterface):
    input_spec = _AnatMaskQCInputSpec
    output_spec = _AnatMaskQCOutputSpec

    def _run_interface(self, runtime):

        t1_img = check_niimg(self.inputs.t1_img)
        mask_img = check_niimg(self.inputs.mask_img)
        fig, panels = _plot_mosaic(t1_img)
        for p in panels:
            p.add_contours(mask_img, levels=[2], colors=["r"], linewidths=0.5)
        self._results["out_file"] = os.path.join(runtime.cwd, "T1w_mask.png")
        fig.savefig(self._results["out_file"], dpi=200)
        return runtime


class _AlignmentQCInputSpec(BaseInterfaceInputSpec):
    input_imgs = traits.List(minlen=2, mandatory=True)
    img_names = traits.List()
    img_type = traits.String("anat", usedefault=True)


class _AlignmentQCOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="T1 segmentation QC image")


class AlignmentQC(SimpleInterface):

    input_spec = _AlignmentQCInputSpec
    output_spec = _AlignmentQCOutputSpec

    def _run_interface(self, runtime):

        input_imgs = [check_niimg(i) for i in self.inputs.input_imgs]
        img_names = self.inputs.img_names
        img_type = self.inputs.img_type

        if not img_names:
            img_names = [f"img{i + 1}" for i in range(len(input_imgs))]

        cut_coords = _find_cut_coords(input_imgs[0])
        tmp_files = []
        for name, img in zip(img_names, input_imgs):
            fig, _ = _plot_mosaic(
                img,
                plot_type=img_type,
                cut_coords=cut_coords,
                title=name,
            )
            out = os.path.join(runtime.cwd, f"{name}.png")
            fig.savefig(out, dpi=200)
            tmp_files.append(out)

        out_file = os.path.join(runtime.cwd, "alignment.gif")
        _alignment_gif(tmp_files, out_file)
        self._results["out_file"] = out_file

        return runtime


class _ImageQCInputSpec(BaseInterfaceInputSpec):
    input_img = File()
    img_type = traits.String("anat", usedefault=True)


class _ImageQCOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Mosaic plot image")


class ImageQC(SimpleInterface):

    input_spec = _ImageQCInputSpec
    output_spec = _ImageQCOutputSpec

    def _run_interface(self, runtime):

        input_img = self.inputs.input_img
        img_type = self.inputs.img_type
        fig, _ = _plot_mosaic(input_img, plot_type=img_type)

        self._results["out_file"] = os.path.join(runtime.cwd, "image.png")
        fig.savefig(self._results["out_file"], dpi=200)
        return runtime


class _TraceOverlayQCInputSpec(BaseInterfaceInputSpec):
    input_img = File(exists=True, mandatory=True, desc="File path of native T1w image")
    overlay_imgs = InputMultiPath()
    cmap = traits.String("rainbow", usedefault=True)


class _TraceOverlayQCOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="T1 segmentation QC image")


class TraceOverlayQC(SimpleInterface):

    input_spec = _TraceOverlayQCInputSpec
    output_spec = _TraceOverlayQCOutputSpec

    def _run_interface(self, runtime):

        input_img = self.inputs.input_img
        overlays = self.inputs.overlay_imgs
        cmap = self.inputs.cmap

        n_overlays = len(overlays)
        colors = cm.get_cmap(cmap, n_overlays)
        colors = colors(range(n_overlays))

        fig, panels = _plot_mosaic(input_img)
        for p in panels:
            for img, c in zip(overlays, colors):
                p.add_contours(img, levels=[1], colors=[c], linewidths=0.3)

        self._results["out_file"] = os.path.join(runtime.cwd, "traces.png")
        fig.savefig(self._results["out_file"], dpi=200)

        return runtime
