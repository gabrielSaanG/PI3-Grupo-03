from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gradcam import GradCAMResult

PrepareFn = Callable[[pd.Series], Tuple]
GenerateFn = Callable[..., Union[GradCAMResult, Tuple[np.ndarray, float, float]]]


def normalize_image_for_display(img: np.ndarray, mode: str = "percentile") -> np.ndarray:
    if mode == "percentile":
        vmin = float(np.percentile(img, 1))
        vmax = float(np.percentile(img, 99))
    else:
        vmin = float(img.min())
        vmax = float(img.max())

    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)

    img_disp = (img - vmin) / (vmax - vmin)
    return np.clip(img_disp, 0, 1).astype(np.float32)


def overlay_cam(
    ax,
    img_disp: np.ndarray,
    cam: np.ndarray,
    *,
    threshold: float = 0.35,
    alpha: float = 0.55,
    cmap: str = "jet",
) -> None:
    ax.imshow(img_disp, cmap="gray", interpolation="nearest")
    cam_masked = np.ma.masked_where(cam < threshold, cam)
    ax.imshow(
        cam_masked,
        cmap=cmap,
        alpha=alpha,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )


def add_marker(
    ax,
    img_shape: Tuple[int, int] = (64, 64),
    center_x: Optional[float] = None,
    center_y: Optional[float] = None,
    radius: float = 9,
    color: str = "cyan",
) -> None:
    h, w = img_shape
    if center_x is None:
        center_x = w / 2
    if center_y is None:
        center_y = h / 2

    circle = patches.Circle(
        (center_x, center_y),
        radius=radius,
        linewidth=1.8,
        edgecolor=color,
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(circle)


# Back-compat aliases from the notebook
normalise_for_display = normalize_image_for_display
add_nodule_marker = add_marker


def _unpack_prepare(prepared: Sequence) -> Tuple[np.ndarray, Tuple]:
    if len(prepared) < 2:
        raise ValueError(
            f"{len(prepared)} value(s)."
        )
    *model_inputs, raw_img = prepared
    if not isinstance(raw_img, np.ndarray):
        raise TypeError("prepare_fn last return value must be the raw image ndarray.")
    return raw_img, tuple(model_inputs)


def _unpack_result(
    result: Union[GradCAMResult, Tuple[np.ndarray, float, float]]
) -> Tuple[np.ndarray, float, float]:
    if isinstance(result, GradCAMResult):
        return result.cam, result.prob, result.logit
    cam, prob, logit = result
    return cam, float(prob), float(logit)


def plot_gradcam_grid(
    cases_df: pd.DataFrame,
    *,
    prepare_fn: PrepareFn,
    generate_fn: GenerateFn,
    title: str = "Grad-CAM Explanations",
    save_path: Optional[str] = None,
    display_mode: str = "percentile",
    marker_radius: float = 9,
    raw_marker_color: str = "red",
    overlay_marker_color: str = "cyan",
    cam_threshold: float = 0.35,
    show: bool = True,
) -> Optional[plt.Figure]:
    n = len(cases_df)
    if n == 0:
        print("[skip] No cases to plot for Grad-CAM grid.")
        return None

    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 3.8 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for plot_i, (_, row) in enumerate(cases_df.iterrows()):
        prepared = prepare_fn(row)
        raw_img, model_inputs = _unpack_prepare(prepared)
        cam, prob, _logit = _unpack_result(generate_fn(*model_inputs))
        img_disp = normalize_image_for_display(raw_img, mode=display_mode)

        true_label = "Malignant" if int(row["label"]) == 1 else "Benign"
        pred_label = "Malignant" if int(row["pred"]) == 1 else "Benign"
        case_type = row.get("case_type", "")
        malignancy = row.get("malignancy_mean", float("nan"))

        subtitle = (
            f"{case_type} | True: {true_label} | Pred: {pred_label}\n"
            f"P={prob:.3f} | malignancy_mean={malignancy:.2f}"
        )

        axes[plot_i, 0].imshow(img_disp, cmap="gray", interpolation="nearest")
        add_marker(
            axes[plot_i, 0],
            img_shape=raw_img.shape,
            radius=marker_radius,
            color=raw_marker_color,
        )
        axes[plot_i, 0].axis("off")
        axes[plot_i, 0].set_title("CT crop\n" + subtitle, fontsize=10, pad=8)

        overlay_cam(
            axes[plot_i, 1],
            img_disp,
            cam,
            threshold=cam_threshold,
        )
        add_marker(
            axes[plot_i, 1],
            img_shape=raw_img.shape,
            radius=marker_radius,
            color=overlay_marker_color,
        )
        axes[plot_i, 1].axis("off")
        axes[plot_i, 1].set_title("Grad-CAM overlay\n" + subtitle, fontsize=10, pad=8)

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.subplots_adjust(top=0.94, hspace=0.50, wspace=0.08)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
