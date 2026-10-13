"""Plot only measured baseline/final accuracies, without interpolated checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.glm53_vision_sft.score import PROTOCOL, metrics_from_answers, read_rows

_RECORDED = Path(__file__).parent / "recorded"
_BASELINE_COLOR = "#64727a"
_TRAINED_COLOR = "#17826b"


def _axis_style(axis) -> None:
    axis.set_ylim(0, 100)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel("Exact-match accuracy (%)", fontsize=11, labelpad=12)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e5e9ec", linewidth=0.8)
    axis.tick_params(axis="both", length=0, labelsize=11, pad=9)
    for name in ("top", "right", "left"):
        axis.spines[name].set_visible(False)
    axis.spines["bottom"].set_color("#c7d0d6")


def _label_bars(axis, bars, correct: list[int], n: list[int], decimals: int = 2) -> None:
    for bar, count, total in zip(bars, correct, n, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{100 * count / total:.{decimals}f}%\n{count}/{total}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#253138",
            linespacing=1.35,
        )


def _validated_metrics(result: dict, answers: Path | None) -> dict:
    if result["protocol"] != PROTOCOL:
        raise ValueError("Unexpected scoring protocol")
    metrics = result["metrics"]
    if answers is not None and metrics_from_answers(read_rows(answers)) != metrics:
        raise ValueError("Reported metrics differ from the per-sample recorded answers")
    for group in (metrics, *metrics["by_sheet_kind"].values()):
        if not 0 <= group["baseline_correct"] <= group["n"] or not 0 <= group["trained_correct"] <= group["n"]:
            raise ValueError("Invalid correct/total counts")
    if sum(group["n"] for group in metrics["by_sheet_kind"].values()) != metrics["n"]:
        raise ValueError("Shape groups do not cover all held-out samples")
    return metrics


def plot(result_path: Path, output: Path, answers: Path | None = None) -> None:
    # Plotting is optional; data preparation and scoring need no Matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    result = json.loads(result_path.read_text())
    metrics = _validated_metrics(result, answers)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    figure, (overall, shapes) = plt.subplots(1, 2, figsize=(13.5, 7.5), gridspec_kw={"width_ratios": [1, 1.65]})
    figure.patch.set_facecolor("white")
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.27, top=0.71, wspace=0.32)
    figure.text(0.075, 0.93, "GLM-5.3-Flash | Vision-only SFT", fontsize=22, weight="bold", color="#182a31")
    figure.text(
        0.075,
        0.875,
        f"geometric_stack_count  /  {metrics['n']} held-out images  /  frozen language model",
        fontsize=12,
        color="#53646e",
    )
    figure.legend(
        handles=[
            Patch(color=_BASELINE_COLOR, label="Baseline (none-v2, iter 95)"),
            Patch(color=_TRAINED_COLOR, label="Vision-only (32 updates)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.069, 0.84),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    for axis in (overall, shapes):
        _axis_style(axis)
    overall.set_title("Overall", fontsize=13, loc="left", pad=18, weight="bold")
    counts = [metrics["baseline_correct"], metrics["trained_correct"]]
    bars = overall.bar(
        [0, 1], [100 * count / metrics["n"] for count in counts], width=0.58, color=[_BASELINE_COLOR, _TRAINED_COLOR]
    )
    _label_bars(overall, bars, counts, [metrics["n"]] * 2, decimals=3)
    overall.set_xticks([0, 1], ["Baseline", "Vision-only"])
    overall.set_xlim(-0.65, 1.65)
    gain = 100 * (counts[1] - counts[0]) / metrics["n"]
    overall.text(0.5, 88, f"{gain:+.3f} pp", ha="center", fontsize=21, weight="bold", color=_TRAINED_COLOR)
    overall.text(0.5, 79, f"{counts[1] - counts[0]:+d} correct answers", ha="center", fontsize=11, color="#53646e")
    shapes.set_title("By Target Sheet", fontsize=13, loc="left", pad=18, weight="bold")
    groups = metrics["by_sheet_kind"]
    kinds = ["circle", "rectangle", "triangle"]
    positions = list(range(len(kinds)))
    width = 0.33
    for model, offset, color in (("baseline", -width / 2, _BASELINE_COLOR), ("trained", width / 2, _TRAINED_COLOR)):
        correct = [groups[kind][f"{model}_correct"] for kind in kinds]
        totals = [groups[kind]["n"] for kind in kinds]
        bars = shapes.bar(
            [position + offset for position in positions],
            [100 * count / total for count, total in zip(correct, totals, strict=True)],
            width=width,
            color=color,
        )
        _label_bars(shapes, bars, correct, totals)
    shapes.set_xticks(positions, [f"{kind.title()}\nn={groups[kind]['n']}" for kind in kinds])
    figure.text(
        0.075,
        0.15,
        f"Paired changes: {metrics['wrong_to_right']} wrong-to-right, {metrics['right_to_wrong']} right-to-wrong."
        f"  Valid in both: {metrics['net_correct_change_both_valid']:+d}; format transitions:"
        f" {metrics['net_correct_change_validity_transitions']:+d}.",
        fontsize=10.5,
        color="#53646e",
    )
    figure.text(
        0.075,
        0.103,
        "Baseline is the inherited none-v2 checkpoint, not untouched pretrained GLM. One run; no error bars.",
        fontsize=10.5,
        color="#53646e",
    )
    figure.text(
        0.075,
        0.056,
        f"Scoring: {PROTOCOL}. No intermediate eval points were measured.",
        fontsize=10,
        color="#71808a",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=_RECORDED / "result.json")
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--output", type=Path, default=_RECORDED / "eval_gain.png")
    args = parser.parse_args()
    plot(args.result, args.output, args.answers)
    print(args.output)


if __name__ == "__main__":
    main()
