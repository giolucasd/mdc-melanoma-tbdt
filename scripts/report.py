import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -----------------------------------------
# TensorBoard reading utilities
# -----------------------------------------


def load_scalars(event_dir: str | Path) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load all TensorBoard scalar metrics found in event files in a directory.
    """
    event_dir = Path(event_dir)
    event_files = list(event_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events found in {event_dir}")

    acc = EventAccumulator(str(event_files[0]))
    acc.Reload()

    scalars: Dict[str, List[Tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]

    return scalars


def load_figures(event_dir: str | Path) -> Dict[str, List[Tuple[int, bytes]]]:
    """
    Load all TensorBoard figures found in event files in a directory.
    """
    event_dir = Path(event_dir)
    event_files = list(event_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard events found in {event_dir}")

    acc = EventAccumulator(str(event_files[0]))
    acc.Reload()

    figures: Dict[str, List[Tuple[int, bytes]]] = {}
    for tag in acc.Tags().get("images", []):
        images = acc.Images(tag)
        figures[tag] = [(img.step, img.encoded_image_string) for img in images]

    return figures


def extract_curve(
    scalars: Dict[str, List[Tuple[int, float]]], tag: str
) -> Tuple[List[int], List[float]]:
    """
    Extract (x,y) from tensorboard scalar dict.
    """
    if tag not in scalars:
        return [], []
    xs = [s for (s, v) in scalars[tag]]
    ys = [v for (s, v) in scalars[tag]]
    return xs, ys


def extract_first_available_curve(
    scalars: Dict[str, List[Tuple[int, float]]], tags: List[str]
) -> Tuple[List[int], List[float]]:
    for t in tags:
        xs, ys = extract_curve(scalars, t)
        if xs:
            return xs, ys
    return [], []


def find_best_step(
    val_loss_steps: List[int], val_loss_values: List[float]
) -> Optional[int]:
    """
    Identify the step with lowest validation loss.
    """
    if not val_loss_values:
        return None
    best_idx = min(range(len(val_loss_values)), key=lambda i: val_loss_values[i])
    return val_loss_steps[best_idx]


def get_value_at_step(
    scalars: Dict[str, List[Tuple[int, float]]], tag: str, step: int
) -> Optional[float]:
    """
    Get metric value for specific step.
    """
    if tag not in scalars:
        return None
    for s, v in scalars[tag]:
        if s == step:
            return v
    return None


# -----------------------------------------
# Plotting utilities
# -----------------------------------------


def save_confusion_matrix(
    figures: Dict[str, List[Tuple[int, bytes]]],
    tag: str,
    best_step: int,
    save_path: Path,
) -> None:
    if tag not in figures:
        print("No confusion matrix found in logs.")
        return

    # Find the closest step (TensorBoard sometimes saves between steps)
    available = sorted(figures[tag], key=lambda t: abs(t[0] - best_step))
    step, img_bytes = available[0]

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    print(f"Saved confusion matrix (closest step={step}) to: {save_path}")


def plot_multi_runs(
    runs: Dict[str, Dict[str, Tuple[List[int], List[float]]]],
    metric_name: str,
    save_path: Path,
) -> None:
    """
    Plot multiple runs in a single figure.

    Each run contributes 2 curves: train and val.
    E.g. metric_name = "loss" plots:
        run1 train_loss
        run1 val_loss
        run2 train_loss
        run2 val_loss
        ...
    """
    plt.figure()

    for run_name, data in runs.items():
        train_x, train_y = data.get(f"train_{metric_name}", ([], []))
        val_x, val_y = data.get(f"val_{metric_name}", ([], []))

        if train_x:
            plt.plot(train_x, train_y, label=f"{run_name} – train")
        if val_x:
            plt.plot(val_x, val_y, label=f"{run_name} – val")

    plt.title(metric_name.capitalize())
    plt.xlabel("Step")
    plt.ylabel(metric_name.capitalize())
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_metrics_yaml(metrics: Dict[str, Dict[str, any]], save_path: Path) -> None:
    """
    Save metrics dictionary to YAML file.
    """
    with open(save_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"Saved metrics to: {save_path}")


# -----------------------------------------
# Main
# -----------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot combined TensorBoard metrics for multiple training stages"
    )
    parser.add_argument(
        "--output-suffixes",
        type=str,
        required=True,
        help="Comma-separated list of experiment names, e.g.: frozen,full",
    )
    args = parser.parse_args()

    suffixes = [s.strip() for s in args.output_suffixes.split(",")]

    # Data structure:
    # runs[run_name][metric] = (x, y)
    runs: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}
    metrics_report: Dict[str, Dict[str, any]] = {}

    print("\nLoading TensorBoard logs...\n")

    for suffix in suffixes:
        run_dir = Path("outputs") / suffix
        tb_dir = run_dir / "tensorboard"
        print(f"  • {suffix}: reading {tb_dir}")

        scalars = load_scalars(tb_dir)
        figures = load_figures(tb_dir)

        run_data = {
            "train_loss": extract_first_available_curve(
                scalars, ["train_loss", "train_loss_step", "train_loss_epoch"]
            ),
            "val_loss": extract_first_available_curve(
                scalars, ["val_loss", "val_loss_step", "val_loss_epoch"]
            ),
            "train_bal_acc": extract_first_available_curve(
                scalars, ["train_bal_acc", "train_bal_acc_step", "train_bal_acc_epoch"]
            ),
            "val_bal_acc": extract_first_available_curve(
                scalars, ["val_bal_acc", "val_bal_acc_step", "val_bal_acc_epoch"]
            ),
        }

        runs[suffix] = run_data

        # Identify best step for this run
        val_loss_x, val_loss_y = run_data["val_loss"]
        best_step = find_best_step(val_loss_x, val_loss_y)
        if best_step is None:
            print(f"    No val_loss found for {suffix}.\n")
            continue

        print(f"    Best step: {best_step}")
        print("    Metrics at best step:")

        metrics_dict = {"best_step": best_step}

        for m in [
            "val_acc",
            "val_bal_acc",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_fbeta",
        ]:
            v = get_value_at_step(scalars, m, best_step)
            print(f"        {m}: {v}")
            metrics_dict[m] = v

        metrics_report[suffix] = metrics_dict

        # Save confusion matrix
        save_confusion_matrix(
            figures, "val_confusion_matrix", best_step, run_dir / "confusion_matrix.png"
        )
        print("")

    # ---------------------------------------------------------
    # Final combined plots
    # ---------------------------------------------------------

    # Where to save merged plots: in the final run directory
    final_dir = Path("outputs") / suffixes[-1]

    print(f"Saving combined plots to: {final_dir}\n")

    # loss_curve.png (train + val for all runs)
    plot_multi_runs(runs, "loss", final_dir / "loss_curve.png")

    # balanced_accuracy_curve.png (train + val for all runs)
    plot_multi_runs(runs, "bal_acc", final_dir / "balanced_accuracy_curve.png")

    # Save metrics report as YAML
    save_metrics_yaml(metrics_report, final_dir / "metrics_report.yaml")


if __name__ == "__main__":
    main()
