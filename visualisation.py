import argparse
import pickle
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path.home() / "Documents" / "simbarca_upload" / "simulation_sessions" / "session_000" / "timeseries"
PICKLE_CANDIDATES = ("agg_timeseries.pkl",)
CSV_CANDIDATES = ("timeseries.csv", "timeseries.txt", "timeseries.tsv", "timeseries")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect or plot a SimBarca timeseries table from a file or session folder."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="Path to a session directory, a timeseries directory, or a timeseries file.",
    )
    parser.add_argument(
        "--mode",
        choices=("table", "plot"),
        default="table",
        help="Display a table preview or generate a plot.",
    )
    parser.add_argument(
        "--key",
        default="pred_vdist",
        help="Dataset key to plot when loading agg_timeseries.pkl.",
    )
    parser.add_argument(
        "--section",
        type=int,
        help="Section id to plot. Defaults to the mean across all sections.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional output image path. Defaults next to the input data.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to show in table mode.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of columns to show in table mode when no section is selected.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening a Matplotlib window.",
    )
    return parser.parse_args()


def resolve_input_file(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidates = []
    if input_path.name.startswith("session_"):
        candidates.append(input_path / "timeseries")
    candidates.append(input_path)

    for candidate_dir in candidates:
        if not candidate_dir.is_dir():
            continue

        for filename in PICKLE_CANDIDATES + CSV_CANDIDATES:
            candidate_file = candidate_dir / filename
            if candidate_file.is_file():
                return candidate_file

    raise FileNotFoundError(
        f"No supported timeseries file found under: {input_path}"
    )


def load_pickle_timeseries(file_path: Path, key: str, section: int | None):
    with file_path.open("rb") as handle:
        payload = pickle.load(handle)

    if key not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(f"Unknown key '{key}'. Available keys: {available}")

    frame = payload[key]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame for '{key}', got {type(frame)}")

    if section is None:
        series = frame.mean(axis=1)
        ylabel = f"{key} mean across {frame.shape[1]} sections"
        title = f"{key} mean over time"
    else:
        if section not in frame.columns:
            preview = ", ".join(map(str, frame.columns[:10]))
            raise KeyError(
                f"Section {section} is not available in '{key}'. First section ids: {preview}"
            )
        series = frame[section]
        ylabel = f"{key} for section {section}"
        title = f"{key} over time for section {section}"

    return series, title, ylabel


def load_csv_timeseries(file_path: Path):
    separator = "\t" if file_path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(file_path, sep=separator)

    if "time" not in frame.columns or "value" not in frame.columns:
        raise ValueError(
            "CSV timeseries files must contain 'time' and 'value' columns."
        )

    series = pd.Series(frame["value"].to_numpy(), index=frame["time"])
    return series, "Timeseries", "value"


def load_series(file_path: Path, key: str, section: int | None):
    if file_path.suffix.lower() == ".pkl":
        return load_pickle_timeseries(file_path, key, section)
    return load_csv_timeseries(file_path)


def load_table(file_path: Path, key: str):
    if file_path.suffix.lower() == ".pkl":
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)

        if key not in payload:
            available = ", ".join(sorted(payload.keys()))
            raise KeyError(f"Unknown key '{key}'. Available keys: {available}")

        frame = payload[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame for '{key}', got {type(frame)}")
        return frame

    separator = "\t" if file_path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(file_path, sep=separator)


def build_output_path(file_path: Path, output: str | None, key: str, section: int | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()

    section_suffix = f"section_{section}" if section is not None else "mean"
    output_dir = Path(__file__).resolve().parent / "figure"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{file_path.stem}_{key}_{section_suffix}.png"


def show_table(frame: pd.DataFrame, rows: int, cols: int, section: int | None):
    if section is not None:
        if section not in frame.columns:
            preview = ", ".join(map(str, frame.columns[:10]))
            raise KeyError(
                f"Section {section} is not available. First section ids: {preview}"
            )
        preview_frame = frame[[section]].head(rows)
    else:
        preview_frame = frame.iloc[:rows, :cols]

    with pd.option_context(
        "display.max_rows", rows,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 40,
    ):
        print(f"Shape: {frame.shape}")
        print(f"Index type: {type(frame.index).__name__}")
        print(f"First columns: {list(frame.columns[:10])}")
        print()
        print(preview_frame)


def main():
    args = parse_args()

    try:
        file_path = resolve_input_file(Path(args.input_path))
        if args.mode == "table":
            frame = load_table(file_path, args.key)
        else:
            series, title, ylabel = load_series(file_path, args.key, args.section)
    except Exception as exc:
        print(f"Error loading file: {exc}")
        raise SystemExit(1) from exc

    if args.mode == "table":
        show_table(frame, rows=args.rows, cols=args.cols, section=args.section)
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.to_numpy(), label=ylabel)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = build_output_path(file_path, args.output, args.key, args.section)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
