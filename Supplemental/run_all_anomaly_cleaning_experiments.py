import contextlib
import gc
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


NOTEBOOK_NAMES = (
    "anomaly_cleaning_train_0-6_inject_7-9.ipynb",
    "anomaly_cleaning_train_0-8_inject_9.ipynb",
)
LATENT_SPACE_COUNTS = (5, 10)
LATENT_DIMENSION = 64


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def load_notebook_cells(notebook_path, latent_spaces):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = []
    config_cells = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if (
            re.search(r"(?m)^NUM_LATENT_SPACES\s*=", source)
            and re.search(r"(?m)^OUTPUT_DIR\s*=", source)
        ):
            config_cells += 1
            source, spaces_replacements = re.subn(
                r"(?m)^NUM_LATENT_SPACES\s*=\s*\d+\s*$",
                f"NUM_LATENT_SPACES = {latent_spaces}",
                source,
            )
            source, dimension_replacements = re.subn(
                r"(?m)^LATENT_DIM\s*=\s*\d+\s*$",
                f"LATENT_DIM = {LATENT_DIMENSION}",
                source,
            )
            output_assignment = (
                'OUTPUT_DIR = (Path("Iterations") / SCENARIO_TAG / f"{NUM_LATENT_SPACES} Spaces"\n'
                '              / f"{LATENT_DIM} Latent Dimensions")'
            )
            isolated_output_assignment = (
                'OUTPUT_DIR = (Path("Iterations") / SCENARIO_TAG / f"{NUM_LATENT_SPACES} Spaces"\n'
                '              / f"{LATENT_DIM} Latent Dimensions" / BATCH_ID)'
            )
            if (
                spaces_replacements != 1
                or dimension_replacements != 1
                or source.count(output_assignment) != 1
            ):
                raise ValueError(
                    f"Unexpected configuration layout in {notebook_path.name}; "
                    "expected one latent-space setting, latent dimension, and output path."
                )
            source = source.replace(output_assignment, isolated_output_assignment, 1)

        cells.append((cell.get("id", "unknown"), source))

    if config_cells != 1:
        raise ValueError(
            f"Expected exactly one configuration cell in {notebook_path.name}; "
            f"found {config_cells}."
        )
    return cells


def write_manifest(manifest_path, manifest):
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def execute_experiment(notebook_path, latent_spaces, batch_id, notebook_dir):
    cells = load_notebook_cells(notebook_path, latent_spaces)
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "BATCH_ID": batch_id,
    }
    original_directory = Path.cwd()
    configured = False

    try:
        os.chdir(notebook_dir)
        for cell_position, (cell_id, source) in enumerate(cells):
            if not configured:
                exec(compile(source, f"{notebook_path.name} [cell {cell_id}]", "exec"), namespace)
                if "OUTPUT_DIR" in namespace:
                    configured = True
                    output_dir = Path(namespace["OUTPUT_DIR"]).resolve()
                    namespace["OUTPUT_DIR"] = output_dir
                    output_dir.mkdir(parents=True, exist_ok=True)
                    log_file = (output_dir / "run.txt").open("w", encoding="utf-8")
                    namespace["_run_log_file"] = log_file
                    stdout = sys.stdout
                    try:
                        with contextlib.redirect_stdout(Tee(stdout, log_file)):
                            for remaining_cell_id, remaining_source in cells[cell_position + 1:]:
                                exec(
                                    compile(
                                        remaining_source,
                                        f"{notebook_path.name} [cell {remaining_cell_id}]",
                                        "exec",
                                    ),
                                    namespace,
                                )
                    finally:
                        log_file.close()
                    break
    finally:
        os.chdir(original_directory)

    if not configured:
        raise RuntimeError(f"No output directory was configured in {notebook_path.name}.")

    torch = namespace.get("torch")
    plt = namespace.get("plt")
    if plt is not None:
        plt.close("all")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    del namespace
    gc.collect()


def main(scenario_filter=None):
    notebook_dir = Path(__file__).resolve().parent
    scenario_notebooks = {
        "0-6": NOTEBOOK_NAMES[0],
        "0-8": NOTEBOOK_NAMES[1],
    }
    if scenario_filter is None:
        selected_notebooks = NOTEBOOK_NAMES
    elif scenario_filter in scenario_notebooks:
        selected_notebooks = (scenario_notebooks[scenario_filter],)
    else:
        raise ValueError(
            f"Unknown scenario filter {scenario_filter!r}; choose '0-6', '0-8', or None."
        )

    notebook_paths = [notebook_dir / name for name in selected_notebooks]
    missing = [path for path in notebook_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required source notebook not found: " + ", ".join(map(str, missing)))

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    iterations_root = notebook_dir / "Iterations"
    batch_root = iterations_root / "Batch Runs" / batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    manifest_path = batch_root / "run_manifest.json"
    manifest = {
        "batch_id": batch_id,
        "scenario_filter": scenario_filter,
        "latent_dimension": LATENT_DIMENSION,
        "latent_space_counts": list(LATENT_SPACE_COUNTS),
        "notebooks": list(selected_notebooks),
        "status": "running",
        "experiments": [],
    }
    write_manifest(manifest_path, manifest)

    total = len(notebook_paths) * len(LATENT_SPACE_COUNTS)
    print(f"Starting {total} experiments (latent dimension {LATENT_DIMENSION}).")
    print(f"Run manifest: {manifest_path}")
    print(f"Experiment results: {iterations_root / '<scenario>' / '<space count> Spaces' / '64 Latent Dimensions' / batch_id}")

    for notebook_path in notebook_paths:
        for latent_spaces in LATENT_SPACE_COUNTS:
            entry = {
                "notebook": notebook_path.name,
                "latent_spaces": latent_spaces,
                "status": "running",
            }
            manifest["experiments"].append(entry)
            write_manifest(manifest_path, manifest)
            print(f"\n[{len(manifest['experiments'])}/{total}] {notebook_path.name} - "
                  f"{latent_spaces} latent spaces")
            try:
                execute_experiment(notebook_path, latent_spaces, batch_id, notebook_dir)
            except Exception as error:
                entry["status"] = "failed"
                entry["error"] = f"{type(error).__name__}: {error}"
                manifest["status"] = "failed"
                write_manifest(manifest_path, manifest)
                raise
            entry["status"] = "completed"
            write_manifest(manifest_path, manifest)

    manifest["status"] = "completed"
    write_manifest(manifest_path, manifest)
    print(f"\nAll {total} experiments completed. Scenario-separated results are under: {iterations_root}")
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main(globals().get("SCENARIO_FILTER"))
