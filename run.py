"""One portable entry point for local and JarvisLabs workflows.

Run `python run.py --help`. Heavy EEG imports are delayed until a data or
training command, so the first warm-up ZIP needs only the small core stack.
"""

import argparse
import json
from pathlib import Path

from track2 import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    warmup = commands.add_parser("warmup", help="build and validate the first constant-reference ZIP")
    warmup.add_argument("--output", type=Path, default=ROOT / "artifacts" / "warmup_constant.zip")
    validate = commands.add_parser("validate", help="validate a flat Codabench ZIP")
    validate.add_argument("zip", type=Path)
    package = commands.add_parser("package", help="package a completed trained run")
    package.add_argument("--run", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="run offline contract, metric, training and export checks")
    smoke.add_argument("--include-eegnet", action="store_true", help="also require and train/test Braindecode EEGNet")
    smoke.add_argument("--tier1", action="store_true", help="also fit/reload/export every Tier 1 preset on synthetic windows")
    commands.add_parser("doctor", help="inspect installed packages and hardware without downloading")
    for action in ("prepare", "audit"):
        child = commands.add_parser(action, help=f"{action} the public PROTEUS release")
        child.add_argument("--data-dir", type=Path, default=ROOT / "data")
    train = commands.add_parser("train", help="fit one decoder and select neural checkpoints on validation")
    choice = train.add_mutually_exclusive_group(required=True)
    choice.add_argument("--config", type=Path, help="Tier 1 experiment JSON; reserves P10 later sessions")
    choice.add_argument("--model", choices=[
        "linear", "eegnet", "eegnet-source-zscore", "eegnet-window-robust",
        "bandpower", "bandpower-low", "covariance-diagonal", "covariance",
    ])
    train.add_argument("--protocol", choices=["screen", "outer"], default="screen")
    train.add_argument("--fold", type=int, choices=range(4), help="required for outer protocol")
    train.add_argument("--data", choices=["proteus", "simulated"], default="proteus")
    train.add_argument("--data-dir", type=Path, default=ROOT / "data")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int, default=33)
    train.add_argument("--epochs", type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--reserve-p10-later", action="store_true", help="reserve P10 sessions 2-3 for longitudinal diagnostics")
    evaluate = commands.add_parser("evaluate", help="reload and score a saved run without fitting")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--data", choices=["proteus", "simulated"], default="proteus")
    evaluate.add_argument("--data-dir", type=Path, default=ROOT / "data")
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    evaluate.add_argument("--split", choices=["val", "test", "longitudinal", "diagnostic"], default="test")
    batch = commands.add_parser("experiment-batch", help="legacy CPU/EEGNet batch; all six experiments use train --config")
    batch.add_argument("--phase", choices=["cpu", "neural"], default="cpu")
    batch.add_argument("--data-dir", type=Path, default=ROOT / "data")
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--seed", type=int, default=33)
    batch.add_argument("--epochs", type=int)
    batch.add_argument("--lr", type=float)
    batch.add_argument("--batch-size", type=int, default=32)
    batch.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                       help="neural phase device; CPU phase always stays on CPU")
    batch.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.command in ("warmup", "package", "validate"):
        from track2 import packaging
        if args.command == "validate":
            print(json.dumps(packaging.validate(args.zip), indent=2))
        else:
            packaging.package(args.output, getattr(args, "run", None))
    elif args.command == "smoke":
        from track2.smoke import smoke
        smoke(include_eegnet=args.include_eegnet, tier1=args.tier1)
    elif args.command == "doctor":
        from track2.workflow import doctor
        doctor()
    elif args.command in ("prepare", "audit"):
        from track2.data import prepare, audit
        (prepare if args.command == "prepare" else audit)(args.data_dir)
    elif args.command == "train":
        from track2.workflow import train
        train(args)
    elif args.command == "evaluate":
        from track2.workflow import evaluate_run
        evaluate_run(args)
    elif args.command == "experiment-batch":
        from track2.workflow import experiment_batch
        experiment_batch(args)


if __name__ == "__main__":
    main()
