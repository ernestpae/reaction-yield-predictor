"""Command-line interface: train once, then predict with the saved pipeline."""
import argparse
import json

from .experiment import predict, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Run model selection and holdout evaluation")
    training.add_argument("--data", default="data/reaction_data.csv")
    training.add_argument("--output", default="artifacts/random")
    training.add_argument("--split", choices=["random", "high_temperature"], default="random")
    training.add_argument("--seed", type=int, default=42)
    inference = commands.add_parser("predict", help="Predict using your trusted saved model")
    inference.add_argument("--model", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.data, args.output, strategy=args.split, seed=args.seed)
            print(json.dumps({k: result[k] for k in ["selected_model", "holdout", "baseline_holdout"]}, indent=2))
        else:
            result = predict(args.model, args.input, args.output)
            print(f"Wrote {len(result)} predictions; {result.outside_training_range.sum()} outside training ranges")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
