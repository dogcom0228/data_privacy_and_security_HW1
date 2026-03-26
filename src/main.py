import argparse
import csv
from typing import List

from algorithms.mondrian import mondrian_k_anonymity
from utils.tools import read_csv, format_value, preprocess_adult
from classes.train import train
from classes.test import test


def run_mondrian(args):
    """執行 Mondrian k-anonymity"""
    data = read_csv(args.input_file)
    if not data:
        print("Input dataset is empty.")
        return

    anonymized_data = mondrian_k_anonymity(
        data=data,
        qi_names=args.qi_names,
        k=args.k,
        is_categorical=args.is_categorical,
    )

    output_file = f"results/anonymized_data_k{args.k}.csv"
    with open(output_file, "w", newline="") as f:
        if anonymized_data:
            headers = list(anonymized_data[0].keys())
            writer = csv.writer(f)
            writer.writerow(headers)
            for record in anonymized_data:
                writer.writerow([format_value(record.get(h)) for h in headers])

    print(f"Anonymized data saved to {output_file}")


def run_train(args):
    print(f"Training model: {args.model_name}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.lr}")

    X, y, columns = preprocess_adult(args.dataset)
    args.columns = columns
    train(X, y, args)


def run_test(args):
    print(f"Testing model: {args.model_path}")
    print(f"  Test data:    {args.test_data}")
    print(f"  Batch size:   {args.batch_size}")

    if args.model_path.endswith(".pkl"):
        import joblib

        ckpt = joblib.load(args.model_path)
        columns = ckpt.get("columns", None)
    else:
        import torch

        ckpt = torch.load(args.model_path, map_location="cpu")
        columns = ckpt.get("columns", None)

    X, y, _ = preprocess_adult(args.test_data, columns=columns)
    test(X, y, args)


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-preserving pipeline: anonymize, train, or test"
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands",
    )

    parser_mondrian = subparsers.add_parser("mondrian", help="Run Mondrian k-anonymity")
    parser_mondrian.add_argument(
        "--input_file",
        type=str,
        default="data/adult.data",
        help="Path to the input CSV file",
    )
    parser_mondrian.add_argument(
        "--qi_names",
        type=List[str],
        default=[
            "age",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
        ],
        help="List of quasi-identifier column names",
    )
    parser_mondrian.add_argument(
        "--is_categorical",
        type=dict,
        default={
            "age": False,
            "hours-per-week": False,
            "workclass": True,
            "education": True,
            "marital-status": True,
            "occupation": True,
            "relationship": True,
            "race": True,
            "gender": True,
            "native-country": True,
        },
        help="Dictionary mapping quasi-identifier column names to their categorical status",
    )
    parser_mondrian.add_argument(
        "--k",
        type=int,
        default=100,
        help="The k value for k-anonymity",
    )
    parser_mondrian.set_defaults(func=run_mondrian)

    parser_train = subparsers.add_parser("train", help="Train an AI model")
    parser_train.add_argument(
        "--dataset",
        type=str,
        default="data/adult.data",
        help="Path to the training dataset",
    )
    parser_train.add_argument(
        "--model_name",
        type=str,
        default="adult",
        help="Name of the model to train",
    )
    parser_train.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser_train.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser_train.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser_train.add_argument(
        "--loss_type",
        type=str,
        default="weighted_bce",
        choices=["bce", "weighted_bce", "focal", "ldam"],
        help="Loss function to use",
    )
    parser_train.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "xgboost"],
        help="Model type to use",
    )
    parser_train.set_defaults(func=run_train)

    parser_test = subparsers.add_parser("test", help="Test an AI model")
    parser_test.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser_test.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to the test dataset",
    )
    parser_test.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Test batch size",
    )
    parser_test.set_defaults(func=run_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
