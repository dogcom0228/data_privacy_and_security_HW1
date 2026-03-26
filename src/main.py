import argparse
import csv
from algorithms.mondrian import mondrian_k_anonymity
from utils.tools import read_csv, format_value


def main():
    parser = argparse.ArgumentParser(description="Mondrian k-anonymity")
    parser.add_argument("--input_file", type=str, default="data/adult.csv", help="Path to the input CSV file")
    parser.add_argument("--k", type=int, default=100, help="The k value for k-anonymity")
    args = parser.parse_args()

    data = read_csv(args.input_file)
    if not data:
        print("Input dataset is empty.")
        return

    qi_names = [
        "age",
        # "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        # "hours-per-week",
        "native-country",
    ]
    is_categorical = {
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
    }

    anonymized_data = mondrian_k_anonymity(
        data=data,
        qi_names=qi_names,
        k=args.k,
        is_categorical=is_categorical,
    )

    output_file = f"results/anonymized_data_k{args.k}.csv"

    # Write the anonymized data to the output file
    with open(output_file, "w", newline="") as f:
        if anonymized_data:
            headers = list(anonymized_data[0].keys())
            writer = csv.writer(f)
            writer.writerow(headers)
            for record in anonymized_data:
                writer.writerow([format_value(record.get(h)) for h in headers])


if __name__ == "__main__":
    main()
