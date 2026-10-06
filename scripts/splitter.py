#!/usr/bin/env python3

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Split the backend column into backend and communication columns."
    )
    parser.add_argument("file", help="Path to the CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    # Split at the last underscore:
    # "kc_mpi_nonblocking" -> "kc_mpi", "nonblocking"
    df[["backend", "communication"]] = df["backend"].str.rsplit(
        "_", n=1, expand=True
    )

    # Overwrite the input file
    df.to_csv(args.file, index=False)


if __name__ == "__main__":
    main()