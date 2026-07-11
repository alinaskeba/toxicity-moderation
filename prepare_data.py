from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = (
    "thesofakillers/"
    "jigsaw-toxic-comment-classification-challenge"
)

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

OUTPUT_PATH = Path("data/training_data.csv")


def prepare_data() -> pd.DataFrame:
    """Download and prepare the binary toxicity dataset."""

    print("Downloading training dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    df = dataset.to_pandas()

    print(f"Raw dataset shape: {df.shape}")

    required_columns = ["comment_text", *LABEL_COLUMNS]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["label"] = (
        df[LABEL_COLUMNS]
        .max(axis=1)
        .astype(int)
    )

    prepared_df = (
        df[["comment_text", "label"]]
        .rename(columns={"comment_text": "text"})
        .dropna(subset=["text"])
        .copy()
    )

    prepared_df["text"] = (
        prepared_df["text"]
        .astype(str)
        .str.strip()
    )

    prepared_df = prepared_df[
        prepared_df["text"] != ""
    ]

    prepared_df = (
        prepared_df
        .drop_duplicates(subset=["text"])
        .reset_index(drop=True)
    )

    print(f"Prepared dataset shape: {prepared_df.shape}")

    print("\nClass counts:")
    print(prepared_df["label"].value_counts())

    print("\nClass shares:")
    print(
        prepared_df["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    return prepared_df


def save_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the prepared dataset to CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print(f"\nDataset saved to: {output_path.resolve()}")


def main() -> None:
    prepared_df = prepare_data()
    save_data(prepared_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()