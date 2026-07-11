from pathlib import Path

import pandas as pd


SOURCE_PATH = Path("data/training_data.csv")
OUTPUT_PATH = Path("data/incoming/batch_001.csv")


def main() -> None:
    df = pd.read_csv(SOURCE_PATH)

    batch = df.sample(
        n=100,
        random_state=42,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"Batch saved to: {OUTPUT_PATH.resolve()}")
    print(f"Rows: {len(batch)}")


if __name__ == "__main__":
    main()