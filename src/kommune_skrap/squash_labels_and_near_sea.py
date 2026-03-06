"""
Load labels and near_sea files and join the tables.
Then, the rows in the table that share the common Hash are squashed into one row,
using the Avgjørelse that is not u (if any), retaining all URL values as a list,
and the latest Dato value, and the Filnavn for that latest Dato. The resulting table is saved as labels_squashed.csv.
"""

from pathlib import Path

import pandas as pd


def squash_labels_and_near_sea(
    labels_file: Path, near_sea_file: Path, output_file: Path
) -> None:
    """Load, join, squash, and save labels and near_sea data."""
    labels_df = pd.read_csv(labels_file, parse_dates=["Dato"])
    near_sea_df = pd.read_csv(near_sea_file)

    # Join on filename
    merged = labels_df.merge(
        near_sea_df, left_on="Filnavn", right_on="filename", how="left"
    ).drop(columns=["filename"])

    # Drop groups where all rows have Avgjørelse == "utsatt"
    merged = merged.groupby("Hash").filter(
        lambda g: (g["Avgjørelse"] != "utsatt").any()
    )

    def squash_group(group: pd.DataFrame) -> pd.Series:
        # Pick the latest row by Dato
        latest = group.loc[group["Dato"].idxmax()]
        # Use Avgjørelse that is not "utsatt", fall back to the latest row's value
        non_utsatt = group[group["Avgjørelse"] != "utsatt"]["Avgjørelse"]
        if len(non_utsatt) > 1:
            raise ValueError(
                f"Multiple non-utsatt Avgjørelse values in group: {non_utsatt.tolist()}"
            )
        avgjørelse = (
            non_utsatt.iloc[0] if not non_utsatt.empty else latest["Avgjørelse"]
        )
        # near_sea: use the value that is not "utsatt"; expect at most one
        non_utsatt_sea = group[group["near_sea"] != "utsatt"]["near_sea"].dropna()
        if len(non_utsatt_sea) > 1:
            raise ValueError(
                f"Multiple non-utsatt near_sea values in group: {non_utsatt_sea.tolist()}"
            )
        near_sea = non_utsatt_sea.iloc[0] if not non_utsatt_sea.empty else None
        return pd.Series(
            {
                "Sakstittel": latest["Filnavn"],
                "Dato": latest["Dato"].strftime("%d/%m-%Y"),
                "Avgjørelse": avgjørelse,
                "Nær sjø": near_sea,
                "URLs": list(
                    group.sort_values("Dato", ascending=False)["URL"].dropna().unique()
                ),
            }
        )

    squashed = (
        merged.groupby("Hash", sort=False)[merged.columns.difference(["Hash"])]
        .apply(squash_group)
        .reset_index(drop=True)
    )
    # Expand URLs list into separate HYPERLINK formula columns
    max_urls = squashed["URLs"].map(len).max()
    for i in range(max_urls):
        squashed[f"URL_{i + 1}"] = squashed["URLs"].map(
            lambda urls, idx=i: (
                f'=HYPERLINK("{urls[idx]}","Fil {idx + 1}")'
                if idx < len(urls)
                else None
            )
        )
    squashed = squashed.drop(columns=["URLs"])
    squashed.to_excel(output_file.with_suffix(".xlsx"), index=False, engine="openpyxl")
    print(f"Saved {len(squashed)} rows to {output_file.with_suffix('.xlsx')}")


if __name__ == "__main__":
    data_folder = Path("D:/kommune-skrap/data/kristiansand")
    labels_file = data_folder / "labels_2025.csv"
    near_sea_file = data_folder / "near_sea_2025.csv"
    labels_squashed_file = data_folder / "filbeskrivelse_2025.xlsx"
    squash_labels_and_near_sea(labels_file, near_sea_file, labels_squashed_file)
