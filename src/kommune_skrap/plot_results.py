from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

data_folder = Path("D:/kommune-skrap/data")
labels_file = data_folder / "training_data/labels.csv"
near_sea_file = data_folder / "training_data/near_sea.csv"

# Load data
df = pd.read_csv(labels_file, parse_dates=["date"])
df_ns = pd.read_csv(near_sea_file)

# Merge dataframes
df = pd.merge(df, df_ns, on="filename")

# Set row number, year, month, and day as indices
df["row_number"] = df.index
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
# Drop 2017 data
df = df[df["year"] > 2007]
df.set_index(["row_number", "year", "month", "day"], inplace=True)
df.drop(columns=["date"], inplace=True)

# Drop parents from filename
df["filename"] = df["filename"].apply(lambda x: Path(x).parts[-1])

# Filter out "utsatt" labels
df = df[df["label"].isin(["innvilget", "avslått"])]

# Count total occurrences per month
yearly_counts = df.groupby(["year"]).size()

# Count occurrences of "innvilget" label per month
innvilget_counts = df[df["label"] == "innvilget"].groupby(["year"]).size()

# Count total occurrences per month
df_ns = df[df["near_sea"] == "ja"]
yearly_counts_ns = df_ns.groupby(["year"]).size()
innvilget_counts_ns = df_ns[df_ns["label"] == "innvilget"].groupby(["year"]).size()

# Create a DataFrame for plotting
plot_df = pd.DataFrame(
    {"Disp. søknader": yearly_counts, "Innvilget": innvilget_counts}
).fillna(0)

# Plot barplot - total applications and accepted applications per year
ax = plot_df.plot(kind="bar", width=0.8)
for container in ax.containers:
    plt.setp(container, width=0.4)
plt.xlabel("År")
plt.ylabel("Antall")
plt.title("Antall dispensasjonssøknader og innvilgede søknader per år")

# Plot barplot - applications that are within 100 meters of the sea
plot_df = pd.DataFrame(
    {"Disp. søknader": yearly_counts_ns, "Innvilget": innvilget_counts_ns}
).fillna(0)

ax = plot_df.plot(kind="bar", width=0.8)
for container in ax.containers:
    plt.setp(container, width=0.4)
plt.xlabel("År")
plt.ylabel("Antall")
plt.title("Antall søkander innenfor 100 meter av sjø og vassdrag per år")

print(len(df[df["label"].isin(("innvilget", "avslått"))]))
print(len(df[df["label"] == "innvilget"]))

plt.show()
