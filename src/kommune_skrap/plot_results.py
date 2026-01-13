from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

data_folder = Path("D:/kommune-skrap/data/kristiansand")
labels_file = data_folder / "labels.csv"
near_sea_file = data_folder / "near_sea.csv"

# Load data
df = pd.read_csv(labels_file, parse_dates=["date"])
df_ns = pd.read_csv(near_sea_file)

df_ns["filename"] = [
    f[:5].replace("data\\", "D:\\kommune-skrap\\data\\") + f[5:]
    for f in df_ns["filename"]
]

# Get whether the file is in Søgne or Sogndalen or not
df["filepath"] = [Path(f) for f in df["filename"]]
df["in_søgne"] = [p.parts[3] == "sogne" for p in df.filepath]
df["in_sogndalen"] = [p.parts[3] == "sogndalen" for p in df.filepath]

# Filter out søgne and sogndalen
df = df[~df["in_søgne"] & ~df["in_sogndalen"]]

# Merge dataframes
df = pd.merge(df, df_ns, on="filename", how="left")

# Set row number, year, month, and day as indices
df["row_number"] = df.index
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
# Drop 2017 data
df = df[(df["year"] > 2007) & (df["year"] < 2026)]
df.set_index(["row_number", "year", "month", "day"], inplace=True)
df.drop(columns=["date"], inplace=True)

# Drop parents from filename
df["filename"] = df["filepath"].apply(lambda x: x.parts[-1])

# Filter out "utsatt" labels
df = df[df["label"].isin(["innvilget", "avslått"])]

# Count total occurrences per month
yearly_counts = df.groupby(["year"]).size()

# Count occurrences of "innvilget" label per month
innvilget_counts = df[df["label"] == "innvilget"].groupby(["year"]).size()

# Percentage of accepted applications per year
innvilget_percentage = innvilget_counts / yearly_counts * 100

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

# Create a DataFrame for plotting
plot_df = pd.DataFrame(
    {"# søknader": yearly_counts, "% innvilget": innvilget_percentage}
).fillna(0)

# Plot barplot - total applications and percentage of accepted applications per year
green = colormaps.get("Paired").colors[3]
fig, ax = plt.subplots()
plot_df["% innvilget"].plot(kind="bar", color=green, ax=ax, width=0.8)
plt.xlabel("År")
plt.ylabel("%")
plt.title("Prosent innvilgede dispensasjoner per år i Kristiansand Kommune")
plt.ylim(0, 100)
fig.savefig("D:/kommune-skrap/data/Andel innvilget.png", dpi=500)

# Plot barplot - total applications and percentage of accepted applications per year
fig, ax = plt.subplots()
# ax2 = ax.twinx()
plot_df.plot(kind="bar", ax=ax)
# plot_df.plot(kind="bar", color="green", ax=ax2)
plt.xlabel("År")
plt.ylabel("# og %")
plt.title("Prosent innvilgede søknader per år i Kristiansand Kommune")
plt.ylim(0, 100)


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
