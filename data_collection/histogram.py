import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def clean_sharpness(value):
    if pd.isna(value):
        return value
    if "blurry" in value:
        return "blurry"
    return "sharp"

def visualize_regenerated_metadata():
    metadata_path = Path("metadata/regenerated_metadata.csv")
    output_dir = metadata_path.parent

    if not metadata_path.exists():
        print("Error: metadata/regenerated_metadata.csv not found.")
        return

    df = pd.read_csv(metadata_path)

    # Normalize sharpness column
    df["sharpness"] = df["sharpness"].apply(clean_sharpness)

    # Filter only valid seasons
    valid_seasons = {"spring", "summer", "fall", "winter"}
    df = df[df["season"].isin(valid_seasons)]

    for city in sorted(df["city"].unique()):
        city_df = df[df["city"] == city]

        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=city_df,
            x="season",
            hue="sharpness",
            palette="Set2"
        )
        plt.title(f"Class Distribution by Season and Sharpness — {city}")
        plt.xlabel("Season")
        plt.ylabel("Number of Images")
        plt.tight_layout()

        plot_path = output_dir / f"{city}_class_distribution.png"
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close()

if __name__ == "__main__":
    visualize_regenerated_metadata()
