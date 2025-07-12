import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_metadata():
    metadata_path = Path("metadata/combined.csv")

    if not metadata_path.exists():
        print("Error: metadata/combined.csv not found.")
        return

    # Load metadata
    df = pd.read_csv(metadata_path)

    print("\nTotal images:", len(df))

    # Basic stats4
    print("\nBy city:")
    print(df["city"].value_counts())

    print("\nBy season:")
    print(df["season"].value_counts())

    print("\nBy sharpness:")
    print(df["sharpness"].value_counts())

    print("\nBy (city, season, sharpness):")
    breakdown = df.groupby(["city", "season", "sharpness"]).size()
    print(breakdown)

    # Save breakdown CSV
    class_dist_path = Path("metadata/class_distribution.csv")
    breakdown.reset_index(name="count").to_csv(class_dist_path, index=False)
    print(f"\nSaved class breakdown to {class_dist_path}")

    # Plot season vs sharpness
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="season", hue="sharpness", palette="Set2")
    plt.title("Season vs Sharpness")
    plt.savefig("metadata/plot_season_sharpness.png")
    print("Saved plot: metadata/plot_season_sharpness.png")
    plt.close()

    # Plot city+season+sharpness combo
    pivot = df.groupby(["city", "season", "sharpness"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True)
    plt.title("Distribution by (City, Season, Sharpness)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("metadata/plot_city_season_sharpness.png")
    print("Saved plot: metadata/plot_city_season_sharpness.png")
    plt.close()

    # Separate histograms for each city
    cities = df["city"].unique()
    for city in cities:
        city_df = df[df["city"] == city]

        plt.figure(figsize=(10, 6))
        sns.countplot(data=city_df, x="season", hue="sharpness", palette="Set2")
        plt.title(f"Season vs Sharpness for {city}")
        plt.xlabel("Season")
        plt.ylabel("Count")
        plt.tight_layout()
        output_path = f"metadata/histogram_{city}.png"
        plt.savefig(output_path)
        print(f"Saved plot: {output_path}")
        plt.close()

if __name__ == "__main__":
    analyze_metadata()
