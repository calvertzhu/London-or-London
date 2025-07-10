import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load metadata
df = pd.read_csv("metadata/combined.csv")

# Summary tables
print("\n📊 Total images:", len(df))
print("\nBy city:")
print(df["city"].value_counts())
print("\nBy season:")
print(df["season"].value_counts())
print("\nBy sharpness:")
print(df["sharpness"].value_counts())
print("\nBy (city, season, sharpness):")
print(df.groupby(["city", "season", "sharpness"]).size())

# Save CSV of breakdowns (optional)
df.groupby(["city", "season", "sharpness"]).size().reset_index(name="count").to_csv("metadata/class_distribution.csv", index=False)

# Plot bar chart by combination
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="season", hue="sharpness", palette="Set2")
plt.title("Season vs Sharpness")
plt.savefig("metadata/plot_season_sharpness.png")
plt.show()

# Optional: Heatmap of all combinations
pivot = df.groupby(["city", "season", "sharpness"]).size().unstack(fill_value=0)
pivot.plot(kind="bar", stacked=True)
plt.title("Distribution by (City, Season, Sharpness)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("metadata/plot_city_season_sharpness.png")
plt.show()
