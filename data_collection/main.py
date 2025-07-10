from data_collection.collect import collect
from data_collection.explore_metadata import analyze_metadata
from data_collection.balance_dataset import balance_dataset

if __name__ == "__main__":
    print("Starting data collection")
    collect()

    print("\nAnalyzing metadata")
    analyze_metadata()

    print("\nBalancing dataset")
    balance_dataset()

    print("\nAll done.")
