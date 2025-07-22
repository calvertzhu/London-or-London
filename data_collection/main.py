import argparse
from datetime import datetime
from data_collection.collect import collect
from data_collection.augment_dataset import augment_dataset
from data_collection.explore_metadata import analyze_metadata
from data_collection.balance_dataset import balance_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True, help="City to collect data for (e.g. london_uk or london_on)")
    parser.add_argument('--target', type=int, default=100, help="Target number of images to collect")
    args = parser.parse_args()

    # Auto-generate a run ID
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"\nStarting new run: {run_id}")

    # Collect
    print(f"\nCollecting data for: {args.city} (target={args.target})")
    collect(city=args.city, target_total=args.target, run_id=run_id)

    # Augment
    print(f"\nAugmenting collected images (run_id={run_id})")
    augment_dataset(run_id=run_id)

    # Explore
    print("\nAnalyzing metadata")
    analyze_metadata()

    # Balance
    print(f"\nBalancing dataset (run_id={run_id})")
    balance_dataset(run_id=run_id)

    print("\nFull pipeline completed.")

if __name__ == "__main__":
    main()
