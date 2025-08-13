import pandas as pd
from pathlib import Path
import sys
sys.path.append('..')
from config import METADATA_CSV_PATH, CITY_BOUNDING_BOXES

def analyze_existing_data():
    """
    Analyze existing data to understand what test data is needed.
    """
    print("Analyzing existing data...")
    
    # Read existing metadata
    if METADATA_CSV_PATH.exists():
        df = pd.read_csv(METADATA_CSV_PATH)
        print(f"Found {len(df)} existing images")
        
        # Analyze by city and season
        print("\n=== Existing Data Analysis ===")
        for city in ["london_on", "london_uk"]:
            city_data = df[df['city'] == city]
            print(f"\n{city.upper()}:")
            print(f"  Total images: {len(city_data)}")
            
            # Season distribution
            season_counts = city_data['season'].value_counts()
            print("  Season distribution:")
            for season, count in season_counts.items():
                print(f"    {season}: {count}")
            
            # Check for winter images
            winter_count = len(city_data[city_data['season'] == 'winter'])
            print(f"  Winter images: {winter_count}")
            
            # Analyze geographic distribution
            if len(city_data) > 0:
                lat_min, lat_max = city_data['lat'].min(), city_data['lat'].max()
                lon_min, lon_max = city_data['lon'].min(), city_data['lon'].max()
                print(f"  Geographic bounds:")
                print(f"    Lat: {lat_min:.5f} to {lat_max:.5f}")
                print(f"    Lon: {lon_min:.5f} to {lon_max:.5f}")
                
                # Check training bounds
                training_box = CITY_BOUNDING_BOXES[city]
                print(f"  Training bounds:")
                print(f"    Lat: {training_box['lat_min']:.5f} to {training_box['lat_max']:.5f}")
                print(f"    Lon: {training_box['lon_min']:.5f} to {training_box['lon_max']:.5f}")
                
                # Find images outside training bounds
                outside_bounds = city_data[
                    (city_data['lat'] < training_box['lat_min']) |
                    (city_data['lat'] > training_box['lat_max']) |
                    (city_data['lon'] < training_box['lon_min']) |
                    (city_data['lon'] > training_box['lon_max'])
                ]
                print(f"  Images outside training bounds: {len(outside_bounds)}")
    else:
        print("No existing metadata found")
    
    print("\n=== Test Data Recommendations ===")
    print("1. Winter images: Need more winter images for both cities")
    print("2. Outside radius: Need images from areas outside current training bounds")
    print("3. Geographic diversity: Ensure test data covers different areas")

def check_test_data_directory():
    """
    Check if test data directory exists and what's in it.
    """
    test_data_dir = Path("test_data")
    if test_data_dir.exists():
        print(f"\nTest data directory exists: {test_data_dir}")
        
        # Check for test metadata
        test_metadata = test_data_dir / "test_metadata.csv"
        if test_metadata.exists():
            df = pd.read_csv(test_metadata)
            print(f"Found {len(df)} test images")
            
            # Analyze test data
            print("\n=== Test Data Analysis ===")
            for city in ["london_on", "london_uk"]:
                city_data = df[df['city'] == city]
                if len(city_data) > 0:
                    print(f"\n{city.upper()} test data:")
                    print(f"  Total: {len(city_data)}")
                    
                    # By test type
                    test_type_counts = city_data['test_type'].value_counts()
                    print("  By test type:")
                    for test_type, count in test_type_counts.items():
                        print(f"    {test_type}: {count}")
                    
                    # By season
                    season_counts = city_data['season'].value_counts()
                    print("  By season:")
                    for season, count in season_counts.items():
                        print(f"    {season}: {count}")
        else:
            print("No test metadata found")
    else:
        print(f"\nTest data directory does not exist: {test_data_dir}")

if __name__ == "__main__":
    analyze_existing_data()
    check_test_data_directory() 