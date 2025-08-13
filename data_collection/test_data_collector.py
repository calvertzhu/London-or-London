#!/usr/bin/env python3
"""
Test Data Collection Script

Collect test data for London city classification.
Collects winter images and outside-radius images for model testing.

Usage:
    python3 test_data_collector.py --mode quick --city london_on --winter 5 --outside 5
    python3 test_data_collector.py --mode standard --city london_uk --winter 20 --outside 20
    python3 test_data_collector.py --mode comprehensive --city both --winter 30 --outside 30
    python3 test_data_collector.py --help
"""

import argparse
import time
import random
import sys
import datetime
from pathlib import Path
sys.path.append('..')
from config import DATA_DIR, CITY_BOUNDING_BOXES, SEASON_MAPPING
from sample_coords import sample_nearby_coordinates
from get_pano_data import get_all_pano_data
from download_panorama import download_panorama
from process_image import crop_and_resize, save_processed_image
from classify_season import classify_season
from classify_blur import classify_sharpness
from log_metadata import log_metadata
from utils import generate_filename

# Test data directory
TEST_DATA_DIR = DATA_DIR.parent / "test_data"
TEST_METADATA_PATH = TEST_DATA_DIR / "test_metadata.csv"

# Extended bounding boxes for outside-radius collection
EXTENDED_BOUNDING_BOXES = {
    "london_uk": {
        "lat_min": 51.15, "lat_max": 51.82,
        "lon_min": -0.65, "lon_max": 0.35,
    },
    "london_on": {
        "lat_min": 42.80, "lat_max": 43.20,
        "lon_min": -81.55, "lon_max": -81.00,
    },
}

def sample_extended_coordinates(city, n=5, radius_deg=0.001):
    """
    Sample coordinates from extended bounding boxes (outside training radius).
    """
    box = EXTENDED_BOUNDING_BOXES[city]
    training_box = CITY_BOUNDING_BOXES[city]
    
    coords = []
    attempts = 0
    max_attempts = n * 10
    
    while len(coords) < n and attempts < max_attempts:
        lat = random.uniform(box["lat_min"], box["lat_max"])
        lon = random.uniform(box["lon_min"], box["lon_max"])
        
        # Check if outside training area
        outside_training = (
            lat < training_box["lat_min"] or 
            lat > training_box["lat_max"] or
            lon < training_box["lon_min"] or 
            lon > training_box["lon_max"]
        )
        
        if outside_training:
            # Add small perturbation
            lat_offset = random.uniform(-radius_deg, radius_deg)
            lon_offset = random.uniform(-radius_deg, radius_deg)
            lat += lat_offset
            lon += lon_offset
            coords.append((lat, lon))
        
        attempts += 1
    
    return coords

def collect_winter_images(city, target_total, run_id, mode="standard", verbose=True):
    """
    Collect winter images specifically.
    
    Args:
        city: City to collect from
        target_total: Number of winter images to collect
        run_id: Run identifier
        mode: "quick", "standard", or "comprehensive"
        verbose: Whether to print detailed progress
    """
    count = 0
    attempts = 0
    
    # Adjust parameters based on mode
    if mode == "quick":
        max_attempts = target_total * 10
        coords_per_batch = 2
    elif mode == "standard":
        max_attempts = target_total * 20
        coords_per_batch = 3
    else:  # comprehensive
        max_attempts = target_total * 30
        coords_per_batch = 3
    
    if verbose:
        print(f"Collecting {target_total} winter images for {city}...")
    
    while count < target_total and attempts < max_attempts:
        coords_list = sample_nearby_coordinates(city, n=coords_per_batch)
        
        for lat, lon in coords_list:
            if count >= target_total:
                break
                
            if verbose:
                print(f"  Sampling {city} at ({lat:.5f}, {lon:.5f})")
            
            pano_list = get_all_pano_data(lat, lon)
            
            if not pano_list:
                continue
                
            for pano in pano_list:
                if count >= target_total:
                    break
                    
                pano_id = pano["pano_id"]
                date = pano["date"]
                season = classify_season(date)
                
                # Only process winter images
                if season != "winter":
                    continue
                    
                if verbose:
                    print(f"  Found winter pano {pano_id} at {date}")
                
                raw_image = download_panorama(pano_id, verbose=verbose)
                if raw_image is None:
                    continue
                
                sharpness = classify_sharpness(raw_image)
                processed = crop_and_resize(raw_image, position="center")
                
                # Save to test data directory
                filename = generate_filename(
                    pano_id, pano["lat"], pano["lon"], date, season, sharpness
                )
                
                save_processed_image(
                    image=processed,
                    city=city,
                    season=season,
                    sharpness=sharpness,
                    pano_id=pano_id,
                    lat=pano["lat"],
                    lon=pano["lon"],
                    date=date,
                    save_dir=TEST_DATA_DIR
                )
                
                # Log metadata
                metadata = {
                    "filename": filename,
                    "city": city,
                    "lat": pano["lat"],
                    "lon": pano["lon"],
                    "date": date,
                    "season": season,
                    "sharpness": sharpness,
                    "pano_id": pano_id,
                    "source": f"test_{mode}",
                    "test_type": "winter",
                    "run_id": run_id
                }
                
                log_metadata(metadata, TEST_METADATA_PATH)
                
                count += 1
                if verbose:
                    print(f"  [{count}/{target_total}] Saved winter: {filename}")
                
            time.sleep(0.1)
        
        attempts += 1
        
        if verbose and attempts % 10 == 0:
            print(f"  Attempts: {attempts}, Found: {count}")
    
    if verbose:
        print(f"  Winter collection complete for {city}: {count}/{target_total}")
    return count

def collect_outside_radius_images(city, target_total, run_id, mode="standard", verbose=True):
    """
    Collect images from outside the training radius.
    
    Args:
        city: City to collect from
        target_total: Number of outside-radius images to collect
        run_id: Run identifier
        mode: "quick", "standard", or "comprehensive"
        verbose: Whether to print detailed progress
    """
    count = 0
    attempts = 0
    
    # Adjust parameters based on mode
    if mode == "quick":
        max_attempts = target_total * 5
        coords_per_batch = 2
    elif mode == "standard":
        max_attempts = target_total * 10
        coords_per_batch = 3
    else:  # comprehensive
        max_attempts = target_total * 15
        coords_per_batch = 3
    
    if verbose:
        print(f"Collecting {target_total} outside-radius images for {city}...")
    
    while count < target_total and attempts < max_attempts:
        coords_list = sample_extended_coordinates(city, n=coords_per_batch)
        
        for lat, lon in coords_list:
            if count >= target_total:
                break
                
            if verbose:
                print(f"  Sampling extended area {city} at ({lat:.5f}, {lon:.5f})")
            
            pano_list = get_all_pano_data(lat, lon)
            
            if not pano_list:
                continue
                
            for pano in pano_list:
                if count >= target_total:
                    break
                    
                pano_id = pano["pano_id"]
                date = pano["date"]
                season = classify_season(date)
                
                if verbose:
                    print(f"  Found outside-radius pano {pano_id} at {date}")
                
                raw_image = download_panorama(pano_id, verbose=verbose)
                if raw_image is None:
                    continue
                
                sharpness = classify_sharpness(raw_image)
                processed = crop_and_resize(raw_image, position="center")
                
                # Save to test data directory
                filename = generate_filename(
                    pano_id, pano["lat"], pano["lon"], date, season, sharpness
                )
                
                save_processed_image(
                    image=processed,
                    city=city,
                    season=season,
                    sharpness=sharpness,
                    pano_id=pano_id,
                    lat=pano["lat"],
                    lon=pano["lon"],
                    date=date,
                    save_dir=TEST_DATA_DIR
                )
                
                # Log metadata
                metadata = {
                    "filename": filename,
                    "city": city,
                    "lat": pano["lat"],
                    "lon": pano["lon"],
                    "date": date,
                    "season": season,
                    "sharpness": sharpness,
                    "pano_id": pano_id,
                    "source": f"test_{mode}",
                    "test_type": "outside_radius",
                    "run_id": run_id
                }
                
                log_metadata(metadata, TEST_METADATA_PATH)
                
                count += 1
                if verbose:
                    print(f"  [{count}/{target_total}] Saved outside-radius: {filename}")
                
            time.sleep(0.1)
        
        attempts += 1
        
        if verbose and attempts % 10 == 0:
            print(f"  Attempts: {attempts}, Found: {count}")
    
    if verbose:
        print(f"  Outside-radius collection complete for {city}: {count}/{target_total}")
    return count

def collect_test_data(city, winter_target, outside_target, mode="standard"):
    """
    Collect test data for specified city and targets.
    
    Args:
        city: City to collect from
        winter_target: Number of winter images to collect
        outside_target: Number of outside-radius images to collect
        mode: "quick", "standard", or "comprehensive"
    """
    # Generate run ID
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create test data directory
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    # Initialize test metadata file if it doesn't exist
    if not TEST_METADATA_PATH.exists():
        with open(TEST_METADATA_PATH, 'w') as f:
            f.write("filename,city,lat,lon,date,season,sharpness,pano_id,run_id,source,test_type\n")
    
    print(f"\nStarting {mode} test data collection")
    print(f"City: {city}")
    print(f"Winter target: {winter_target}")
    print(f"Outside-radius target: {outside_target}")
    print(f"Run ID: {run_id}")
    print(f"Output: {TEST_DATA_DIR}")
    
    start_time = time.time()
    
    # Collect winter images
    winter_count = collect_winter_images(city, winter_target, run_id, mode)
    
    # Collect outside radius images
    outside_count = collect_outside_radius_images(city, outside_target, run_id, mode)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{mode.capitalize()} test data collection complete!")
    print(f"City: {city}")
    print(f"Winter images: {winter_count}/{winter_target}")
    print(f"Outside-radius images: {outside_count}/{outside_target}")
    print(f"Total: {winter_count + outside_count}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Saved to: {TEST_DATA_DIR}")
    
    return winter_count + outside_count

def main():
    parser = argparse.ArgumentParser(
        description="Test data collection for London city classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  quick: Fast collection with lower attempt limits (good for testing)
  standard: Balanced collection with moderate attempt limits
  comprehensive: Thorough collection with high attempt limits (best quality)

Examples:
  python3 test_data_collector.py --mode quick --city london_on --winter 5 --outside 5
  python3 test_data_collector.py --mode standard --city london_uk --winter 20 --outside 20
  python3 test_data_collector.py --mode comprehensive --city both --winter 30 --outside 30
  python3 test_data_collector.py --mode quick --city london_on --winter 10
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "standard", "comprehensive"],
        default="standard",
        help="Collection mode (default: standard)"
    )
    
    parser.add_argument(
        "--city", 
        choices=["london_on", "london_uk", "both"],
        required=True,
        help="City to collect data for (london_on, london_uk, or both)"
    )
    
    parser.add_argument(
        "--winter", 
        type=int, 
        default=0,
        help="Number of winter images to collect (default: 0)"
    )
    
    parser.add_argument(
        "--outside", 
        type=int, 
        default=0,
        help="Number of outside-radius images to collect (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.winter == 0 and args.outside == 0:
        print("Error: Must specify at least one target (--winter or --outside)")
        sys.exit(1)
    
    if args.winter < 0 or args.outside < 0:
        print("Error: Targets must be non-negative")
        sys.exit(1)
    
    # Collect data
    if args.city == "both":
        print("Collecting for both cities...")
        total_on = collect_test_data("london_on", args.winter, args.outside, args.mode)
        total_uk = collect_test_data("london_uk", args.winter, args.outside, args.mode)
        print(f"\nGrand total: {total_on + total_uk} images")
    else:
        collect_test_data(args.city, args.winter, args.outside, args.mode)

if __name__ == "__main__":
    main() 