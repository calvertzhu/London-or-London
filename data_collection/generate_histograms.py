#!/usr/bin/env python3
"""
Generate histograms for training data distribution
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def analyze_data_distribution():
    """Analyze the distribution of images in the data directory."""
    
    data_dir = Path("../data")
    metadata_dir = Path("../metadata")
    metadata_dir.mkdir(exist_ok=True)
    
    # Collect data from directory structure
    data_records = []
    
    for city in ["london_on", "london_uk"]:
        city_dir = data_dir / city
        if not city_dir.exists():
            continue
            
        for season_dir in city_dir.iterdir():
            if season_dir.is_dir():
                season = season_dir.name
                for sharpness_dir in season_dir.iterdir():
                    if sharpness_dir.is_dir():
                        sharpness = sharpness_dir.name
                        image_count = len(list(sharpness_dir.glob("*.jpg")))
                        
                        if image_count > 0:
                            data_records.append({
                                'city': city,
                                'season': season,
                                'sharpness': sharpness,
                                'count': image_count
                            })
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    if df.empty:
        print("No data found in data directory")
        return
    
    print(f"Total images: {df['count'].sum()}")
    print(f"Records: {len(df)}")
    
    # Save metadata
    metadata_path = metadata_dir / "training_data_distribution.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Saved metadata to: {metadata_path}")
    
    # Generate histograms
    generate_histograms(df, metadata_dir)
    
def generate_histograms(df, output_dir):
    """Generate various histogram visualizations with proper scaling."""
    
    # 1. Overall distribution by city
    plt.figure(figsize=(10, 6))
    city_counts = df.groupby('city')['count'].sum()
    bars = plt.bar(city_counts.index, city_counts.values, color=['#ff7f0e', '#1f77b4'])
    
    # Add value labels on bars
    for bar, value in zip(bars, city_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Total Images by City')
    plt.ylabel('Number of Images')
    plt.xlabel('City')
    plt.xticks([0, 1], ['London Ontario', 'London UK'])
    plt.ylim(0, max(city_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / 'city_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'city_distribution.png'}")
    
    # 2. Distribution by season and city
    plt.figure(figsize=(12, 8))
    season_city_pivot = df.groupby(['season', 'city'])['count'].sum().unstack(fill_value=0)
    
    # Reorder seasons properly
    season_order = ['spring', 'summer', 'fall', 'winter']
    season_city_pivot = season_city_pivot.reindex(season_order)
    
    season_city_pivot.plot(kind='bar', stacked=True, color=['#ff7f0e', '#1f77b4'])
    plt.title('Image Distribution by Season and City')
    plt.ylabel('Number of Images')
    plt.xlabel('Season')
    plt.legend(['London Ontario', 'London UK'])
    plt.xticks(rotation=45)
    plt.ylim(0, season_city_pivot.sum(axis=1).max() * 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / 'season_city_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'season_city_distribution.png'}")
    
    # 3. Distribution by sharpness and city
    plt.figure(figsize=(10, 6))
    sharpness_city_pivot = df.groupby(['sharpness', 'city'])['count'].sum().unstack(fill_value=0)
    
    # Reorder sharpness to show sharp first
    sharpness_order = ['sharp', 'blurry']
    sharpness_city_pivot = sharpness_city_pivot.reindex(sharpness_order)
    
    # Use original colors
    sharpness_city_pivot.plot(kind='bar', color=['#ff7f0e', '#1f77b4'])
    plt.title('Image Distribution by Sharpness and City')
    plt.ylabel('Number of Images')
    plt.xlabel('Sharpness')
    plt.legend(['London Ontario', 'London UK'])
    plt.xticks(rotation=45)
    plt.ylim(0, sharpness_city_pivot.max().max() * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / 'sharpness_city_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'sharpness_city_distribution.png'}")
    
    # 4. Detailed breakdown heatmap
    plt.figure(figsize=(12, 8))
    heatmap_data = df.groupby(['season', 'sharpness', 'city'])['count'].sum().unstack(fill_value=0)
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Number of Images'})
    plt.title('Detailed Image Distribution Heatmap')
    plt.xlabel('City')
    plt.ylabel('Season - Sharpness')
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'detailed_heatmap.png'}")
    
    # 5. Individual city breakdowns using combined CSV
    generate_individual_city_breakdowns(output_dir)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nTotal Images: {df['count'].sum():,}")
    print(f"London Ontario: {df[df['city'] == 'london_on']['count'].sum():,}")
    print(f"London UK: {df[df['city'] == 'london_uk']['count'].sum():,}")
    
    print(f"\nBy Season:")
    season_totals = df.groupby('season')['count'].sum()
    for season, count in season_totals.items():
        percentage = (count / df['count'].sum()) * 100
        print(f"  {season}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nBy Sharpness:")
    sharpness_totals = df.groupby('sharpness')['count'].sum()
    for sharpness, count in sharpness_totals.items():
        percentage = (count / df['count'].sum()) * 100
        print(f"  {sharpness}: {count:,} ({percentage:.1f}%)")

def generate_individual_city_breakdowns(output_dir):
    """Generate individual city breakdown charts using combined CSV data."""
    
    # Load combined CSV
    combined_csv_path = Path("../metadata/combined.csv")
    if not combined_csv_path.exists():
        print("Warning: combined.csv not found, skipping individual city breakdowns")
        combined_df = pd.read_csv(combined_csv_path)
        print(f"Loaded combined CSV with {len(combined_df)} records")
        
        # Generate breakdowns for each city
        for city in combined_df['city'].unique():
            city_df = combined_df[combined_df['city'] == city]
        
            # Order seasons properly
            season_order = ['spring', 'summer', 'fall', 'winter']
            city_df['season'] = pd.Categorical(city_df['season'], categories=season_order, ordered=True)
            
            # Order sharpness to ensure sharp is first (left side)
            sharpness_order = ['sharp', 'blurry']
            city_df['sharpness'] = pd.Categorical(city_df['sharpness'], categories=sharpness_order, ordered=True)
            
            plt.figure(figsize=(10, 6))
            
            # Use original seaborn palette
            ax = sns.countplot(data=city_df, x='season', hue='sharpness', palette='Set2', hue_order=sharpness_order)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontweight='bold')
            
            city_name = 'London Ontario' if city == 'london_on' else 'London UK'
            plt.title(f'Image Distribution for {city_name}')
            plt.xlabel('Season')
            plt.ylabel('Number of Images')
            plt.legend(title='Sharpness')
            
            # Set proper Y-axis limits with padding
            max_height = max([rect.get_height() for rect in ax.patches])
            plt.ylim(0, max_height * 1.15)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{city}_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / f'{city}_distribution.png'}")
   
if __name__ == "__main__":
    analyze_data_distribution() 