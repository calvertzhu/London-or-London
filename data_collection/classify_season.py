from data_collection.config import SEASON_MAPPING

def classify_season(date_string: str) -> str:
    """
    Convert a date string ('YYYY-MM') to a season label using SEASON_MAPPING.

    Args:
        date_string (str): e.g., '2020-06'

    Returns:
        str: 'winter', 'spring', 'summer', or 'fall'
    """
    try:
        month = int(date_string.split("-")[1])
        for season, months in SEASON_MAPPING.items():
            if month in months:
                return season
    except Exception as e:
        print(f"Failed to parse date '{date_string}': {e}")
        return "unknown"
    
    