#!/usr/bin/env python3
"""
Pixabay Image Downloader

This script allows searching and downloading images from Pixabay using their API.
All parameters are optional, with sensible defaults provided.
"""

import os
import sys
import logging
import argparse
import time
import traceback
import json
from datetime import datetime
import requests
from urllib.parse import urlencode
from collections import Counter
import glob
import dotenv
from tqdm import tqdm
import threading

# Available options for API parameters
LANGUAGES = ["cs", "da", "de", "en", "es", "fr", "id", "it", "hu", "nl", "no", "pl", 
            "pt", "ro", "sk", "fi", "sv", "tr", "vi", "th", "bg", "ru", "el", "ja", "ko", "zh"]
CATEGORIES = ["backgrounds", "fashion", "nature", "science", "education", "feelings", 
             "health", "people", "religion", "places", "animals", "industry", 
             "computer", "food", "sports", "transportation", "travel", "buildings", 
             "business", "music"]
ORDER_OPTIONS = ["popular", "latest"]

# API Configuration
API_URL = "https://pixabay.com/api/"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to load environment variables from .env file
dotenv.load_dotenv()

# Default settings for the downloader
DEFAULT_SETTINGS = {
    'PER_PAGE': 200,        # Maximum per_page allowed by Pixabay API (3-200)
    'BATCH_SIZE': 200,      # Match batch size to per_page for optimal performance
    'MAX_RETRIES': 3,       # Default number of retry attempts for failed downloads
    'QUALITY': 'highest',   # Default image quality (highest, high, medium, low)
    'IMAGE_TYPE': 'all',    # Default image type (all, photo, illustration, vector)
    'ORIENTATION': 'all',   # Default orientation (all, horizontal, vertical)
    'ORDER': 'popular',     # Default order (popular, latest)
    'LANG': 'en',          # Default language for search
    'SAFESEARCH': True,     # If True, only images suitable for all ages are returned
    
    # Advanced options
    'SKIP_ERRORS': False,   # If True, continue downloading even when API errors occur
    'SAVE_PROGRESS': False, # If True, save download progress periodically for resume capability
}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Download images from Pixabay based on search criteria.",
        epilog="Example: python pixabay.py -q \"beautiful landscape\" -n 100 --type photo --orientation horizontal"
    )
    
    # Basic options
    parser.add_argument('-q', '--query', type=str,
                        help='Search term for images (required unless --resume is used)')
    parser.add_argument('-o', '--output-dir', type=str, 
                        help='Custom output directory (default: Output/YYYY-MM-DD/search_term)')
    parser.add_argument('-n', '--max-images', type=int, default=0, 
                        help='Maximum number of images to download (default: 0, which means all available images)')
    
    # Image filters
    parser.add_argument('-t', '--type', dest='type', choices=['all', 'photo', 'illustration', 'vector'], 
                        default=DEFAULT_SETTINGS['IMAGE_TYPE'],
                        help=f'Type of images to download (default: {DEFAULT_SETTINGS["IMAGE_TYPE"]})')
    parser.add_argument('--orientation', type=str, choices=['all', 'horizontal', 'vertical'], 
                        default=DEFAULT_SETTINGS['ORIENTATION'],
                        help=f'Image orientation (default: {DEFAULT_SETTINGS["ORIENTATION"]})')
    parser.add_argument('--category', type=str, choices=CATEGORIES,
                        help='Category of images (default: none)')
    parser.add_argument('--min-width', type=int,
                        help='Minimum image width in pixels')
    parser.add_argument('--min-height', type=int,
                        help='Minimum image height in pixels')
    parser.add_argument('--colors', type=str,
                        help='Filter by comma-separated colors (e.g. "red,blue")')
    parser.add_argument('--order', type=str, choices=ORDER_OPTIONS, default=DEFAULT_SETTINGS['ORDER'],
                        help=f'How to order the results (default: {DEFAULT_SETTINGS["ORDER"]})')
    parser.add_argument('--lang', type=str, choices=LANGUAGES, default=DEFAULT_SETTINGS['LANG'],
                        help=f'Language code for the search (default: {DEFAULT_SETTINGS["LANG"]})')
    parser.add_argument('--per-page', type=int, default=DEFAULT_SETTINGS['PER_PAGE'],
                        help=f'Number of results per page (3-200, default: {DEFAULT_SETTINGS["PER_PAGE"]}). Pixabay API maximum is 200.')
    parser.add_argument('--safesearch', action='store_true', default=DEFAULT_SETTINGS['SAFESEARCH'],
                        help='Only return images suitable for all ages')
    parser.add_argument('--editors-choice', action='store_true',
                        help='Filter for images selected by Pixabay editors')
    
    # Download options
    parser.add_argument('--quality', type=str, choices=['highest', 'high', 'medium', 'low'], 
                        default=DEFAULT_SETTINGS['QUALITY'],
                        help=f'Image quality to download (default: {DEFAULT_SETTINGS["QUALITY"]})')
    
    # Retry settings
    parser.add_argument('--retries', type=int, default=DEFAULT_SETTINGS['MAX_RETRIES'],
                        help=f'Number of times to retry failed downloads (default: {DEFAULT_SETTINGS["MAX_RETRIES"]})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_SETTINGS['BATCH_SIZE'],
                        help=f'Number of images to download in parallel (default: {DEFAULT_SETTINGS["BATCH_SIZE"]}). Higher values = faster downloads but use more memory. Set to 0 to auto-calculate based on per_page.')
    parser.add_argument('--skip-errors', action='store_true',
                        help='Continue downloading even if API errors occur')
    
    # Progress and resumption
    parser.add_argument('--save-progress', action='store_true',
                        help='Save download progress to enable resuming')
    parser.add_argument('--resume', action='store_true',
                        help='Resume previous download (requires --output-dir)')
    
    # Logging options
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--error-log', type=str,
                        help='Path to save error log JSON file. Use "auto" to save in the output directory.')
    
    # API key
    parser.add_argument('-k', '--key', type=str,
                        help='Pixabay API key (default: read from .env file)')
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        args.query = input("Enter search term: ")
        if not args.query.strip():
            parser.print_help()
            sys.exit(1)
        return args
    
    # Validate arguments
    if not args.resume and not args.query:
        args.query = input("Enter search term: ")
        if not args.query.strip():
            logger.error("No search term provided")
            parser.print_help()
            sys.exit(1)
    
    if args.resume and not args.output_dir:
        parser.error("--resume requires --output-dir to be specified")
    
    return args

def get_best_image_url(image, quality='highest'):
    """Get the best available image URL based on desired quality"""
    if quality == 'highest' and image.get("imageURL"):
        return image["imageURL"]
    elif (quality == 'highest' or quality == 'high') and image.get("largeImageURL"):
        return image["largeImageURL"]
    elif (quality in ['highest', 'high', 'medium']) and image.get("webformatURL"):
        return image["webformatURL"]
    elif image.get("previewURL"):
        return image["previewURL"]
    return None

def save_error_log(error_log_path, errors):
    """Save error details to a file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        
        with open(error_log_path, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.info(f"Error log saved to {error_log_path}")
    except Exception as e:
        logger.error(f"Failed to save error log: {e}")

def save_download_progress(output_dir, downloaded_ids):
    """Save download progress to a file"""
    try:
        progress_file = os.path.join(output_dir, 'download_progress.json')
        with open(progress_file, 'w') as f:
            json.dump({
                'downloaded_ids': downloaded_ids,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Download progress saved to {progress_file}")
    except Exception as e:
        logger.error(f"Failed to save download progress: {e}")

def load_download_progress(output_dir):
    """Load download progress from a file"""
    try:
        progress_file = os.path.join(output_dir, 'download_progress.json')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                logger.info(f"Loaded progress from {progress_file} ({len(progress.get('downloaded_ids', []))} images)")
                return progress.get('downloaded_ids', [])
        return []
    except Exception as e:
        logger.error(f"Failed to load download progress: {e}")
        return []

def download_all_images(client, search_params, output_dir, max_images=0, quality='highest', 
                        verbose=False, debug=False, retries=3, batch_size=0, 
                        error_log_path=None, skip_errors=False, save_progress=False, resume=False):
    """Download images in batches, respecting rate limits"""
    # Initialize counters and flags
    downloaded_count = 0
    page = 1
    total_hits = 0
    rate_limit_remaining = 100
    rate_limit_reset = 60
    failed_downloads = []
    error_types = Counter()
    downloaded_ids = []
    
    # Store original search parameters to use as base for variations
    original_params = search_params.copy()
    
    # Define search variations to bypass the 500 image limit
    search_variations = [
        {"name": "Default", "params": {}},  # Original search 
        {"name": "Latest", "params": {"order": "latest"}},  # Change sort order
        {"name": "SafeSearch", "params": {"safesearch": "true"}},  # Enable safe search
        {"name": "Editor's Choice", "params": {"editors_choice": "true"}},  # Editor's choice only
    ]
    
    # Add color variations
    colors = ["grayscale", "transparent", "red", "orange", "yellow", "green", 
              "turquoise", "blue", "lilac", "pink", "white", "gray", "black", "brown"]
    for color in colors:
        search_variations.append({"name": f"Color: {color}", "params": {"colors": color}})
    
    # Add combination variations
    search_variations.append({"name": "Latest + SafeSearch", "params": {"order": "latest", "safesearch": "true"}})
    search_variations.append({"name": "Latest + Editor's Choice", "params": {"order": "latest", "editors_choice": "true"}})
    search_variations.append({"name": "SafeSearch + Editor's Choice", "params": {"safesearch": "true", "editors_choice": "true"}})
    search_variations.append({"name": "Latest + SafeSearch + Editor's Choice", "params": {"order": "latest", "safesearch": "true", "editors_choice": "true"}})
    
    # Add color combinations with other filters
    for color in colors[:5]:  # Limit to first 5 colors to avoid too many variations
        search_variations.append({"name": f"Latest + Color: {color}", "params": {"order": "latest", "colors": color}})
        search_variations.append({"name": f"Editor's Choice + Color: {color}", "params": {"editors_choice": "true", "colors": color}})
    
    # Add keyword variations with appended terms
    original_query = original_params.get('q', '').strip()
    if original_query:
        keyword_variations = ["quality", "visual", "hq", "professional", "high resolution", "best", "clear", "detailed"]
        for keyword in keyword_variations:
            search_variations.append({"name": f"Keyword: {keyword}", "params": {"q": f"{original_query} {keyword}"}})
    
    # Load progress if resuming
    if resume:
        downloaded_ids = load_download_progress(output_dir)
        if downloaded_ids:
            logger.info(f"Resuming download - skipping {len(downloaded_ids)} already downloaded images")
            downloaded_count = len(downloaded_ids)
    
    # First, query to get total number of images across all variations
    try:
        logger.info(f"Searching for: \"{original_params.get('q', '')}\"")
        logger.info("Querying Pixabay API to get total available images...")
        initial_params = original_params.copy()
        initial_params['page'] = 1
        initial_params['per_page'] = 3
        
        if debug:
            logger.debug(f"Initial query parameters: {initial_params}")
            
        response = requests.get(API_URL, params=initial_params)
        response.raise_for_status()
        
        results = response.json()
        total_hits = results.get("total", 0)  # Total matching images
        api_accessible = results.get("totalHits", 0)  # Limited to 500 per query
        
        if total_hits == 0:
            logger.error("No images found matching your criteria")
            return 0
            
        logger.info(f"Found {total_hits} total images (API allows max 500 per variation)")
        
        # Determine how many images to download
        if max_images <= 0:
            # Try to download everything
            target_total = total_hits  
            logger.info(f"Will attempt to download all {target_total} images using search variations")
        else:
            target_total = min(max_images, total_hits)
            logger.info(f"Will download up to {target_total} images using search variations")
        
        # Set optimal per_page for the API (max allowed is 200)
        per_page = min(200, original_params.get('per_page', DEFAULT_SETTINGS['PER_PAGE']))
        
        # Create a variation download plan
        variations_needed = (target_total + 499) // 500  # Ceiling division for how many variations we need
        logger.info(f"Planning to download {target_total} images using at least {variations_needed} search variations")
        logger.info(f"Total of {len(search_variations)} variations available if needed")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to query API: {e}")
        if hasattr(e, 'response') and e.response:
            if hasattr(e.response, 'text'):
                logger.error(f"API Response: {e.response.text}")
            logger.error(f"Request URL: {e.response.url}")
        return 0
    
    # Create progress bar for overall progress
    pbar = tqdm(total=target_total, 
                desc=f"Downloading images", 
                unit="img",
                position=0,
                leave=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Update progress bar with any resumed progress
    if resume and downloaded_count > 0:
        pbar.update(downloaded_count)
    
    try:
        # Try each variation until we've downloaded enough images
        for variation_idx, variation in enumerate(search_variations):
            # Stop if we've downloaded enough images
            if downloaded_count >= target_total:
                break
                
            # Create a new set of parameters for this variation
            current_params = original_params.copy()
            current_params.update(variation["params"])
            
            logger.info(f"\nTrying search variation {variation_idx+1}/{len(search_variations)}: {variation['name']}")
            
            # Reset page for new variation
            page = 1
            
            # Calculate how many more images we need
            remaining_target = target_total - downloaded_count
            
            # Process each page for this variation
            while downloaded_count < target_total:
                # Calculate current page size
                current_page_size = min(per_page, remaining_target)
                    
                # Update parameters for this page
                current_params['page'] = page
                current_params['per_page'] = current_page_size
                
                # Check rate limit
                if rate_limit_remaining < 5:
                    wait_time = rate_limit_reset + 1
                    logger.info(f"Rate limit almost reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                
                # Make the API request
                if verbose:
                    logger.info(f"Downloading page {page} ({current_page_size} images)")
                
                try:
                    response = requests.get(API_URL, params=current_params)
                    response.raise_for_status()
                    
                    # Update rate limit info
                    rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
                    rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 60))
                    
                    results = response.json()
                    hits = results.get("hits", [])
                    
                    if not hits:
                        logger.info(f"No more images found for variation '{variation['name']}'")
                        break
                    
                    # Prepare batch download
                    download_batch = []
                    for image in hits:
                        image_id = image.get("id")
                        
                        # Skip already downloaded images
                        if str(image_id) in downloaded_ids:
                            if debug:
                                logger.debug(f"Skipping already downloaded image {image_id}")
                            continue
                        
                        image_url = get_best_image_url(image, quality)
                        if not image_url:
                            logger.warning(f"No valid URL found for image {image_id}")
                            continue
                        
                        download_batch.append({
                            'url': image_url,
                            'filename': f"pixabay_{image_id}.jpg",
                            'id': image_id,
                            'info': {
                                'width': image.get('imageWidth'),
                                'height': image.get('imageHeight'),
                                'size': image.get('imageSize'),
                                'views': image.get('views'),
                                'downloads': image.get('downloads'),
                                'type': image.get('type')
                            }
                        })
                    
                    # Download the batch
                    if download_batch:
                        batch_size = len(download_batch)
                        
                        successful, failures = download_batch_with_retry(
                            download_batch, 
                            output_dir, 
                            retries, 
                            debug
                        )
                        
                        # Update tracking
                        new_downloads = len(successful)
                        downloaded_count += new_downloads
                        downloaded_ids.extend([str(item.get('id')) for item in download_batch])
                        failed_downloads.extend(failures)
                        
                        # Update main progress bar
                        pbar.update(new_downloads)
                        
                        # Update progress description
                        pbar.set_description(f"Downloaded {downloaded_count}/{target_total} images")
                        
                        # Track error types
                        for failure in failures:
                            error_type = failure.get('error_type', 'unknown')
                            error_types[error_type] += 1
                        
                        # Save progress periodically if requested
                        if save_progress:
                            save_download_progress(output_dir, downloaded_ids)
                        
                        # Log how many unique images we found in this batch
                        logger.info(f"Downloaded {new_downloads} new unique images (variation '{variation['name']}', page {page})")
                    
                    # Check if we've downloaded enough images
                    if downloaded_count >= target_total:
                        logger.info(f"Reached target of {target_total} images")
                        break
                    
                    # Check if this batch had fewer images than expected (might be duplicate-heavy)
                    # If we got less than 25% unique images, move to next variation
                    if len(download_batch) > 0 and new_downloads < len(download_batch) * 0.25:
                        logger.info(f"Low unique image rate ({new_downloads}/{len(download_batch)}), trying next variation")
                        break
                    
                    # Update remaining target
                    remaining_target = target_total - downloaded_count
                    
                    # Move to next page
                    page += 1
                    
                    # Break if we've hit the API limit for this variation (3 pages = ~600 images)
                    if page > 3:
                        logger.info("Reached API limit for this variation (3 pages), moving to next variation")
                        break
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error making request: {e}")
                    if hasattr(e, 'response') and e.response:
                        if e.response.status_code == 429:  # Rate limit
                            wait_time = int(e.response.headers.get('X-RateLimit-Reset', 60)) + 5
                            logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        elif e.response.status_code == 400:  # Bad request
                            logger.warning(f"API returned error 400 on page {page} - likely hit the result limit")
                            break
                        
                    if skip_errors:
                        logger.warning("Continuing despite API error")
                        time.sleep(5)
                        continue
                    
                    # Move to next variation on error
                    break
    
    finally:
        pbar.close()
    
    # Report results
    if downloaded_count > 0:
        logger.info(f"\nSuccessfully downloaded {downloaded_count} unique images to {output_dir}")
        
        # Check actual files vs reported downloads
        actual_files = len(glob.glob(os.path.join(output_dir, "pixabay_*.jpg")))
        if actual_files != downloaded_count:
            logger.warning(f"Discrepancy detected: {downloaded_count} reported downloads vs {actual_files} files found")
    
    if failed_downloads:
        logger.warning(f"{len(failed_downloads)} images failed to download after {retries} retries")
        logger.warning("Error types encountered:")
        for error_type, count in error_types.most_common():
            logger.warning(f"  {error_type}: {count} occurrences")
        
        if error_log_path:
            save_error_log(error_log_path, {
                'failed_downloads': failed_downloads,
                'error_types': dict(error_types),
                'total_attempted': downloaded_count + len(failed_downloads),
                'total_successful': downloaded_count,
                'total_failed': len(failed_downloads)
            })
    
    # Final progress save
    if save_progress:
        save_download_progress(output_dir, downloaded_ids)
    
    return downloaded_count

def download_batch_with_retry(batch, output_dir, retries=3, debug=False):
    """Download a batch of images with retry logic and parallel processing"""
    successful = []
    failed = []
    lock = threading.Lock()
    
    # Create progress bar for this batch
    batch_progress = tqdm(total=len(batch), 
                         desc="Downloading batch", 
                         unit="img",
                         position=1,
                         leave=False,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def download_worker(item):
        """Worker function to download a single image with retries"""
        image_url = item['url']
        filename = item['filename']
        save_path = os.path.join(output_dir, filename)
        
        for attempt in range(retries + 1):
            try:
                response = requests.get(image_url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                with lock:
                    successful.append(save_path)
                    batch_progress.update(1)
                return
                
            except Exception as e:
                if attempt == retries:
                    with lock:
                        item['error_type'] = type(e).__name__
                        item['error_message'] = str(e)
                        failed.append(item)
                        batch_progress.update(1)
                    if debug:
                        logger.debug(f"Failed to download {filename} after {retries} attempts: {e}")
                else:
                    time.sleep(1)  # Brief pause before retry
    
    try:
        # Create and start threads
        threads = []
        for item in batch:
            thread = threading.Thread(target=download_worker, args=(item,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
    except Exception as e:
        logger.error(f"Error in batch download: {e}")
        if debug:
            logger.error(traceback.format_exc())
        
        # Add all remaining items as failed
        with lock:
            for item in batch:
                if item not in failed and os.path.join(output_dir, item['filename']) not in successful:
                    item['error_type'] = 'batch_exception'
                    item['error_message'] = str(e)
                    failed.append(item)
    
    finally:
        batch_progress.close()
    
    return successful, failed

def build_search_params(args):
    """Build search parameters from command-line arguments"""
    search_params = {
        "q": args.query,
        "page": 1,
        "per_page": args.per_page,
        "image_type": args.type,
        "orientation": args.orientation,
        "order": args.order,
        "lang": args.lang,
        "safesearch": 'true' if args.safesearch else 'false'
    }
    
    # Add optional parameters if provided
    if args.category:
        search_params["category"] = args.category
    if args.min_width:
        search_params["min_width"] = args.min_width
    if args.min_height:
        search_params["min_height"] = args.min_height
    if args.colors:
        search_params["colors"] = args.colors
    if args.editors_choice:
        search_params["editors_choice"] = 'true'
        
    return search_params

def configure_logging(debug=False, verbose=False):
    """Configure logging levels based on command-line flags"""
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    elif verbose:
        logger.setLevel(logging.INFO)
        logger.info("Verbose logging enabled")
    else:
        logger.setLevel(logging.WARNING)

def get_api_key(key_arg=None):
    """Get Pixabay API key from arguments or .env file"""
    # First try from command-line argument
    if key_arg:
        logger.debug("Using API key from command-line argument")
        return key_arg
        
    # Then try from environment variable
    env_key = os.environ.get('PIXABAY_API_KEY')
    if env_key:
        logger.debug("Using API key from environment variable")
        return env_key
        
    # Try to load from .env file
    try:
        dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if os.path.exists(dotenv_path):
            with open(dotenv_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('PIXABAY_API_KEY='):
                        api_key = line.strip().split('=', 1)[1].strip()
                        if api_key and api_key != '':
                            # Remove quotes if present
                            if (api_key.startswith('"') and api_key.endswith('"')) or \
                               (api_key.startswith("'") and api_key.endswith("'")):
                                api_key = api_key[1:-1]
                            logger.debug("Using API key from .env file")
                            return api_key
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
    
    # API key not found
    logger.error("No Pixabay API key found. Please set PIXABAY_API_KEY in .env file or provide with --key argument.")
    print("ERROR: Pixabay API key is required. You can:")
    print("1. Create a .env file with PIXABAY_API_KEY=your_key_here")
    print("2. Set PIXABAY_API_KEY environment variable")
    print("3. Use the --key command-line argument")
    print("\nGet your API key at https://pixabay.com/api/docs/")
    return None

def clean_directory_name(name):
    """Clean a string to be used as a directory name."""
    if not name:
        return "general"
    # Replace special characters and spaces
    cleaned = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in name)
    # Replace spaces with underscores and trim
    cleaned = cleaned.replace(' ', '_').strip()
    # Limit length
    return cleaned[:50]  # Limit to 50 characters

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Setup logging
    configure_logging(args.debug, args.verbose)
    
    # Authenticate API client
    api_key = get_api_key(args.key)
    if not api_key:
        return
    
    # Create search parameters
    client = {"api_key": api_key}
    search_params = build_search_params(args)
    
    # Add API key to search parameters
    search_params["key"] = api_key
    
    if args.debug:
        logger.debug(f"Search parameters: {search_params}")
    
    # Set or create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create date-based directory structure
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        query_dir = clean_directory_name(args.query if args.query else "general")
        output_dir = os.path.join("Output", date_dir, query_dir)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Check for query if not resuming
    if not args.query and not args.resume:
        args.query = input("Enter search term: ")
        search_params["q"] = args.query
    
    # Print search settings
    logger.info(f"Searching for: {args.query or 'Resuming previous search'}")
    logger.info(f"Maximum images: {args.max_images if args.max_images > 0 else 'All available'}")
    logger.info(f"Image type: {args.type}")
    
    # Add recommendations for large downloads
    if args.max_images > 100 or args.max_images <= 0:
        logger.info("LARGE DOWNLOAD RECOMMENDATIONS:")
        logger.info("  - For optimal performance with large downloads:")
        logger.info("    * Use --per-page 200 (API maximum)")
        logger.info("    * Use --batch-size 200 (or higher for even faster downloads)")
        logger.info("    * Enable --save-progress to resume interrupted downloads")
        if args.per_page < 200:
            logger.info(f"  - NOTICE: Your current --per-page setting ({args.per_page}) is below optimal 200")
        if args.batch_size < 200 and args.batch_size > 0:
            logger.info(f"  - NOTICE: Your current --batch-size setting ({args.batch_size}) is below optimal 200")
        if not args.save_progress:
            logger.info("  - NOTICE: Progress saving is disabled. Enable with --save-progress")
    
    # Log additional settings
    logger.info(f"Using batch size: {args.batch_size if args.batch_size > 0 else f'auto (based on per_page: {args.per_page})'}")
    logger.info(f"Retry attempts: {args.retries}")
    logger.info(f"Progress saving: {'Enabled' if args.save_progress else 'Disabled'}")
    logger.info(f"Error handling: {'Skip and continue' if args.skip_errors else 'Stop on API errors'}")
    
    # Create error log path if specified
    error_log_path = None
    if args.error_log:
        if args.error_log == "auto":
            error_log_path = os.path.join(output_dir, "error_log.json")
        else:
            error_log_path = args.error_log
        logger.info(f"Will save error log to: {error_log_path}")
    
    # Download images
    start_time = time.time()
    downloaded = download_all_images(
        client=client,
        search_params=search_params,
        output_dir=output_dir,
        max_images=args.max_images,
        quality=args.quality,
        verbose=args.verbose,
        debug=args.debug,
        retries=args.retries,
        batch_size=args.batch_size,
        error_log_path=error_log_path,
        skip_errors=args.skip_errors,
        save_progress=args.save_progress,
        resume=args.resume
    )
    
    # Calculate download statistics
    elapsed = time.time() - start_time
    rate = downloaded / elapsed if elapsed > 0 else 0
    
    # Report statistics
    logger.info(f"Download completed in {elapsed:.2f} seconds")
    logger.info(f"Downloaded {downloaded} images ({rate:.2f} images/second)")
    
    # Check actual number of files vs reported downloaded
    actual_files = len(glob.glob(os.path.join(output_dir, "pixabay_*.jpg")))
    if actual_files != downloaded:
        logger.warning(f"Discrepancy detected: {downloaded} reported downloads vs {actual_files} files found on disk")
        logger.warning("Possible causes: Some files may have been skipped or failed to download")
        if not args.error_log:
            logger.warning("Recommendation: Use --error-log option to track failed downloads")
    
    # Suggest next steps
    if downloaded > 0:
        logger.info("Next steps:")
        if args.save_progress:
            logger.info(f"  - To resume this download later: python {sys.argv[0]} --resume --output-dir \"{output_dir}\"")
        logger.info(f"  - Your images are saved in: {os.path.abspath(output_dir)}")
    elif args.save_progress:
        logger.info(f"To retry this download: python {sys.argv[0]} --resume --output-dir \"{output_dir}\"")

if __name__ == "__main__":
    main() 