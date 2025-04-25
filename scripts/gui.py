#!/usr/bin/env python3
import os
import sys
import gradio as gr
from pixabay import (
    DEFAULT_SETTINGS, parse_args, download_all_images, get_api_key, build_search_params,
    LANGUAGES, CATEGORIES, ORDER_OPTIONS  # Import constants directly
)
import logging
from datetime import datetime

# Configure logging to capture output
logging.basicConfig(level=logging.INFO, format='%(message)s')

def clean_directory_name(name):
    """Clean a string to be used as a directory name."""
    if not name:
        return "general"
    cleaned = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in name)
    cleaned = cleaned.replace(' ', '_').strip()
    return cleaned[:50]

def download_images(
    query: str,
    max_images: int,
    image_type: str,
    orientation: str,
    category: str,
    quality: str,
    order: str,
    lang: str,
    safesearch: bool,
    editors_choice: bool,
    skip_errors: bool,
    progress=gr.Progress(track_tqdm=True)
):
    """Handle the download process and return status"""
    try:
        # Create output directory for images
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        query_dir = clean_directory_name(query)
        output_dir = os.path.join("output", date_dir, query_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create logs directory
        logs_dir = os.path.join("logs", date_dir)
        os.makedirs(logs_dir, exist_ok=True)
        error_log_path = os.path.join(logs_dir, f"{query_dir}_error_log.json")
        
        # Get API key
        api_key = get_api_key()
        if not api_key:
            return "Error: No API key found. Please set PIXABAY_API_KEY in .env file."
        
        # Build search parameters
        search_params = {
            "q": query,
            "image_type": image_type,
            "orientation": orientation,
            "order": order,
            "lang": lang,
            "per_page": 200,  # Always use maximum page size
            "safesearch": safesearch,
            "editors_choice": editors_choice,
            "key": api_key
        }
        
        if category:
            search_params["category"] = category
        
        # Create client
        client = {"api_key": api_key}
        
        # Download images
        downloaded = download_all_images(
            client=client,
            search_params=search_params,
            output_dir=output_dir,
            max_images=max_images,
            quality=quality,
            verbose=True,
            debug=False,
            retries=DEFAULT_SETTINGS['MAX_RETRIES'],
            batch_size=200,  # Use max batch size for optimal performance
            error_log_path=error_log_path,
            skip_errors=skip_errors,
            save_progress=False,
            resume=False
        )
        
        if downloaded > 0:
            return f"Successfully downloaded {downloaded} images to:\n{os.path.abspath(output_dir)}"
        else:
            return "No images were downloaded. Please check your search criteria."
            
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Pixabay Image Downloader", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Pixabay Image Downloader")
        
        with gr.Row():
            with gr.Column():
                # Basic options
                query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter search terms...",
                    info="What images are you looking for?"
                )
                
                max_images = gr.Number(
                    label="Maximum Images",
                    value=0,
                    info="0 = download all available images using smart variations to get more than 500",
                    precision=0
                )
                
                image_type = gr.Dropdown(
                    choices=['all', 'photo', 'illustration', 'vector'],
                    value=DEFAULT_SETTINGS['IMAGE_TYPE'],
                    label="Image Type"
                )
                
                orientation = gr.Dropdown(
                    choices=['all', 'horizontal', 'vertical'],
                    value=DEFAULT_SETTINGS['ORIENTATION'],
                    label="Orientation"
                )
                
                category = gr.Dropdown(
                    choices=[''] + CATEGORIES,
                    value='',
                    label="Category",
                    info="Optional: Filter by category"
                )
                
            with gr.Column():
                # Download options
                quality = gr.Dropdown(
                    choices=['highest', 'high', 'medium', 'low'],
                    value=DEFAULT_SETTINGS['QUALITY'],
                    label="Image Quality"
                )
                
                order = gr.Dropdown(
                    choices=ORDER_OPTIONS,
                    value=DEFAULT_SETTINGS['ORDER'],
                    label="Sort Order"
                )
                
                lang = gr.Dropdown(
                    choices=LANGUAGES,
                    value=DEFAULT_SETTINGS['LANG'],
                    label="Language"
                )
                
                # Boolean options (moved to right column)
                safesearch = gr.Checkbox(
                    label="Safe Search",
                    value=False,
                    info="Only show images suitable for all ages"
                )
                
                editors_choice = gr.Checkbox(
                    label="Editor's Choice",
                    value=False,
                    info="Only show images selected by Pixabay editors"
                )
                
                skip_errors = gr.Checkbox(
                    label="Skip Errors",
                    value=DEFAULT_SETTINGS['SKIP_ERRORS'],
                    info="Continue downloading if errors occur"
                )
        
        # Download button and output
        download_btn = gr.Button("Start Download", variant="primary")
        output = gr.Textbox(label="Status", interactive=False)
        
        # Handle the download
        download_btn.click(
            fn=download_images,
            inputs=[
                query, max_images, image_type, orientation, category,
                quality, order, lang, safesearch, editors_choice, skip_errors
            ],
            outputs=output
        )
    
    return interface

# Instructions for LLMs editing this file
# Launch with public sharing enabled
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="127.0.0.1",  # Only allow local connections
        show_api=False,          # Don't show API docs
        share=False             # Don't create public URL
    ) 