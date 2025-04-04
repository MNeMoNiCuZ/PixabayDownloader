# PixabayDownloader

PixabayDownloader is a powerful tool designed to search and download images from Pixabay using their API. This application streamlines the process of downloading multiple images based on search queries with various filtering options.

<!-- 
Replace this placeholder with an actual screenshot of your application.
You can generate a screenshot by:
1. Running the application
2. Taking a screenshot (Windows: Win+Shift+S, Mac: Cmd+Shift+4)
3. Saving it to Documentation/screenshot.png
4. Updating the path below if needed
-->
![image](https://github.com/user-attachments/assets/37d20358-1ba6-4926-a4a1-151e25ad1f15)

*Screenshot: Example of the Pixabay Downloader GUI interface*

## Features

- Simple search interface with comprehensive filtering options
- Smart search variations to download more than 500 images per query (Pixabay API limit)
- Customizable image quality options (highest, high, medium, low)
- Various image filters (type, orientation, category, etc.)
- Batch processing for optimal download speeds
- Progress tracking with real-time updates
- Error handling and detailed logging
- Ability to resume interrupted downloads
- Support for advanced API parameters
- Easy-to-use GUI interface

## Requirements

- Python 3.6 or higher
- Pixabay API key ([Sign up for free and get your API key here](https://pixabay.com/api/docs/))

## Installation

1. Clone or download this repository
2. Run `venv_create.bat` to set up your environment:
   - Choose your Python version when prompted
   - Accept the default virtual environment name (venv) or choose your own
   - Allow pip upgrade when prompted
   - Allow installation of dependencies from requirements.txt

The script will create:
- A virtual environment
- `venv_activate.bat` for activating the environment
- `venv_update.bat` for updating pip

## Usage

### GUI Mode

1. Copy or rename the `/scripts/.env.template` file to `/scripts/.env`
2. Place your Pixabay API key in the `.env` file: `PIXABAY_API_KEY=your-api-key-here`
3. Run `PixabayDownloader.bat` to start the application
4. Enter your search query and configure the download options:
   - **Search Query**: What images you're looking for
   - **Maximum Images**: How many to download (0 = all available)
   - **Image Type**: All, Photo, Illustration, or Vector
   - **Orientation**: All, Horizontal, or Vertical
   - **Category**: Filter by specific categories
   - **Image Quality**: Choose from highest to low quality
   - **Sort Order**: Popular or Latest
   - **Language**: Select search language
   - **Safe Search**: Only show images suitable for all ages
   - **Editor's Choice**: Only show images selected by Pixabay editors
5. Click "Start Download" to begin

### Command Line Mode

You can also use the tool from the command line for more advanced usage or automation:

```bash
python scripts/pixabay.py -q "search query" [options]
```

Common command-line options:
```
-q, --query         Search term for images
-n, --max-images    Maximum number of images to download (0 = all available)
-t, --type          Type of images (all, photo, illustration, vector)
--orientation       Image orientation (all, horizontal, vertical)
--category          Filter by category
--quality           Image quality to download (highest, high, medium, low)
--order             Sort order (popular, latest)
--save-progress     Save download progress for resuming
--resume            Resume previous download (requires --output-dir)
-o, --output-dir    Custom output directory
```

For a full list of options, run:
```bash
python scripts/pixabay.py --help
```

## Output Organization

- Downloaded images are saved in dated folders: `Output/YYYY-MM-DD/search_term/`
- Error logs are saved in: `logs/YYYY-MM-DD/search_term_error_log.json`
- Each image is saved with its Pixabay ID in the filename for reference

## Advanced Options

- **Skip Errors**: Continue downloading even when API errors occur
- **Save Progress**: Enable ability to resume interrupted downloads
- **Quality Options**:
  - **Highest**: Original high-resolution images (when available)
  - **High**: Large sized images
  - **Medium**: Web-optimized sizes
  - **Low**: Preview thumbnails

## Pixabay API Limits

- By default, you can make up to 100 API requests per 60 seconds
- Each search can return a maximum of 500 results per query variation
- This tool implements smart search variations to potentially download more than 500 images
- Requests must be cached for 24 hours according to Pixabay's terms
- The tool automatically handles rate limiting and will wait when limits are reached

## License

This tool is intended for personal use in accordance with the [Pixabay API Terms of Service](https://pixabay.com/service/terms/). Make sure your usage complies with Pixabay's terms.

Important notes from Pixabay's API terms:
- Returned image URLs may be used for temporarily displaying search results
- Permanent hotlinking of images is not allowed
- If you intend to use the images permanently, download them to your server first
- Systematic mass downloads are subject to Pixabay's terms of service

## Troubleshooting

- If you encounter API errors, check your API key and connection
- For rate limit errors, the tool will automatically wait and retry
- Check the error logs for detailed information about failed downloads
- Make sure you have sufficient disk space for downloads

## Contributing

Contributions to improve PixabayDownloader are welcome. Here's how you can contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to update tests and documentation as appropriate. 
