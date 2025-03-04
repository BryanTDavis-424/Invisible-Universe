# Video Frame Semantic Search API

This project implements a Python-based API that enables semantic search on a video library, returning relevant images (video frames) based on a user's text query.

## Overview

The system extracts frames from videos, generates semantic embeddings for each frame, and provides an API endpoint that allows users to search for relevant frames using natural language queries.

## Features

- Frame extraction from videos at configurable intervals
- Semantic search using CLIP (Contrastive Language-Image Pre-training) model
- FastAPI-based REST API with health check endpoint
- Returns up to 4 most relevant frames based on query similarity

## Technical Approach

### 1. Frame Extraction

The system extracts frames from videos at regular intervals using OpenCV:

- **Approach**: We extract frames at fixed time intervals (default: 1 second) rather than every frame to reduce redundancy while maintaining good coverage.
- **Design Decision**: Using OpenCV provides a balance between performance and ease of implementation. The frame interval is configurable to adjust the density of extracted frames based on video content.

### 2. Semantic Embedding Generation

We use OpenAI's CLIP (Contrastive Language-Image Pre-training) model to generate embeddings for both images and text:

- **Approach**: CLIP is a multimodal model trained on a variety of image-text pairs, making it ideal for semantic search across modalities.
- **Design Decision**: CLIP was chosen because it can understand both visual content and natural language queries in the same embedding space, enabling semantic matching between text descriptions and visual content without requiring explicit training on our specific dataset.

### 3. Vector Storage and Retrieval

We use FAISS (Facebook AI Similarity Search) for efficient similarity search:

- **Approach**: Frame embeddings are stored in a FAISS index optimized for inner product similarity search (cosine similarity).
- **Design Decision**: FAISS was selected for its high performance with large vector datasets and ability to perform efficient nearest neighbor searches. The IndexFlatIP implementation provides exact search results with reasonable performance for our dataset size.

### 4. API Implementation

The API is built using FastAPI:

- **Approach**: A simple REST API with endpoints for search and health checking.
- **Design Decision**: FastAPI was chosen for its performance, automatic documentation generation, and type checking. The static file serving capability makes it easy to serve frame images directly from the API.

## Project Structure

```
ember-video-search/
├── app.py                # FastAPI application
├── process_videos.py     # Script to process videos and create index
├── helpers/
│   ├── frame_extraction.py     # Functions for extracting frames from videos
│   ├── generate_embeddings.py  # Functions for generating CLIP embeddings
│   └── store_embeddings.py     # Functions for creating and saving FAISS index
├── videos/               # Directory containing the Ember videos
├── frames/               # Directory to store extracted frames
├── index.faiss           # FAISS index file
├── frame_paths.pkl       # Pickle file with frame paths
└── requirements.txt      # Dependencies
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Process videos to extract frames and create the search index:
   ```
   python process_videos.py
   ```
4. Start the API server:
   ```
   uvicorn app:app --reload
   ```

## API Documentation

### Search Endpoint

```
GET /search?query=<search_query>
```

Parameters:
- `query` (string): The search query describing the desired image content

Returns up to 4 most relevant frames based on the query.

Example request:
```
GET /search?query=Ember holding phone
```

Example response:
```json
{
  "results": [
    {"image_url": "/frames/video1_frame_120.jpg"},
    {"image_url": "/frames/video3_frame_45.jpg"},
    {"image_url": "/frames/video2_frame_78.jpg"},
    {"image_url": "/frames/video1_frame_125.jpg"}
  ]
}
```

## Performance Considerations

- Frame extraction interval can be adjusted based on video content and desired granularity
- For larger video libraries, consider:
  - Using a more efficient FAISS index type (e.g., IndexIVFFlat)
  - Implementing pagination for search results
  - Moving frame storage to cloud storage (S3, GCS)
  - Adding caching for frequent queries

## Future Improvements

- Implement scene detection for more intelligent frame extraction
- Add filtering options (by video source, timestamp, etc.)
- Implement user feedback mechanism to improve search results
- Add support for video segment retrieval rather than just individual frames
- Implement batch processing for large video libraries

## Dependencies

- FastAPI: Web framework for building the API
- OpenCV: Video processing and frame extraction
- CLIP: Multimodal embeddings for semantic search
- FAISS: Efficient similarity search
- PyTorch: Deep learning framework for CLIP
- Uvicorn: ASGI server for running the FastAPI application
