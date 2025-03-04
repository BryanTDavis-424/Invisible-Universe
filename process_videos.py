from pathlib import Path

from helpers.frame_extraction import extract_frames
from helpers.generate_embeddings import generate_frame_embeddings
from helpers.store_embeddings import create_vector_index, save_index

def process_videos(videos_dir, frames_dir, index_path, paths_path):
    # Extract frames from all videos
    for video_path in Path(videos_dir).glob("*.mp4"):
        extract_frames(str(video_path), frames_dir)
    
    # Generate emgenerate_frame_embeddings
    embeddings = generate_frame_embeddings(frames_dir)
    
    # Create and save thecreate_vector_index
    index, frame_paths = create_vector_index(embeddings)
    save_index(index, frame_paths, index_path, paths_path)
    
    print(f"Processed {len(frame_paths)} frames from videos in {videos_dir}")

if __name__ == "__main__":
    videos_dir = "videos"  # Directory containing the Ember videos
    frames_dir = "frames"  # Directory to save extracted frames
    index_path = "index.faiss"  # Path to save the FAISS index
    paths_path = "frame_paths.pkl"  # Path to save the frame paths
    
    process_videos(videos_dir, frames_dir, index_path, paths_path)  