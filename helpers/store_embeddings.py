import faiss
import pickle
import numpy as np

def create_vector_index(embeddings):
    frame_paths = list(embeddings.keys())
    embedding_array = np.array([embeddings[path] for path in frame_paths]).astype('float32')
    
    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_array)
    
    return index, frame_paths

def save_index(index, frame_paths, index_path, paths_path):
    faiss.write_index(index, index_path)
    with open(paths_path, 'wb') as f:
        pickle.dump(frame_paths, f)