import torch
from PIL import Image
from pathlib import Path

try:
    import clip
except ImportError:
    raise ImportError("OpenAI CLIP is not installed. Install it with: pip install git+https://github.com/openai/CLIP.git")

def generate_frame_embeddings(frames_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
    except AttributeError:
        raise ImportError("You may have the wrong 'clip' package installed. The OpenAI CLIP model can be installed with: pip install git+https://github.com/openai/CLIP.git")
    
    embeddings = {}
    
    for frame_path in Path(frames_dir).glob("*.jpg"):
        image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        embeddings[str(frame_path)] = image_features.cpu().numpy()[0]
    
    return embeddings