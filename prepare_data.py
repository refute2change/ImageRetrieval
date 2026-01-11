import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. CONFIGURATION
# Make sure this matches your folder name
IMAGE_FOLDER = 'seg_test' 
OUTPUT_VECTORS = 'db_vectors.npy'
OUTPUT_FILENAMES = 'image_filenames.pkl'

print("Loading CLIP model...")
model = SentenceTransformer('clip-ViT-B-32')

# 2. READ IMAGES RECURSIVELY
images = []
valid_filenames = []

print(f"Scanning '{IMAGE_FOLDER}' recursively... (This will take a while)")

# Walk through all subfolders
for root, dirs, files in os.walk(IMAGE_FOLDER):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, filename)
            
            try:
                # Relative path (e.g. "buildings/10.jpg") for the app to use later
                relative_path = os.path.relpath(full_path, IMAGE_FOLDER)
                
                # Load image
                img = Image.open(full_path).convert('RGB')
                
                images.append(img)
                valid_filenames.append(relative_path)
                
                # Simple progress counter
                if len(images) % 1000 == 0:
                    print(f"Loaded {len(images)} images so far...")
                    
            except Exception as e:
                print(f"Error loading {filename}: {e}")

# 3. ENCODE AND SAVE
if images:
    print(f"Finished loading. Now generating vectors for all {len(images)} images.")
    print("This step is heavy on the CPU. Please wait...")
    
    # Batch encoding (batch_size=32 is standard for CPU)
    embeddings = model.encode(images, batch_size=32, show_progress_bar=True)

    # Save to disk
    np.save(OUTPUT_VECTORS, embeddings)
    with open(OUTPUT_FILENAMES, 'wb') as f:
        pickle.dump(valid_filenames, f)
        
    print("Success! Full dataset prepared.")
    print(f"Total images processed: {len(images)}")
else:
    print("No images found! Please check your folder name.")