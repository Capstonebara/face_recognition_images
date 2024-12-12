import sys
import os
import torch
from torchvision import transforms
import json  # Import json module for saving embeddings in JSON format

# Add edgeface folder to the Python path (ensure this is correct based on your folder structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../edgeface')))

# Import local modules
from face_alignment import align  # Ensure this is correctly pointing to your local 'face_alignment' module
from backbones import get_model  # Ensure this is correctly pointing to your local 'backbones' module

# Set up directories for input and output

# Ensure the output folder exists

# Load the EdgeFace model (use local get_model function instead of downloading from GitHub)
model_name = "edgeface_s_gamma_05"  # or other variants like edgeface_xs_gamma_06
model = get_model(model_name)  # This should load the model from the local repo
model.eval()  # Set the model to evaluation mode

# Define image preprocessing steps
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def process_and_embed_face(image_path):
    """
    Process and embed a face image.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        torch.Tensor: The embedding for the face.
    """
    # Align face (ensure you have the alignment function available)
    aligned_face = align.get_aligned_face(image_path)
    if aligned_face is None:
        print(f"Error aligning face for image {image_path}")
        return None
    
    # Apply the transformation
    transformed_face = transform(aligned_face).unsqueeze(0)  # Add batch dimension
    
    # Extract embedding using the EdgeFace model
    with torch.no_grad():
        embedding = model(transformed_face)
    
    return embedding

def save_embedding(embedding, image_name, embedded_face_folder):
    """
    Save the embedding to a JSON file.
    
    Args:
        embedding (torch.Tensor): The face embedding.
        image_name (str): The name of the image (used for the embedding file).
    """
    embedding_path = os.path.join(embedded_face_folder, f"{image_name}_embedding.json")
    
    # Convert the embedding to a list for JSON serialization
    embedding_list = embedding.squeeze(0).tolist()  # Remove batch dimension and convert to list
    
    # Save the embedding as a JSON file
    with open(embedding_path, 'w') as f:
        json.dump(embedding_list, f)
    
    print(f"Saved embedding for {image_name} to {embedding_path}")

def process_faces_for_embedding(face_images_folder, embedded_face_folder):
    os.makedirs(embedded_face_folder, exist_ok=True)
    """
    Process all images in the face_images folder, extract embeddings, and save them.
    """
    # Check if the input folder exists
    if not os.path.exists(face_images_folder):
        print(f"The folder {face_images_folder} does not exist.")
        return
    
    # Process each image in the folder
    for image_name in os.listdir(face_images_folder):
        image_path = os.path.join(face_images_folder, image_name)
        
        # Only process files with valid image extensions
        if image_name.lower().endswith(('jpg', 'jpeg', 'png')):
            print(f"Processing image {image_name}...")
            
            # Get the embedding for the image
            embedding = process_and_embed_face(image_path)
            if embedding is not None:
                # Save the embedding to the embedded_faces folder
                save_embedding(embedding, os.path.splitext(image_name)[0], embedded_face_folder)

if __name__ == "__main__":
    # Process all face images in the folder and save the embeddings
    process_faces_for_embedding()
