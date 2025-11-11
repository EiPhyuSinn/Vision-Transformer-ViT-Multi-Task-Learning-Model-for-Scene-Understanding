import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import json
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
from main import Patches, PatchEncoder, MultiHeadSelfAttention, TransformerBlock, VisionTransformerMultiTask

# Define categories (must match training)
weather_categories = ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined']
scene_categories = ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined']
timeofday_categories = ['dawn/dusk', 'daytime', 'night', 'undefined']


# ==================== INFERENCE DATASET ====================
class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Store file path for later image saving
        file_name = os.path.basename(image_path)
        
        if self.transform:
            image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'file_path': image_path,
            'file_name': file_name  # Add file name separately
        }
# ==================== VISUALIZATION FUNCTIONS ====================
def create_annotated_image(original_image, predictions, output_path):
    """Create and save annotated image with predictions"""
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(original_image)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot original image
    ax1.imshow(img_array)
    ax1.set_title('Original Image', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Plot predictions
    ax2.axis('off')
    ax2.text(0.1, 0.9, 'PREDICTION RESULTS', fontsize=18, fontweight='bold', 
             transform=ax2.transAxes, color='darkblue')
    
    # Weather prediction
    ax2.text(0.1, 0.75, f"Weather: {predictions['weather']}", fontsize=14, 
             transform=ax2.transAxes, color='green', fontweight='bold')
    ax2.text(0.1, 0.70, f"Confidence: {predictions['weather_confidence']:.3f}", 
             fontsize=12, transform=ax2.transAxes, color='darkgreen')
    
    # Scene prediction
    ax2.text(0.1, 0.60, f"Scene: {predictions['scene']}", fontsize=14, 
             transform=ax2.transAxes, color='blue', fontweight='bold')
    ax2.text(0.1, 0.55, f"Confidence: {predictions['scene_confidence']:.3f}", 
             fontsize=12, transform=ax2.transAxes, color='darkblue')
    
    # Time of day prediction
    ax2.text(0.1, 0.45, f"Time of Day: {predictions['timeofday']}", fontsize=14, 
             transform=ax2.transAxes, color='purple', fontweight='bold')
    ax2.text(0.1, 0.40, f"Confidence: {predictions['timeofday_confidence']:.3f}", 
             fontsize=12, transform=ax2.transAxes, color='darkblue')
    
    # Add probability bars for top 3 predictions in each category
    y_pos = 0.30
    
    # Weather probabilities
    ax2.text(0.1, y_pos, "Weather Probabilities:", fontsize=12, 
             transform=ax2.transAxes, color='black', fontweight='bold')
    y_pos -= 0.05
    
    weather_probs = predictions['weather_probabilities']
    top_weather = sorted(weather_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for weather, prob in top_weather:
        ax2.text(0.15, y_pos, f"{weather}: {prob:.3f}", fontsize=10, 
                transform=ax2.transAxes, color='gray')
        y_pos -= 0.04
    
    # Scene probabilities
    y_pos -= 0.02
    ax2.text(0.1, y_pos, "Scene Probabilities:", fontsize=12, 
             transform=ax2.transAxes, color='black', fontweight='bold')
    y_pos -= 0.05
    
    scene_probs = predictions['scene_probabilities']
    top_scene = sorted(scene_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for scene, prob in top_scene:
        ax2.text(0.15, y_pos, f"{scene}: {prob:.3f}", fontsize=10, 
                transform=ax2.transAxes, color='gray')
        y_pos -= 0.04
    
    # Time of day probabilities
    y_pos -= 0.02
    ax2.text(0.1, y_pos, "Time of Day Probabilities:", fontsize=12, 
             transform=ax2.transAxes, color='black', fontweight='bold')
    y_pos -= 0.05
    
    time_probs = predictions['timeofday_probabilities']
    top_time = sorted(time_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for timeofday, prob in top_time:
        ax2.text(0.15, y_pos, f"{timeofday}: {prob:.3f}", fontsize=10, 
                transform=ax2.transAxes, color='gray')
        y_pos -= 0.04
    
    # Add file name
    ax2.text(0.1, 0.02, f"File: {predictions['file_name']}", fontsize=10, 
             transform=ax2.transAxes, color='darkred', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

# ==================== INFERENCE FUNCTIONS ====================
def load_model(model_path, device):
    """Load the trained model"""
    model = VisionTransformerMultiTask()
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    return model

class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        file_name = os.path.basename(image_path)
        
        if self.transform:
            image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'file_path': image_path,
            'file_name': file_name
        }

def predict_batch(model, dataloader, device, output_dir):
    """Perform batch inference and save annotated images"""
    all_predictions = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            file_paths = batch['file_path']
            file_names = batch['file_name']
            
            outputs = model(images)
            
            # Get predictions with probabilities
            weather_probs = torch.softmax(outputs['weather'], dim=1)
            scene_probs = torch.softmax(outputs['scene'], dim=1)
            timeofday_probs = torch.softmax(outputs['timeofday'], dim=1)
            
            weather_preds = torch.argmax(outputs['weather'], dim=1)
            scene_preds = torch.argmax(outputs['scene'], dim=1)
            timeofday_preds = torch.argmax(outputs['timeofday'], dim=1)
            
            # Convert to numpy
            weather_preds_np = weather_preds.cpu().numpy()
            scene_preds_np = scene_preds.cpu().numpy()
            timeofday_preds_np = timeofday_preds.cpu().numpy()
            
            weather_probs_np = weather_probs.cpu().numpy()
            scene_probs_np = scene_probs.cpu().numpy()
            timeofday_probs_np = timeofday_probs.cpu().numpy()
            
            # Process each image in batch
            for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
                # Load original image separately for annotation
                original_image = Image.open(file_path).convert('RGB')
                base_name = os.path.splitext(file_name)[0]
                
                prediction = {
                    'file_name': file_name,
                    'file_path': file_path,
                    'weather': weather_categories[weather_preds_np[i]],
                    'weather_confidence': float(weather_probs_np[i][weather_preds_np[i]]),
                    'scene': scene_categories[scene_preds_np[i]],
                    'scene_confidence': float(scene_probs_np[i][scene_preds_np[i]]),
                    'timeofday': timeofday_categories[timeofday_preds_np[i]],
                    'timeofday_confidence': float(timeofday_probs_np[i][timeofday_preds_np[i]]),
                    'weather_probabilities': {cat: float(prob) for cat, prob in zip(weather_categories, weather_probs_np[i])},
                    'scene_probabilities': {cat: float(prob) for cat, prob in zip(scene_categories, scene_probs_np[i])},
                    'timeofday_probabilities': {cat: float(prob) for cat, prob in zip(timeofday_categories, timeofday_probs_np[i])}
                }
                
                # Save annotated image
                output_image_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
                create_annotated_image(original_image, prediction, output_image_path)
                
                # Also save simple version
                output_simple_path = os.path.join(output_dir, f"{base_name}_simple.jpg")
                
                prediction['annotated_image_path'] = output_image_path
                prediction['simple_image_path'] = output_simple_path
                
                all_predictions.append(prediction)
                
                print(f"Processed: {file_name} -> {output_image_path}")
    
    return all_predictions

def save_predictions(predictions, output_file, output_dir):
    """Save predictions to CSV and JSON files"""
    # Create simplified version for CSV
    simplified_predictions = []
    for pred in predictions:
        simplified = {
            'file_name': pred['file_name'],
            'weather': pred['weather'],
            'weather_confidence': pred['weather_confidence'],
            'scene': pred['scene'],
            'scene_confidence': pred['scene_confidence'],
            'timeofday': pred['timeofday'],
            'timeofday_confidence': pred['timeofday_confidence'],
            'annotated_image': pred['annotated_image_path'],
            'simple_image': pred['simple_image_path']
        }
        simplified_predictions.append(simplified)
    
    # Save as CSV
    csv_file = os.path.join(output_dir, "predictions.csv")
    df = pd.DataFrame(simplified_predictions)
    df.to_csv(csv_file, index=False)
    print(f"Predictions saved to {csv_file}")
    
    # Save detailed version as JSON
    json_file = os.path.join(output_dir, "predictions_detailed.json")
    with open(json_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Detailed predictions saved to {json_file}")

def print_predictions_summary(predictions):
    """Print a summary of predictions"""
    weather_counts = {}
    scene_counts = {}
    timeofday_counts = {}
    
    for pred in predictions:
        weather = pred['weather']
        scene = pred['scene']
        timeofday = pred['timeofday']
        
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        timeofday_counts[timeofday] = timeofday_counts.get(timeofday, 0) + 1
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(predictions)}")
    
    print("\nWeather Distribution:")
    for weather, count in sorted(weather_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(predictions)) * 100
        print(f"  {weather}: {count} ({percentage:.1f}%)")
    
    print("\nScene Distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(predictions)) * 100
        print(f"  {scene}: {count} ({percentage:.1f}%)")
    
    print("\nTime of Day Distribution:")
    for timeofday, count in sorted(timeofday_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(predictions)) * 100
        print(f"  {timeofday}: {count} ({percentage:.1f}%)")
    print("="*60)

# ==================== MAIN INFERENCE SCRIPT ====================
def main():
    # Configuration
    MODEL_PATH = "vision_transformer_multitask.pth"  # Path to your trained model
    IMAGE_FOLDER = "/home/gwm-279/Desktop/multitask_classification/Cross-stitch-Networks-for-Multi-task-Learning/weather_old/archive/test_dataset/test_images"               # Folder containing images for inference
    OUTPUT_DIR = "inference_results"                 # Output directory for results
    BATCH_SIZE = 8                                   # Batch size for inference
    SUPPORTED_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' not found.")
        print("Please create the folder and add images, or update the IMAGE_FOLDER path.")
        return
    
    # Find all image files
    image_paths = []
    for extension in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, extension)))
        image_paths.extend(glob.glob(os.path.join(IMAGE_FOLDER, extension.upper())))
    
    if not image_paths:
        print(f"No images found in {IMAGE_FOLDER} with supported extensions: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"Found {len(image_paths)} images for inference")
    
    # Data transforms (must match training)
    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset = InferenceDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Perform inference
    print("Starting batch inference...")
    predictions = predict_batch(model, dataloader, device, OUTPUT_DIR)

    
    # Print summary
    print_predictions_summary(predictions)
    
    print(f"\nInference completed!")
    print(f"Annotated images saved to: {OUTPUT_DIR}")
    print(f"Each image has two versions:")
    print(f"  - '*_annotated.jpg': Detailed matplotlib version with probabilities")
    print(f"  - '*_simple.jpg': Simple PIL version with just predictions")

if __name__ == "__main__":
    main()