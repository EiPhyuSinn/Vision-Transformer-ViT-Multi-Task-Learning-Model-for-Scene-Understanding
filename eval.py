import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from PIL import Image
import pandas as pd

# Import the same classes from your training script
from main import CustomDataset, VisionTransformerMultiTask, weather_categories, scene_categories, timeofday_categories

class ModelEvaluator:
    def __init__(self, model, device, weather_categories, scene_categories, timeofday_categories):
        self.model = model
        self.device = device
        self.weather_categories = weather_categories
        self.scene_categories = scene_categories
        self.timeofday_categories = timeofday_categories
        
    def evaluate(self, test_loader):
        """Comprehensive evaluation of the model"""
        self.model.eval()
        
        # Storage for predictions and ground truth
        all_predictions = {
            'weather': [], 'scene': [], 'timeofday': []
        }
        all_targets = {
            'weather': [], 'scene': [], 'timeofday': []
        }
        all_probabilities = {
            'weather': [], 'scene': [], 'timeofday': []
        }
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                weather_labels = batch['weather'].cpu().numpy()
                scene_labels = batch['scene'].cpu().numpy()
                timeofday_labels = batch['timeofday'].cpu().numpy()
                
                outputs = self.model(images)
                
                # Get predictions
                weather_pred = torch.argmax(outputs['weather'], dim=1).cpu().numpy()
                scene_pred = torch.argmax(outputs['scene'], dim=1).cpu().numpy()
                timeofday_pred = torch.argmax(outputs['timeofday'], dim=1).cpu().numpy()
                
                # Get probabilities
                weather_probs = torch.softmax(outputs['weather'], dim=1).cpu().numpy()
                scene_probs = torch.softmax(outputs['scene'], dim=1).cpu().numpy()
                timeofday_probs = torch.softmax(outputs['timeofday'], dim=1).cpu().numpy()
                
                # Store results
                all_predictions['weather'].extend(weather_pred)
                all_predictions['scene'].extend(scene_pred)
                all_predictions['timeofday'].extend(timeofday_pred)
                
                all_targets['weather'].extend(weather_labels)
                all_targets['scene'].extend(scene_labels)
                all_targets['timeofday'].extend(timeofday_labels)
                
                all_probabilities['weather'].extend(weather_probs)
                all_probabilities['scene'].extend(scene_probs)
                all_probabilities['timeofday'].extend(timeofday_probs)
        
        return all_predictions, all_targets, all_probabilities
    
    def get_actual_classes(self, targets, task):
        """Get the actual classes present in the test data"""
        unique_classes = np.unique(targets[task])
        if task == 'weather':
            all_categories = self.weather_categories
        elif task == 'scene':
            all_categories = self.scene_categories
        else:
            all_categories = self.timeofday_categories
        
        # Only include categories that actually appear in the test data
        actual_categories = [all_categories[i] for i in unique_classes]
        return unique_classes, actual_categories
    
    def generate_classification_report(self, predictions, targets):
        """Generate detailed classification reports"""
        reports = {}
        
        for task in ['weather', 'scene', 'timeofday']:
            # Get actual classes present in test data
            unique_classes, actual_categories = self.get_actual_classes(targets, task)
            
            print(f"{task.upper()} - Unique classes in test data: {unique_classes}")
            print(f"{task.upper()} - Actual categories: {actual_categories}")
            
            reports[task] = classification_report(
                targets[task], 
                predictions[task], 
                labels=unique_classes,
                target_names=actual_categories, 
                output_dict=True,
                zero_division=0
            )
            
        return reports
    
    def plot_confusion_matrices(self, predictions, targets, save_path=None):
        """Plot confusion matrices for all tasks"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        tasks = ['weather', 'scene', 'timeofday']
        titles = ['Weather Classification', 'Scene Classification', 'Time of Day Classification']
        
        for idx, (task, title) in enumerate(zip(tasks, titles)):
            # Get actual classes present in test data
            unique_classes, actual_categories = self.get_actual_classes(targets, task)
            
            cm = confusion_matrix(targets[task], predictions[task], labels=unique_classes)
            accuracy = accuracy_score(targets[task], predictions[task])
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=actual_categories, yticklabels=actual_categories,
                       ax=axes[idx])
            axes[idx].set_title(f'{title}\nAccuracy: {accuracy:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
            plt.setp(axes[idx].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_overall_metrics(self, predictions, targets):
        """Calculate overall metrics"""
        metrics = {}
        
        for task in ['weather', 'scene', 'timeofday']:
            accuracy = accuracy_score(targets[task], predictions[task])
            metrics[task] = {
                'accuracy': accuracy,
                'total_samples': len(targets[task]),
                'unique_classes': len(np.unique(targets[task]))
            }
        
        # Calculate combined accuracy (average of all tasks)
        combined_accuracy = np.mean([metrics[task]['accuracy'] for task in metrics])
        metrics['combined_accuracy'] = combined_accuracy
        
        return metrics
    
    def print_detailed_report(self, reports, metrics):
        """Print detailed evaluation report"""
        print("=" * 80)
        print("VISION TRANSFORMER MULTI-TASK EVALUATION REPORT")
        print("=" * 80)
        
        # Print overall metrics
        print("\nOVERALL ACCURACIES:")
        print("-" * 40)
        for task in ['weather', 'scene', 'timeofday']:
            print(f"{task.upper():12}: {metrics[task]['accuracy']:.4f} "
                  f"({metrics[task]['total_samples']} samples, {metrics[task]['unique_classes']} classes)")
        print(f"{'COMBINED':12}: {metrics['combined_accuracy']:.4f}")
        
        # Print detailed reports for each task
        for task in ['weather', 'scene', 'timeofday']:
            print(f"\n{task.upper()} CLASSIFICATION DETAILS:")
            print("-" * 50)
            
            report_df = pd.DataFrame(reports[task]).transpose()
            # Remove support column for cleaner display and handle string rows
            if 'precision' in report_df.columns:
                report_display = report_df[['precision', 'recall', 'f1-score']].round(4)
                print(report_display)
            else:
                print("No detailed metrics available (possibly only one class)")
    
    def analyze_misclassifications(self, predictions, targets, num_examples=5):
        """Analyze specific misclassification examples"""
        print(f"\nANALYZING TOP {num_examples} MISCLASSIFICATIONS PER TASK:")
        print("=" * 60)
        
        for task in ['weather', 'scene', 'timeofday']:
            if task == 'weather':
                categories = self.weather_categories
            elif task == 'scene':
                categories = self.scene_categories
            else:
                categories = self.timeofday_categories
            
            # Find misclassified indices
            misclassified_idx = np.where(np.array(predictions[task]) != np.array(targets[task]))[0]
            
            print(f"\n{task.upper()} - {len(misclassified_idx)} misclassifications found:")
            print("-" * 50)
            
            for i, idx in enumerate(misclassified_idx[:num_examples]):
                true_label = categories[targets[task][idx]]
                pred_label = categories[predictions[task][idx]]
                print(f"Example {i+1}: True: {true_label:15} -> Predicted: {pred_label:15}")
            
            if len(misclassified_idx) == 0:
                print("No misclassifications found!")

def load_model(model_path, device, image_size=72, patch_size=6):
    """Load the trained model"""
    model = VisionTransformerMultiTask(
        image_size=image_size,
        patch_size=patch_size
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    return model

def evaluate_single_image(model, image_path, transform, device, weather_categories, scene_categories, timeofday_categories):
    """Evaluate a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Convert to probabilities and get predictions
    weather_probs = torch.softmax(outputs['weather'], dim=1)[0]
    scene_probs = torch.softmax(outputs['scene'], dim=1)[0]
    timeofday_probs = torch.softmax(outputs['timeofday'], dim=1)[0]
    
    weather_pred = torch.argmax(outputs['weather'], dim=1).item()
    scene_pred = torch.argmax(outputs['scene'], dim=1).item()
    timeofday_pred = torch.argmax(outputs['timeofday'], dim=1).item()
    
    print(f"\nPREDICTIONS FOR: {os.path.basename(image_path)}")
    print("=" * 50)
    
    print(f"\nWEATHER:")
    for i, category in enumerate(weather_categories):
        print(f"  {category:15}: {weather_probs[i].item():.4f}")
    print(f"  → Prediction: {weather_categories[weather_pred]}")
    
    print(f"\nSCENE:")
    for i, category in enumerate(scene_categories):
        print(f"  {category:15}: {scene_probs[i].item():.4f}")
    print(f"  → Prediction: {scene_categories[scene_pred]}")
    
    print(f"\nTIME OF DAY:")
    for i, category in enumerate(timeofday_categories):
        print(f"  {category:15}: {timeofday_probs[i].item():.4f}")
    print(f"  → Prediction: {timeofday_categories[timeofday_pred]}")
    
    return {
        'weather': (weather_pred, weather_probs.cpu().numpy()),
        'scene': (scene_pred, scene_probs.cpu().numpy()),
        'timeofday': (timeofday_pred, timeofday_probs.cpu().numpy())
    }

def analyze_dataset_classes(dataset):
    """Analyze which classes are present in the dataset"""
    print("\nDATASET CLASS ANALYSIS:")
    print("=" * 40)
    
    weather_labels = []
    scene_labels = []
    timeofday_labels = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        weather_labels.append(item['weather'].item())
        scene_labels.append(item['scene'].item())
        timeofday_labels.append(item['timeofday'].item())
    
    print(f"Weather - Unique classes: {np.unique(weather_labels)}")
    print(f"Scene - Unique classes: {np.unique(scene_labels)}")
    print(f"TimeOfDay - Unique classes: {np.unique(timeofday_labels)}")
    
    return weather_labels, scene_labels, timeofday_labels

def main():
    # Configuration
    MODEL_PATH = "vision_transformer_multitask.pth"  # Update this path
    JSON_FILE = "/home/gwm-279/Desktop/multitask_classification/split_dataset/test_annotations.json"
    IMAGE_DIR = "/home/gwm-279/Desktop/multitask_classification/split_dataset/test"
    BATCH_SIZE = 64
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    
    model = load_model(MODEL_PATH, device)
    
    # Data transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    try:
        with open(JSON_FILE, 'r') as f:
            annotations = json.load(f)
        
        # Use a subset for evaluation (e.g., last 20%)
        test_size = len(annotations) // 5
        test_annotations = annotations[-test_size:]
        
        test_dataset = CustomDataset(test_annotations, IMAGE_DIR, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Analyze dataset classes
        analyze_dataset_classes(test_dataset)
        
    except FileNotFoundError:
        print(f"Error: JSON file '{JSON_FILE}' not found.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, weather_categories, scene_categories, timeofday_categories)
    
    # Perform evaluation
    print("Starting evaluation...")
    predictions, targets, probabilities = evaluator.evaluate(test_loader)
    
    # Generate reports
    reports = evaluator.generate_classification_report(predictions, targets)
    metrics = evaluator.calculate_overall_metrics(predictions, targets)
    
    # Print detailed report
    evaluator.print_detailed_report(reports, metrics)
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices(predictions, targets, save_path='confusion_matrices.png')
    
    # Analyze misclassifications
    evaluator.analyze_misclassifications(predictions, targets)
    

if __name__ == "__main__":
    main()