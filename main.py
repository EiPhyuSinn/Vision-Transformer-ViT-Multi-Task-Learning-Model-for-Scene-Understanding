import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter  # ADD THIS
from torchvision import transforms
from PIL import Image
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define categories
weather_categories = ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined']
scene_categories = ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined']
timeofday_categories = ['dawn/dusk', 'daytime', 'night', 'undefined']

# ==================== DATA PREPARATION ====================
class CustomDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform
        
        # Initialize label encoders
        self.weather_encoder = LabelEncoder()
        self.scene_encoder = LabelEncoder()
        self.timeofday_encoder = LabelEncoder()
        
        # Fit encoders
        self.weather_encoder.fit(weather_categories)
        self.scene_encoder.fit(scene_categories)
        self.timeofday_encoder.fit(timeofday_categories)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = os.path.join(self.image_dir, item['name'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Encode labels
        weather_label = self.weather_encoder.transform([item['attributes']['weather']])[0]
        scene_label = self.scene_encoder.transform([item['attributes']['scene']])[0]
        timeofday_label = self.timeofday_encoder.transform([item['attributes']['timeofday']])[0]
        
        return {
            'image': image,
            'weather': torch.tensor(weather_label, dtype=torch.long),
            'scene': torch.tensor(scene_label, dtype=torch.long),
            'timeofday': torch.tensor(timeofday_label, dtype=torch.long)
        }

# ==================== MODEL ARCHITECTURE ====================
class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, images):
        batch_size, channels, height, width = images.shape
        
        # Rearrange to extract patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches = patches.view(batch_size, -1, self.patch_size * self.patch_size * channels)
        
        return patches

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(6*6*3, projection_dim)  # 6x6 patches with 3 channels
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        
    def forward(self, patch):
        positions = torch.arange(0, self.num_patches, device=patch.device)
        encoded_patch = self.projection(patch) + self.position_embedding(positions)
        return encoded_patch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.num_heads different pieces
        values = self.values(values).view(N, value_len, self.num_heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.num_heads, self.head_dim)
        queries = self.queries(query).view(N, query_len, self.num_heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Normalize energy values
        attention = torch.softmax(energy / (self.embed_dim ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class VisionTransformerMultiTask(nn.Module):
    def __init__(self, 
                 image_size=72,
                 patch_size=6,
                 num_layers=8,
                 embed_dim=64,
                 num_heads=4,
                 forward_expansion=4,
                 dropout=0.2,
                 mlp_dropout=0.2):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch extraction and encoding
        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(mlp_dropout)
        
        # Shared feature extraction
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.num_patches * embed_dim, 2048),
            nn.GELU(),
            nn.Dropout(mlp_dropout)
        )
        
        # Task-specific branches
        # Weather branch (7 classes)
        self.weather_branch = nn.Sequential(
            nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(128, len(weather_categories))
        )
        
        # Scene branch (7 classes)
        self.scene_branch = nn.Sequential(
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(128, len(scene_categories))
        )
        
        # Time of day branch (4 classes)
        self.timeofday_branch = nn.Sequential(
            nn.Linear(2048, 512), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(128, len(timeofday_categories))
        )
    
    def forward(self, x):
        # Extract and encode patches
        patches = self.patches(x)
        encoded_patches = self.patch_encoder(patches)
        
        # Transformer layers
        for layer in self.layers:
            encoded_patches = layer(encoded_patches, encoded_patches, encoded_patches)
        
        # Normalize and flatten
        representation = self.norm(encoded_patches)
        representation = representation.reshape(representation.shape[0], -1)
        representation = self.dropout(representation)
        
        # Shared features
        features = self.shared_mlp(representation)
        
        # Task-specific outputs
        weather_out = self.weather_branch(features)
        scene_out = self.scene_branch(features)
        timeofday_out = self.timeofday_branch(features)
        
        return {
            'weather': weather_out,
            'scene': scene_out,
            'timeofday': timeofday_out
        }

# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, num_epochs, device, checkpoint_dir='checkpoints'):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter('runs/vision_transformer_experiment')
    
    # Loss functions and optimizer
    criterion_weather = nn.CrossEntropyLoss()
    criterion_scene = nn.CrossEntropyLoss()
    criterion_timeofday = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accuracies = {'weather': [], 'scene': [], 'timeofday': []}
    val_accuracies = {'weather': [], 'scene': [], 'timeofday': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = {'weather': 0, 'scene': 0, 'timeofday': 0}
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            weather_labels = batch['weather'].to(device)
            scene_labels = batch['scene'].to(device)
            timeofday_labels = batch['timeofday'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            # Calculate losses with equal weighting
            loss_weather = criterion_weather(outputs['weather'], weather_labels)
            loss_scene = criterion_scene(outputs['scene'], scene_labels)
            loss_timeofday = criterion_timeofday(outputs['timeofday'], timeofday_labels)
            
            total_loss = loss_weather + loss_scene + loss_timeofday
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate training accuracy
            _, weather_pred = torch.max(outputs['weather'], 1)
            _, scene_pred = torch.max(outputs['scene'], 1)
            _, timeofday_pred = torch.max(outputs['timeofday'], 1)
            
            train_correct['weather'] += (weather_pred == weather_labels).sum().item()
            train_correct['scene'] += (scene_pred == scene_labels).sum().item()
            train_correct['timeofday'] += (timeofday_pred == timeofday_labels).sum().item()
            train_total += weather_labels.size(0)
            
            # Log batch loss to TensorBoard
            if batch_idx % 10 == 0:  # Log every 10 batches
                writer.add_scalar('Training Batch Loss', total_loss.item(), epoch * len(train_loader) + batch_idx)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = {'weather': 0, 'scene': 0, 'timeofday': 0}
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                weather_labels = batch['weather'].to(device)
                scene_labels = batch['scene'].to(device)
                timeofday_labels = batch['timeofday'].to(device)
                
                outputs = model(images)
                
                loss_weather = criterion_weather(outputs['weather'], weather_labels)
                loss_scene = criterion_scene(outputs['scene'], scene_labels)
                loss_timeofday = criterion_timeofday(outputs['timeofday'], timeofday_labels)
                
                total_loss = loss_weather + loss_scene + loss_timeofday
                val_loss += total_loss.item()
                
                # Calculate validation accuracy
                _, weather_pred = torch.max(outputs['weather'], 1)
                _, scene_pred = torch.max(outputs['scene'], 1)
                _, timeofday_pred = torch.max(outputs['timeofday'], 1)
                
                val_correct['weather'] += (weather_pred == weather_labels).sum().item()
                val_correct['scene'] += (scene_pred == scene_labels).sum().item()
                val_correct['timeofday'] += (timeofday_pred == timeofday_labels).sum().item()
                val_total += weather_labels.size(0)
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_acc_weather = 100 * train_correct['weather'] / train_total
        train_acc_scene = 100 * train_correct['scene'] / train_total
        train_acc_timeofday = 100 * train_correct['timeofday'] / train_total
        
        val_acc_weather = 100 * val_correct['weather'] / val_total
        val_acc_scene = 100 * val_correct['scene'] / val_total
        val_acc_timeofday = 100 * val_correct['timeofday'] / val_total
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies['weather'].append(train_acc_weather)
        train_accuracies['scene'].append(train_acc_scene)
        train_accuracies['timeofday'].append(train_acc_timeofday)
        val_accuracies['weather'].append(val_acc_weather)
        val_accuracies['scene'].append(val_acc_scene)
        val_accuracies['timeofday'].append(val_acc_timeofday)
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Weather_Train', train_acc_weather, epoch)
        writer.add_scalar('Accuracy/Weather_Val', val_acc_weather, epoch)
        writer.add_scalar('Accuracy/Scene_Train', train_acc_scene, epoch)
        writer.add_scalar('Accuracy/Scene_Val', val_acc_scene, epoch)
        writer.add_scalar('Accuracy/TimeOfDay_Train', train_acc_timeofday, epoch)
        writer.add_scalar('Accuracy/TimeOfDay_Val', val_acc_timeofday, epoch)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Acc - Weather: {train_acc_weather:.2f}%, Scene: {train_acc_scene:.2f}%, TimeOfDay: {train_acc_timeofday:.2f}%')
        print(f'Val Acc   - Weather: {val_acc_weather:.2f}%, Scene: {val_acc_scene:.2f}%, TimeOfDay: {val_acc_timeofday:.2f}%')
        print('-' * 60)
    
    # Close TensorBoard writer
    writer.close()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# ==================== PLOTTING FUNCTION ====================
def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot weather accuracy
    ax2.plot(train_accuracies['weather'], label='Train Weather Acc')
    ax2.plot(val_accuracies['weather'], label='Val Weather Acc')
    ax2.set_title('Weather Classification Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot scene accuracy
    ax3.plot(train_accuracies['scene'], label='Train Scene Acc')
    ax3.plot(val_accuracies['scene'], label='Val Scene Acc')
    ax3.set_title('Scene Classification Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot timeofday accuracy
    ax4.plot(train_accuracies['timeofday'], label='Train TimeOfDay Acc')
    ax4.plot(val_accuracies['timeofday'], label='Val TimeOfDay Acc')
    ax4.set_title('Time of Day Classification Accuracy')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== MAIN EXECUTION ====================
def main():
    # Configuration - UPDATED FOR PRE-SPLIT DATA
    TRAIN_JSON_FILE = "/home/gwm-279/Desktop/multitask_classification/split_dataset/train_annotations.json"
    TEST_JSON_FILE = "/home/gwm-279/Desktop/multitask_classification/split_dataset/test_annotations.json"
    
    TRAIN_IMAGE_DIR = "/home/gwm-279/Desktop/multitask_classification/split_dataset/train"
    TEST_IMAGE_DIR = "/home/gwm-279/Desktop/multitask_classification/split_dataset/test"
    
    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        with open(TRAIN_JSON_FILE, 'r') as f:
            train_annotations = json.load(f)
        with open(TEST_JSON_FILE, 'r') as f:
            test_annotations = json.load(f)
        print(f"Loaded {len(train_annotations)} train annotations")
        print(f"Loaded {len(test_annotations)} test annotations")
    except FileNotFoundError as e:
        print(f"Error: JSON file not found - {e}")
        return
    
    # Check if image directories exist
    if not os.path.exists(TRAIN_IMAGE_DIR):
        print(f"Error: Train image directory '{TRAIN_IMAGE_DIR}' not found.")
        return
    if not os.path.exists(TEST_IMAGE_DIR):
        print(f"Error: Test image directory '{TEST_IMAGE_DIR}' not found.")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((72, 72)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CustomDataset(train_annotations, TRAIN_IMAGE_DIR, transform=transform)
    test_dataset = CustomDataset(test_annotations, TEST_IMAGE_DIR, transform=transform)

    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = VisionTransformerMultiTask().to(device)
    print("Model initialized successfully!")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, NUM_EPOCHS, device
    )
    
    # Plot results
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save final model
    torch.save(model.state_dict(), 'vision_transformer_multitask.pth')
    print("Model saved as 'vision_transformer_multitask.pth'")
    
    print("Training completed! To view TensorBoard logs, run:")
    print("tensorboard --logdir=runs/")

if __name__ == "__main__":
    main()