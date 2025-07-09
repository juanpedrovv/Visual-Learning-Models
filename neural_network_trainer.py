import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import json
import os
from datetime import datetime
import pandas as pd
import pickle
import time
from tqdm import tqdm
import sys

class NeuralNetworkModel(nn.Module):
    """CNN model similar to the TensorFlow version"""
    def __init__(self, input_channels, num_classes):
        super(NeuralNetworkModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate the size for the first linear layer
        # For MNIST (28x28) and CIFAR-10 (32x32), after convolutions and pooling
        self.flatten = nn.Flatten()
        
        # Dense layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Will be adjusted dynamically
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Store layer outputs for visualization
        self.layer_outputs = {}
    
    def forward(self, x):
        # Store intermediate outputs for visualization
        self.layer_outputs = {}
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        self.layer_outputs['conv1'] = x.clone()
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        self.layer_outputs['conv2'] = x.clone()
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        self.layer_outputs['conv3'] = x.clone()
        
        # Flatten
        x = self.flatten(x)
        
        # Adjust fc1 input size dynamically
        if not hasattr(self, '_fc1_input_size'):
            self._fc1_input_size = x.shape[1]
            self.fc1 = nn.Linear(self._fc1_input_size, 128)
            if x.is_cuda:
                self.fc1 = self.fc1.cuda()
        
        # Dense layers
        x = F.relu(self.fc1(x))
        self.layer_outputs['dense1'] = x.clone()
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        self.layer_outputs['dense2'] = x.clone()
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_layer_output(self, layer_name):
        """Get the output of a specific layer"""
        return self.layer_outputs.get(layer_name, None)

class NeuralNetworkVisualizer:
    def __init__(self, output_dir="visualization_data"):
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(output_dir, exist_ok=True)
        
    def load_dataset(self, dataset_name):
        """Load and preprocess datasets"""
        if dataset_name == 'mnist':
            print(f"   📥 Loading MNIST dataset...")
            
            # Check if data already exists
            data_path = './data/MNIST'
            if os.path.exists(data_path):
                print(f"   🎯 MNIST data found locally, loading...")
            else:
                print(f"   📥 MNIST data not found, downloading...")
                print(f"   ⏳ This may take a few minutes for first download...")
            
            start_time = time.time()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            print(f"   🔄 Creating MNIST train dataset...")
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            print(f"   🔄 Creating MNIST test dataset...")
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            input_channels = 1
            num_classes = 10
            
            load_time = time.time() - start_time
            print(f"   ✅ MNIST loaded in {load_time:.2f}s - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
        elif dataset_name == 'cifar10':
            print(f"   📥 Loading CIFAR-10 dataset...")
            
            # Check if data already exists
            data_path = './data/cifar-10-batches-py'
            if os.path.exists(data_path):
                print(f"   🎯 CIFAR-10 data found locally, loading...")
            else:
                print(f"   📥 CIFAR-10 data not found, downloading (~170MB)...")
                print(f"   ⏳ This may take 5-10 minutes depending on internet speed...")
            
            start_time = time.time()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            print(f"   🔄 Creating CIFAR-10 train dataset...")
            train_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform
            )
            
            print(f"   🔄 Creating CIFAR-10 test dataset...")
            test_dataset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform
            )
            
            input_channels = 3
            num_classes = 10
            
            load_time = time.time() - start_time
            print(f"   ✅ CIFAR-10 loaded in {load_time:.2f}s - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders
        print(f"   🔄 Creating data loaders with batch size 128...")
        start_time = time.time()
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        loader_time = time.time() - start_time
        print(f"   ✅ Data loaders created in {loader_time:.2f}s - Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, test_loader, input_channels, num_classes
    
    def apply_dimensionality_reduction(self, activations, method='tsne'):
        """Apply t-SNE or UMAP for dimensionality reduction"""
        print(f"     🔄 Applying {method.upper()} to {activations.shape[0]} samples with {activations.shape[1]} features...")
        
        start_time = time.time()
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, verbose=0)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, verbose=False)
        
        result = reducer.fit_transform(activations)
        
        end_time = time.time()
        print(f"     ✅ {method.upper()} completed in {end_time - start_time:.2f}s")
        
        return result
    
    def extract_activations(self, model, data_loader, layer_names, sample_size=1000):
        """Extract activations from specified layers"""
        model.eval()
        activations_data = {layer: [] for layer in layer_names}
        labels_data = []
        
        print(f"   🔬 Extracting activations from {len(layer_names)} layers...")
        
        with torch.no_grad():
            total_samples = 0
            extraction_pbar = tqdm(data_loader, desc="Extracting batches", unit="batch", leave=False)
            
            for batch_idx, (data, labels) in enumerate(extraction_pbar):
                if total_samples >= sample_size:
                    break
                
                data = data.to(self.device)
                
                # Forward pass
                _ = model(data)
                
                # Extract activations
                for layer_name in layer_names:
                    layer_output = model.get_layer_output(layer_name)
                    if layer_output is not None:
                        # Flatten the output for visualization
                        flattened = layer_output.view(layer_output.size(0), -1)
                        activations_data[layer_name].append(flattened.cpu().numpy())
                
                labels_data.append(labels.numpy())
                total_samples += data.size(0)
                
                # Update progress bar
                extraction_pbar.set_postfix({
                    'Samples': f'{total_samples}/{sample_size}',
                    'Progress': f'{min(100, total_samples/sample_size*100):.1f}%'
                })
        
        # Concatenate all batches
        print(f"   📊 Concatenating activations...")
        final_activations = {}
        for layer_name in layer_names:
            if activations_data[layer_name]:
                final_activations[layer_name] = np.concatenate(activations_data[layer_name])[:sample_size]
        
        final_labels = np.concatenate(labels_data)[:sample_size]
        
        print(f"   ✅ Extracted {len(final_labels)} samples from {len(final_activations)} layers")
        
        return final_activations, final_labels
    
    def train_and_extract(self, dataset_name, epochs=20):
        """Train model and extract activations"""
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING ON {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Load data
        print("📊 Loading dataset...")
        train_loader, test_loader, input_channels, num_classes = self.load_dataset(dataset_name)
        
        # Create model
        print("🧠 Creating neural network model...")
        print(f"   🔄 Initializing model architecture...")
        start_time = time.time()
        
        model = NeuralNetworkModel(input_channels, num_classes)
        model_time = time.time() - start_time
        print(f"   ✅ Model created in {model_time:.2f}s")
        
        # Move to GPU (this can take time on first run)
        print(f"   🚀 Moving model to {self.device}...")
        if self.device.type == 'cuda':
            print(f"   ⏳ First GPU initialization may take 1-3 minutes...")
            
        gpu_start = time.time()
        model = model.to(self.device)
        gpu_time = time.time() - gpu_start
        
        if self.device.type == 'cuda':
            print(f"   ✅ Model moved to GPU in {gpu_time:.2f}s")
            print(f"   💾 GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        else:
            print(f"   ✅ Model ready on CPU in {gpu_time:.2f}s")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Define optimizer and loss function
        print(f"⚙️ Setting up training components...")
        print(f"   🔄 Creating Adam optimizer...")
        start_time = time.time()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        if self.device.type == 'cuda':
            criterion = criterion.to(self.device)
            
        setup_time = time.time() - start_time
        print(f"   ✅ Optimizer and loss function ready in {setup_time:.2f}s")
        
        # Define layers to extract activations from
        layer_names = ['conv1', 'conv2', 'conv3', 'dense1', 'dense2']
        print(f"   📊 Monitoring {len(layer_names)} layers: {layer_names}")
        
        # Define epochs to extract activations
        extract_epochs = [0, 2, 5, 10, 15, epochs-1]
        print(f"   ⏰ Will extract activations at epochs: {extract_epochs}")
        
        # Storage for activations
        all_activations_data = []
        
        # Training metrics
        train_losses = []
        train_accuracies = []
        
        print(f"\n🏋️ Starting training on {self.device}")
        print(f"🎯 Training for {epochs} epochs with {len(train_loader)} batches per epoch")
        
        # Test GPU with first batch to warm up
        if self.device.type == 'cuda':
            print(f"🔥 Warming up GPU with first batch...")
            model.train()
            warmup_start = time.time()
            
            try:
                for batch_idx, (data, labels) in enumerate(train_loader):
                    data, labels = data.to(self.device), labels.to(self.device)
                    _ = model(data)
                    break  # Only do one batch for warmup
                    
                warmup_time = time.time() - warmup_start
                print(f"   ✅ GPU warmed up in {warmup_time:.2f}s")
                print(f"   💾 GPU Memory after warmup: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            except Exception as e:
                print(f"   ⚠️ GPU warmup failed: {e}")
        
        # Training loop with progress bar
        epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Batch progress bar
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                             unit="batch", leave=False)
            
            for batch_idx, (data, labels) in enumerate(batch_pbar):
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%',
                    'GPU': f'{torch.cuda.memory_allocated()/1024**2:.0f}MB' if torch.cuda.is_available() else 'CPU'
                })
            
            # Calculate metrics
            avg_loss = train_loss / len(train_loader)
            accuracy = 100. * correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            # Extract activations at specified epochs
            if epoch in extract_epochs:
                print(f"\n🔍 Extracting activations for epoch {epoch}...")
                
                # Progress bar for activation extraction
                activation_pbar = tqdm(layer_names, desc="Extracting layers", unit="layer")
                
                activations, labels = self.extract_activations(model, test_loader, layer_names)
                
                for layer_name in activation_pbar:
                    if layer_name in activations:
                        all_activations_data.append({
                            'epoch': epoch,
                            'layer': layer_name,
                            'activations': activations[layer_name],
                            'labels': labels,
                            'indices': np.arange(len(labels))
                        })
                        
                    activation_pbar.set_postfix({'Layer': layer_name})
                
                print(f"✅ Activations extracted for {len(layer_names)} layers")
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'Best Acc': f'{max(train_accuracies):.2f}%'
            })
        
        print(f"\n🎉 Training completed!")
        print(f"📊 Final metrics:")
        print(f"   • Loss: {train_losses[-1]:.4f}")
        print(f"   • Accuracy: {train_accuracies[-1]:.2f}%")
        print(f"   • Best Accuracy: {max(train_accuracies):.2f}%")
        print(f"   • Total activations extracted: {len(all_activations_data)}")
        
        # Process and save visualization data
        print(f"\n💾 Processing visualization data...")
        self.process_visualization_data(all_activations_data, dataset_name)
        
        return model, all_activations_data
    
    def process_visualization_data(self, activations_data, dataset_name):
        """Process activations and create visualization data"""
        visualization_data = {
            'dataset': dataset_name,
            'epochs': [],
            'layers': [],
            'projections': {}
        }
        
        # Group data by epoch and layer
        print(f"🔬 Processing {len(activations_data)} activation datasets...")
        
        # Progress bar for processing activations
        process_pbar = tqdm(activations_data, desc="Processing activations", unit="dataset")
        
        for data in process_pbar:
            epoch = data['epoch']
            layer = data['layer']
            activations = data['activations']
            labels = data['labels']
            
            # Update progress bar
            process_pbar.set_postfix({
                'Epoch': epoch,
                'Layer': layer,
                'Samples': len(labels)
            })
            
            # Apply dimensionality reduction
            print(f"\n📉 Applying dimensionality reduction for epoch {epoch}, layer {layer}...")
            
            # t-SNE with timing
            start_time = time.time()
            tsne_projection = self.apply_dimensionality_reduction(activations, 'tsne')
            tsne_time = time.time() - start_time
            
            # UMAP with timing
            start_time = time.time()
            umap_projection = self.apply_dimensionality_reduction(activations, 'umap')
            umap_time = time.time() - start_time
            
            print(f"   ✅ t-SNE: {tsne_time:.2f}s | UMAP: {umap_time:.2f}s")
            
            # Create unique key for this epoch-layer combination
            key = f"epoch_{epoch}_layer_{layer}"
            
            visualization_data['projections'][key] = {
                'epoch': epoch,
                'layer': layer,
                'tsne_coords': tsne_projection.tolist(),
                'umap_coords': umap_projection.tolist(),
                'labels': labels.tolist(),
                'n_samples': len(labels)
            }
            
            if epoch not in visualization_data['epochs']:
                visualization_data['epochs'].append(epoch)
            if layer not in visualization_data['layers']:
                visualization_data['layers'].append(layer)
        
        # Sort epochs and layers
        visualization_data['epochs'] = sorted(visualization_data['epochs'])
        visualization_data['layers'] = sorted(visualization_data['layers'])
        
        # Save to JSON file
        print(f"\n💾 Saving visualization data...")
        output_file = os.path.join(self.output_dir, f"{dataset_name}_visualization_data.json")
        
        with open(output_file, 'w') as f:
            json.dump(visualization_data, f, indent=2)
        
        file_size = os.path.getsize(output_file) / 1024 / 1024  # MB
        print(f"   ✅ Saved {output_file} ({file_size:.2f} MB)")
        
        # Also save a summary
        summary = {
            'dataset': dataset_name,
            'epochs': visualization_data['epochs'],
            'layers': visualization_data['layers'],
            'total_projections': len(visualization_data['projections']),
            'file_size_mb': round(file_size, 2),
            'generated_at': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.output_dir, f"{dataset_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Summary saved to {summary_file}")
        
        return visualization_data

def main():
    """Main function to run the neural network training and visualization data generation"""
    print("\n" + "="*80)
    print("🧠 NEURAL NETWORK LEARNING EVOLUTION VISUALIZATION")
    print("="*80)
    
    print("🔧 Initializing system...")
    init_start = time.time()
    
    print("   🔄 Creating visualizer instance...")
    visualizer = NeuralNetworkVisualizer()
    
    init_time = time.time() - init_start
    print(f"   ✅ System initialized in {init_time:.2f}s")
    
    # System information
    print(f"\n🖥️  System Information:")
    print(f"   • Device: {visualizer.device}")
    if torch.cuda.is_available():
        print(f"   • GPU: {torch.cuda.get_device_name(0)}")
        print(f"   • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   • CUDA Version: {torch.version.cuda}")
        print(f"   • GPU Driver: {torch.version.cuda}")
    print(f"   • PyTorch Version: {torch.__version__}")
    
    # Check data directory
    if os.path.exists('./data'):
        print(f"   📁 Data directory exists - checking for existing datasets...")
        for dataset in ['MNIST', 'cifar-10-batches-py']:
            path = f'./data/{dataset}'
            if os.path.exists(path):
                size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / 1024**2
                print(f"      ✅ {dataset} found ({size:.1f}MB)")
            else:
                print(f"      📥 {dataset} will be downloaded")
    else:
        print(f"   📁 Data directory will be created")
    
    # Train on multiple datasets
    datasets = ['mnist', 'cifar10']
    total_start_time = time.time()
    
    print(f"\n🚀 Starting training on {len(datasets)} datasets...")
    
    for i, dataset in enumerate(datasets):
        print(f"\n{'='*80}")
        print(f"📊 DATASET {i+1}/{len(datasets)}: {dataset.upper()}")
        print(f"{'='*80}")
        
        dataset_start_time = time.time()
        
        try:
            model, activations = visualizer.train_and_extract(dataset, epochs=20)
            
            dataset_time = time.time() - dataset_start_time
            print(f"\n✅ Successfully processed {dataset.upper()} in {dataset_time/60:.2f} minutes")
            
            # Show dataset summary
            print(f"📈 Dataset Summary:")
            print(f"   • Total activations: {len(activations)}")
            print(f"   • Unique epochs: {len(set(a['epoch'] for a in activations))}")
            print(f"   • Unique layers: {len(set(a['layer'] for a in activations))}")
            
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("🎉 ALL DATASETS PROCESSED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes")
    print(f"💾 Data saved to: {visualizer.output_dir}/")
    print(f"🌐 Ready for D3.js visualization!")
    print(f"{'='*80}")
    
    # Show next steps
    print(f"\n📋 Next Steps:")
    print(f"   1. Run: python server.py")
    print(f"   2. Open: http://localhost:5000")
    print(f"   3. Explore the neural network evolution!")
    print(f"\n   Or use: python run_project.py --only-server")

if __name__ == "__main__":
    main() 