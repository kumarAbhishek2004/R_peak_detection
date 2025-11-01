import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ==================== DATASET CLASS ====================
class ECGDataset(Dataset):
    """ECG Dataset with Distance Transform labels"""
    
    def __init__(self, ecg_signals, r_peak_locations, signal_length=5000):
        self.ecg_signals = ecg_signals
        self.r_peak_locations = r_peak_locations
        self.signal_length = signal_length
    
    def __len__(self):
        return len(self.ecg_signals)
    
    def __getitem__(self, idx):
        ecg = self.ecg_signals[idx]
        r_peaks = self.r_peak_locations[idx]
        
        # Create Distance Transform map
        dt_map = self.create_distance_transform(ecg, r_peaks)
        
        # Convert to torch tensors
        ecg_tensor = torch.FloatTensor(ecg).unsqueeze(0)  # (1, signal_length)
        dt_tensor = torch.FloatTensor(dt_map).unsqueeze(0)  # (1, signal_length)
        
        return ecg_tensor, dt_tensor
    
    def create_distance_transform(self, ecg, r_peaks):
        """Create Distance Transform map from R-peak locations"""
        # Create binary mask with R-peaks
        mask = np.zeros(len(ecg))
        for peak in r_peaks:
            if 0 <= peak < len(ecg):
                mask[int(peak)] = 1
        
        # Compute distance transform
        # Distance of each point to nearest R-peak
        if np.sum(mask) > 0:
            # Invert mask for distance transform (peaks are boundaries)
            inverted_mask = 1 - mask
            dt_map = distance_transform_edt(inverted_mask)
        else:
            dt_map = np.ones(len(ecg)) * len(ecg)
        
        # Normalize distance transform
        if np.max(dt_map) > 0:
            dt_map = dt_map / np.max(dt_map)
        
        return dt_map


# ==================== INCEPTION-RESIDUAL BLOCK ====================
class InceptionResBlock(nn.Module):
    """Inception-Residual Block as described in the paper"""
    
    def __init__(self, in_channels):
        super(InceptionResBlock, self).__init__()
        
        # 1x1 convolution for dimension reduction
        self.conv_1x1 = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1)
        
        # Inception branches with different kernel sizes (15, 17, 19, 21)
        self.branch1 = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=15, padding=7)
        self.branch2 = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=17, padding=8)
        self.branch3 = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=19, padding=9)
        self.branch4 = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=21, padding=10)
        
        # 1x1 convolution to combine branches
        self.conv_combine = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        # Store input for residual connection
        identity = x
        
        # 1x1 convolution
        out = self.conv_1x1(x)
        
        # Inception branches
        branch1 = self.branch1(out)
        branch2 = self.branch2(out)
        branch3 = self.branch3(out)
        branch4 = self.branch4(out)
        
        # Concatenate branches
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # Combine and normalize
        out = self.conv_combine(out)
        out = self.bn(out)
        
        # Residual connection
        out = out + identity
        out = self.relu(out)
        
        return out


# ==================== ENCODER BLOCK ====================
class EncoderBlock(nn.Module):
    """Encoder block with strided convolution, BatchNorm, LeakyReLU, and Inception-Res"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(EncoderBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.inception_res = InceptionResBlock(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.inception_res(x)
        return x


# ==================== DECODER BLOCK ====================
class DecoderBlock(nn.Module):
    """Decoder block with transpose convolution, BatchNorm, ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2):
        super(DecoderBlock, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, 
                                                  kernel_size=kernel_size, 
                                                  stride=stride, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x, skip_connection=None):
        x = self.conv_transpose(x)
        
        # Concatenate with skip connection if provided
        if skip_connection is not None:
            # Handle size mismatch
            if x.size(2) != skip_connection.size(2):
                diff = skip_connection.size(2) - x.size(2)
                x = F.pad(x, (diff // 2, diff - diff // 2))
            x = torch.cat([x, skip_connection], dim=1)
        
        x = self.bn(x)
        x = self.activation(x)
        return x


# ==================== INCRES-UNET MODEL ====================
class IncResUNet(nn.Module):
    """IncRes-UNet architecture for ECG R-peak detection via Distance Transform"""
    
    def __init__(self, in_channels=1, base_channels=64):
        super(IncResUNet, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2)
        )
        
        # Encoder (8 layers with downsampling)
        self.enc1 = EncoderBlock(base_channels, base_channels * 2)      # 64 -> 128
        self.enc2 = EncoderBlock(base_channels * 2, base_channels * 4)  # 128 -> 256
        self.enc3 = EncoderBlock(base_channels * 4, base_channels * 8)  # 256 -> 512
        self.enc4 = EncoderBlock(base_channels * 8, 1024)               # 512 -> 1024
        self.enc5 = EncoderBlock(1024, 1024)                            # 1024 -> 1024
        self.enc6 = EncoderBlock(1024, 1024)                            # 1024 -> 1024
        self.enc7 = EncoderBlock(1024, 1024)                            # 1024 -> 1024
        self.enc8 = EncoderBlock(1024, 1024)                            # 1024 -> 1024 (bottleneck)
        
        # Decoder (8 layers with upsampling and skip connections)
        self.dec1 = DecoderBlock(1024, 1024)                  # 1024 -> 1024
        self.dec2 = DecoderBlock(2048, 1024)                  # 2048 (with skip) -> 1024
        self.dec3 = DecoderBlock(2048, 1024)                  # 2048 (with skip) -> 1024
        self.dec4 = DecoderBlock(2048, 1024)                  # 2048 (with skip) -> 1024
        self.dec5 = DecoderBlock(2048, base_channels * 8)     # 2048 (with skip) -> 512
        self.dec6 = DecoderBlock(base_channels * 16, base_channels * 4)  # 1024 (with skip) -> 256
        self.dec7 = DecoderBlock(base_channels * 8, base_channels * 2)   # 512 (with skip) -> 128
        self.dec8 = DecoderBlock(base_channels * 4, base_channels)       # 256 (with skip) -> 64
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, 1, kernel_size=1)
        )
    
    def forward(self, x):
        # Initial convolution
        x0 = self.initial(x)
        
        # Encoder with skip connections
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)  # Bottleneck
        
        # Decoder with skip connections
        d1 = self.dec1(x8)
        d2 = self.dec2(d1, x7)
        d3 = self.dec3(d2, x6)
        d4 = self.dec4(d3, x5)
        d5 = self.dec5(d4, x4)
        d6 = self.dec6(d5, x3)
        d7 = self.dec7(d6, x2)
        d8 = self.dec8(d7, x1)
        
        # Final output
        out = self.final(torch.cat([d8, x0], dim=1))
        
        return out


# ==================== SMOOTH L1 LOSS ====================
class SmoothL1Loss(nn.Module):
    """SmoothL1 Loss as described in the paper"""
    
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
    
    def forward(self, pred, target):
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # SmoothL1: 0.5 * (diff)^2 if |diff| < 1, else |diff| - 0.5
        loss = torch.where(abs_diff < 1, 0.5 * diff ** 2, abs_diff - 0.5)
        
        return torch.mean(loss)


# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, num_epochs=500, device='cuda'):
    """Train the IncRes-UNet model"""
    
    model = model.to(device)
    criterion = SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    # Learning rate scheduler: decrease by factor of 10 every 150 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (ecg, dt_map) in enumerate(train_loader):
            ecg = ecg.to(device)
            dt_map = dt_map.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_dt = model(ecg)
            loss = criterion(pred_dt, dt_map)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for ecg, dt_map in val_loader:
                ecg = ecg.to(device)
                dt_map = dt_map.to(device)
                
                pred_dt = model(ecg)
                loss = criterion(pred_dt, dt_map)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_incres_unet.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, '
                  f'Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    return model, train_losses, val_losses


# ==================== POST-PROCESSING ====================
def extract_r_peaks_from_dt(distance_map, tolerance=75, sampling_rate=500):
    """Extract R-peak locations from Distance Transform map"""
    
    # Find valleys (minima) in distance map - these are R-peak locations
    from scipy.signal import find_peaks
    
    # Invert distance map to find peaks (valleys become peaks)
    inverted_dt = -distance_map
    
    # Minimum distance between peaks in samples
    min_distance = int(0.2 * sampling_rate)  # 200ms minimum
    
    # Find peaks with minimum distance constraint
    peaks, _ = find_peaks(inverted_dt, distance=min_distance)
    
    return peaks


def evaluate_predictions(pred_peaks, true_peaks, tolerance=75, sampling_rate=500):
    """Evaluate R-peak detection with tolerance window"""
    
    tolerance_samples = int(tolerance / 1000 * sampling_rate)  # Convert ms to samples
    
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    
    matched_true = set()
    matched_pred = set()
    
    # Match predicted peaks to true peaks
    for pred_idx, pred_peak in enumerate(pred_peaks):
        matched = False
        for true_idx, true_peak in enumerate(true_peaks):
            if true_idx not in matched_true:
                if abs(pred_peak - true_peak) <= tolerance_samples:
                    TP += 1
                    matched_true.add(true_idx)
                    matched_pred.add(pred_idx)
                    matched = True
                    break
        
        if not matched:
            FP += 1
    
    # Count false negatives
    FN = len(true_peaks) - len(matched_true)
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# ==================== DATA LOADING ====================
def load_mitbih_data(filepath, target_length=5000, sampling_rate=500):
    """Load and preprocess MIT-BIH dataset"""
    
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, header=None)
    
    # Assuming last column is label, rest are ECG samples
    ecg_signals = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    print(f"Loaded {len(ecg_signals)} ECG signals")
    print(f"Signal length: {ecg_signals.shape[1]}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Resample to target length if needed
    if ecg_signals.shape[1] != target_length:
        print(f"Resampling signals from {ecg_signals.shape[1]} to {target_length}...")
        ecg_resampled = []
        for ecg in ecg_signals:
            x_old = np.linspace(0, 1, len(ecg))
            x_new = np.linspace(0, 1, target_length)
            f = interp1d(x_old, ecg, kind='cubic')
            ecg_new = f(x_new)
            ecg_resampled.append(ecg_new)
        ecg_signals = np.array(ecg_resampled)
    
    return ecg_signals, labels


def detect_r_peaks_traditional(ecg_signal, sampling_rate=500):
    """Traditional R-peak detection for creating labels (Pan-Tompkins inspired)"""
    
    from scipy.signal import butter, filtfilt, find_peaks
    
    # Bandpass filter
    nyquist = sampling_rate / 2
    low = 5 / nyquist
    high = 15 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg_signal)
    
    # Differentiation
    diff = np.diff(filtered)
    diff = np.append(diff, 0)
    
    # Squaring
    squared = diff ** 2
    
    # Moving window integration
    window_size = int(0.15 * sampling_rate)
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # Find peaks
    min_distance = int(0.2 * sampling_rate)  # 200ms
    peaks, _ = find_peaks(integrated, distance=min_distance, 
                          prominence=np.std(integrated) * 0.5)
    
    return peaks


# ==================== VISUALIZATION ====================
def visualize_results(ecg, true_dt, pred_dt, true_peaks, pred_peaks):
    """Visualize ECG signal with true and predicted distance maps"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Original ECG with R-peaks
    axes[0].plot(ecg, 'b-', linewidth=0.8, label='ECG Signal')
    axes[0].plot(true_peaks, ecg[true_peaks], 'go', markersize=8, label='True R-peaks')
    axes[0].plot(pred_peaks, ecg[pred_peaks], 'r^', markersize=6, label='Predicted R-peaks')
    axes[0].set_title('ECG Signal with R-peaks', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # True Distance Transform
    axes[1].plot(true_dt, 'g-', linewidth=1.0)
    axes[1].set_title('True Distance Transform', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Distance')
    axes[1].grid(True, alpha=0.3)
    
    # Predicted Distance Transform
    axes[2].plot(pred_dt, 'r-', linewidth=1.0)
    axes[2].set_title('Predicted Distance Transform', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Distance')
    axes[2].set_xlabel('Sample Index')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rpnet_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function"""
    
    print("=" * 70)
    print("RPNet: Deep Learning R-Peak Detection using IncRes-UNet")
    print("Paper: A Deep Learning approach for robust R Peak detection")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Hyperparameters
    signal_length = 5000
    sampling_rate = 500
    batch_size = 32
    num_epochs = 500
    
    # Load dataset
    print("\n1. Loading MIT-BIH dataset...")
    filepath = 'mitbih_train.csv'  # Update with your file path
    ecg_signals, labels = load_mitbih_data(filepath, target_length=signal_length, 
                                           sampling_rate=sampling_rate)
    
    # Detect R-peaks for each signal (for creating ground truth)
    print("\n2. Detecting R-peaks for ground truth labels...")
    r_peak_locations = []
    for i, ecg in enumerate(ecg_signals):
        peaks = detect_r_peaks_traditional(ecg, sampling_rate=sampling_rate)
        r_peak_locations.append(peaks)
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i+1}/{len(ecg_signals)} signals")
    
    # Split dataset (96% train, 4% test as per paper)
    print("\n3. Splitting dataset...")
    train_ecg, test_ecg, train_peaks, test_peaks = train_test_split(
        ecg_signals, r_peak_locations, test_size=0.04, random_state=42
    )
    
    # Further split training into train and validation
    train_ecg, val_ecg, train_peaks, val_peaks = train_test_split(
        train_ecg, train_peaks, test_size=0.1, random_state=42
    )
    
    print(f"   Train: {len(train_ecg)}, Val: {len(val_ecg)}, Test: {len(test_ecg)}")
    
    # Create datasets and dataloaders
    print("\n4. Creating datasets and dataloaders...")
    train_dataset = ECGDataset(train_ecg, train_peaks, signal_length=signal_length)
    val_dataset = ECGDataset(val_ecg, val_peaks, signal_length=signal_length)
    test_dataset = ECGDataset(test_ecg, test_peaks, signal_length=signal_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Create model
    print("\n5. Creating IncRes-UNet model...")
    model = IncResUNet(in_channels=1, base_channels=64)
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\n6. Training model for {num_epochs} epochs...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs, device=device
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_incres_unet.pth'))
    model.eval()
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    all_precision = []
    all_recall = []
    all_f1 = []
    
    with torch.no_grad():
        for i, (ecg, true_dt) in enumerate(test_loader):
            ecg = ecg.to(device)
            pred_dt = model(ecg)
            
            # Convert to numpy
            ecg_np = ecg.cpu().numpy()[0, 0]
            true_dt_np = true_dt.numpy()[0, 0]
            pred_dt_np = pred_dt.cpu().numpy()[0, 0]
            
            # Extract R-peaks
            pred_peaks = extract_r_peaks_from_dt(pred_dt_np, sampling_rate=sampling_rate)
            true_peaks = test_peaks[i]
            
            # Evaluate
            precision, recall, f1 = evaluate_predictions(
                pred_peaks, true_peaks, tolerance=75, sampling_rate=sampling_rate
            )
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            
            # Visualize first test sample
            if i == 0:
                visualize_results(ecg_np, true_dt_np, pred_dt_np, true_peaks, pred_peaks)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Precision: {np.mean(all_precision):.4f} ± {np.std(all_precision):.4f}")
    print(f"Recall:    {np.mean(all_recall):.4f} ± {np.std(all_recall):.4f}")
    print(f"F1-Score:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print("=" * 70)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('SmoothL1 Loss')
    plt.title('Training History', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_history_rpnet.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nTraining completed successfully!")
    print("Model saved as 'best_incres_unet.pth'")


if __name__ == "__main__":
    main()