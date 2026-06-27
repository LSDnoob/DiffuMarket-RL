import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# VERIFY AND ROUTE COMPUTATIONS TO NVIDIA GPU VIA CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- [GPU Verification Engine] ---")
print(f"Active Compute Target: {device.type.upper()}")
if device.type == "cuda":
    print(f"GPU Hardware Model   : {torch.cuda.get_device_name(0)}")
print(f"---------------------------------\n")
# device = torch.device("cpu")


#  DEFINE THE MICROSTRUCTURE CNN TOPOLOGY

class MarketMicrostructureCNN(nn.Module):
     def __init__(self):
        super(MarketMicrostructureCNN, self).__init__()

        # Input tensor shape: (Batch, Channels=3, Price_Levels=20, Time_Steps=50)
        # We treat the order book grid as a 2D image matrix
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduces spatial grid resolution to (10, 25)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces spatial grid resolution to (5, 6)
        )

        # Dense linear classifier network mapping features to directional probabilities
        self.classifier = nn.Sequential(
            nn.Linear(32 * 5 * 6 * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 3) # 3 Distinct Outputs: [0: Price Down, 1: Neutral, 2: Price Up]
        )
    
     def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1) # Flatten tensor map for linear layers
        return self.classifier(x)
     

# SGENERATE SYNTHETIC ORDER BOOK SAMPLES

def execute_gpu_training_simulation():
    
    # Simulating a dataset of 1,000 recorded order book tensor matrices
    print("Synthesizing market microstructure matrices...")
    X_sample = np.random.randn(1000, 3, 20, 50).astype(np.float32)
    y_sample = np.random.randint(0, 3, size=(1000,)).astype(np.int64)

    # Convert Numpy arrays to PyTorch Tensors
    X_tensor = torch.from_numpy(X_sample)
    y_tensor = torch.from_numpy(y_sample)

    #  Instantiate the network and push its weights onto the GPU VRAM
    model = MarketMicrostructureCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Beginning tensor optimization routine across GPU CUDA cores...")
    model.train()

     # Run a fast 5-epoch training sequence to optimize weights
    for epoch in range(1, 6):
        # Push variables onto the GPU memory
        inputs, targets = X_tensor.to(device), y_tensor.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch}/5] -> Matrix Optimization Loss: {loss.item():.5f}")

    # EXPORT OPTIMIZED WEIGHTS TO PORTABLE ONNX FORMAT
    # ONNX converts PyTorch structures into raw mathematical C++ primitives
    print("\nTraining complete. Exporting model weights to ONNX format...")
    model.eval()

     # Create a dummy tensor match matching our shape for the export tracer
    dummy_input = torch.randn(1, 3, 20, 50).to(device)
    onnx_filename = "microstructure_cnn.onnx"

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename, 
        export_params=True, 
        opset_version=11, 
        do_constant_folding=True,
        input_names=['input_tensor'], 
        output_names=['prediction_output']
    )
    print(f"Success! Model compiled and saved as: '{onnx_filename}'")

if __name__ == "__main__":
    execute_gpu_training_simulation()