
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from escnn_train_v2 import build_equivariant_resnet18

def test():
    print("Testing model construction for SO2...")
    try:
        model = build_equivariant_resnet18(num_classes=6, group_type="SO2", maximum_frequency=4, base_channels=16)
        print("Model built successfully!")
        
        # Test forward pass
        print("Testing forward pass...")
        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(2, 3, 224, 224).cuda()
        else:
            x = torch.randn(2, 3, 224, 224)
            
        y = model(x)
        print(f"Forward pass successful. Output shape: {y.shape}")
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
