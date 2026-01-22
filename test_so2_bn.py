
import torch
import escnn.gspaces as gspaces
import escnn.nn as enn

def test_so2_bn():
    print("Testing InnerBatchNorm with SO(2)...")
    try:
        gspace = gspaces.rot2dOnR2(N=-1, maximum_frequency=4)
        ft = enn.FieldType(gspace, [gspace.irrep(0), gspace.irrep(1)])
        bn = enn.InnerBatchNorm(ft)
        print("InnerBatchNorm created successfully!")
        
        x = enn.GeometricTensor(torch.randn(2, 2, 16, 16), ft)
        y = bn(x)
        print(f"Forward pass successful. Output shape: {y.tensor.shape}")
        
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_so2_bn()
