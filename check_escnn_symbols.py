import escnn.nn as enn
import sys

print("Searching for NormBatchNorm in escnn.nn...")
try:
    if hasattr(enn, 'NormBatchNorm'):
        print("Found NormBatchNorm")
    else:
        print("NormBatchNorm NOT found in enn")
        # Let's list what we have that looks like BatchNorm
        print("Available BatchNorms:", [x for x in dir(enn) if 'BatchNorm' in x])

    if hasattr(enn, 'NormMaxPool'):
        print("Found NormMaxPool")
    else:
        print("NormMaxPool NOT found in enn")
        print("Available MaxPools:", [x for x in dir(enn) if 'MaxPool' in x])

except Exception as e:
    print(f"Error: {e}")
