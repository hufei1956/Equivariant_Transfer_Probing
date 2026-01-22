import sys
try:
    print("Importing torch...")
    import torch
    print(f"Torch version: {torch.__version__}")
    
    print("Importing escnn...")
    import escnn
    print(f"Escnn version: {escnn.__version__}")
    
    import escnn.nn as enn
    print("Escnn.nn imported.")
    
    print("Checking NormBatchNorm...")
    if hasattr(enn, 'NormBatchNorm'):
        print("YES: NormBatchNorm found")
    else:
        print("NO: NormBatchNorm not found")
        
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
