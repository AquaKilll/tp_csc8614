import sys
import torch
import transformers
import sklearn
import platform

print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")