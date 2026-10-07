import torch
import snntorch as snn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"snnTorch: {snn.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

df = pd.read_csv('data/exoTrain.csv')
print(f"\nShape del dataset: {df.shape}")
print(f"Distribución de etiquetas:")
print(df['LABEL'].value_counts())