import torch
import numpy as np
from models.diffusion import DiffusionModel

data = np.load("data/processed/train.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
Y = torch.tensor(data["Y"][:,0,:], dtype=torch.float32)

model = DiffusionModel(X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for epoch in range(20):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
