import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import random
import argparse

from model.tnd import TND

seed = 0

torch.manual_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda idx.')
parser.add_argument('--l', type=int, default=40, help='number of time stamps.')
parser.add_argument('--bt', type=int, default=25, help='batch size')
parser.add_argument('--neuron', type=int, default=400, help='number neurons.')
parser.add_argument('--p', type=float, default=0.2, help='connectivity probability')
parser.add_argument('--dt', type=float, default=0.2, help='connectivity probability')
args = parser.parse_args()


l = args.l
batch_size = args.bt
device = args.cuda
n_neurons = args.neuron
p_connect = args.p
dt = args.dt

device = torch.device(device if torch.cuda.is_available() else "cpu")


x = []
y = []

print("loading data!!!")
fp = open("dt/c2.txt")

idx = 0
for line in fp.readlines():
    x_txt, y_txt = line.strip().split('\t')
    x.append([int(itm) for itm in x_txt.split(',')])
    y.append([int(itm) for itm in y_txt.split(',')])
    
    idx += 1
    #if idx > 1000: break

fp.close()


x_new = []
y_new = []


for j in range(0, len(x) - l, 1):
    x_new.append(x[j:j+l])
    y_new.append(y[j+l])


cnt = int(len(x_new)/(batch_size*l))

x_new_prune = x_new[:cnt*(batch_size*l)]
y_new_prune = y_new[:cnt*(batch_size*l)]

x_tensor = torch.tensor(x_new_prune, dtype=torch.float32)
y_tensor = torch.tensor(y_new_prune, dtype=torch.float32)

x_tensor = x_tensor.reshape(-1, l, *x_tensor.shape[1:]).transpose(0, 1)
y_tensor = y_tensor.reshape(-1, l, *y_tensor.shape[1:]).transpose(0, 1)

x_tensor = x_tensor.reshape(l*batch_size, -1, *x_tensor.shape[2:])
y_tensor = y_tensor.reshape(l*batch_size, -1, *y_tensor.shape[2:])

x_tensor = x_tensor.permute(1, 0, 2, 3)
y_tensor = y_tensor.permute(1, 0, 2)

print("finished preparing dataset!!!")

model = TND(num_neurons=n_neurons, num_inputs=24, num_outputs=3, dt=dt, connection_prob=p_connect, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))

total_params = sum(p.numel() for p in model.parameters())
print(f"snn total parameters: {total_params:,}")  # 

print("start training!!!")
num_epochs = 800
for epoch in range(num_epochs):
    running_loss = 0.0
    states = None
    neuron_outputs = None
    count = 0
    for idx in range(x_tensor.shape[0]):
        batch_x = x_tensor[idx,:].to(device)
        batch_y = y_tensor[idx,:].to(device)
        # Forward pass
        outputs, states, neuron_outputs = model(batch_x, states, neuron_outputs)
        #print(outputs)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #print("batch: ", loss.item())
        running_loss += loss.item()
        
        states = states.detach()
        neuron_outputs = neuron_outputs.detach()
        #print(count)
        count += 1

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
    if running_loss < 0.13:
        break

torch.save(model.state_dict(), "save/tnd_model_"+str(dt)+"_"+str(n_neurons)+"_"+str(p_connect)+"_"+str(batch_size)+"_"+str(l)+".pth")
print("Training finished, nyaa~! 🐾")
