import torch
from torch import nn
import torch.nn.init as init
import random
import torch.nn.functional as F

seed = 0

torch.manual_seed(seed)
random.seed(seed) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TND(nn.Module):
    def __init__(self, num_neurons=50, num_inputs=2, num_outputs=2, dt=0.2, connection_prob=0.8, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.device = device
        self.dt = dt

        self.input_indices = list(range(num_inputs))
        self.output_indices = list(range(num_inputs, num_inputs + num_outputs))
        
        adj_mask = self._build_vectorized_connections(connection_prob)
        self.register_buffer('adj_mask', adj_mask)

        self.weights = nn.Parameter(torch.empty(num_neurons, num_neurons))
        nn.init.xavier_uniform_(self.weights)
        self._apply_spectral_normalization(target_radius=0.95)
        
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        self.alpha = nn.Parameter(torch.full((num_neurons,), 0.5))
        self.input_weight = nn.Parameter(torch.ones(num_inputs))

    def reset_state(self):
        pass

    def _get_stable_weights(self):
        w = self.weights * self.adj_mask
        norm = torch.norm(w)
        target_norm = 0.95 * (self.num_neurons ** 0.5)
        if norm > target_norm:
            w = w * (target_norm / norm)
        return w

    def _apply_spectral_normalization(self, target_radius=0.95):
        with torch.no_grad():
            effective_w = self.weights * self.adj_mask
            
            eigs = torch.linalg.eigvals(effective_w.cpu())
            max_eig = torch.max(torch.abs(eigs))
            
            if max_eig > 0:
                self.weights.data = (self.weights.data / max_eig) * target_radius

    def _build_vectorized_connections(self, prob):
        mask = torch.zeros(self.num_neurons, self.num_neurons)
        
        hidden_start = self.num_inputs + self.num_outputs
        for i in self.output_indices:
            row_mask = (torch.rand(self.num_neurons - hidden_start) < prob).float()
            mask[i, hidden_start:] = row_mask

        for i in range(hidden_start, self.num_neurons):
            possible_srcs = [j for j in range(self.num_neurons) if j not in self.output_indices]
            for j in possible_srcs:
                if i != j and torch.rand(1) < prob:
                    mask[i, j] = 1

        target_nodes = list(range(self.num_inputs, self.num_neurons))
        for i in target_nodes:
            if mask[i, :].sum() == 0:
                possible_sources = [j for j in range(self.num_neurons) if j != i and j not in self.output_indices]
                if possible_sources:
                    mask[i, random.choice(possible_sources)] = 1
        
        hidden_indices = list(range(hidden_start, self.num_neurons))
        for j in hidden_indices:
            if mask[:, j].sum() == 0:
                possible_targets = [i for i in range(self.num_inputs, self.num_neurons) if i != j]
                if possible_targets:
                    mask[random.choice(possible_targets), j] = 1
        return mask

    def forward(self, input_seq, states=None, last_neuron_outputs=None):
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        if states is None:
            states = torch.zeros(batch_size, self.num_neurons, device=device)
        if last_neuron_outputs is None:
            last_neuron_outputs = torch.zeros(batch_size, self.num_neurons, device=device)

        effective_weights = self.weights * self.adj_mask
        
        full_input_stream = torch.zeros(batch_size, seq_len, self.num_neurons, device=device)
        full_input_stream[:, :, self.input_indices] = input_seq * self.input_weight

        w_t = effective_weights.t()

        for t in range(seq_len):
            weighted_sum = torch.matmul(last_neuron_outputs, w_t)
            total_input = full_input_stream[:, t, :] + weighted_sum + self.bias + self.alpha * states
            derivative = torch.tanh(total_input) - states

            states = states + self.dt * derivative
            last_neuron_outputs = torch.tanh(total_input)

        outputs = last_neuron_outputs[:, self.output_indices]
        
        return outputs, states, last_neuron_outputs

    def pred(self, input_seq, states=None, last_neuron_outputs=None):
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        if states is None:
            states = torch.zeros(batch_size, self.num_neurons, device=device)
        if last_neuron_outputs is None:
            last_neuron_outputs = torch.zeros(batch_size, self.num_neurons, device=device)

        effective_weights = self.weights * self.adj_mask
        
        full_input_stream = torch.zeros(batch_size, seq_len, self.num_neurons, device=device)
        full_input_stream[:, :, self.input_indices] = input_seq * self.input_weight

        w_t = effective_weights.t()

        for t in range(seq_len):
            weighted_sum = torch.matmul(last_neuron_outputs, w_t)
            total_input = full_input_stream[:, t, :] + weighted_sum + self.bias + self.alpha * states
            derivative = torch.tanh(total_input) - states

            states = states + self.dt * derivative
            last_neuron_outputs = torch.tanh(total_input)

        outputs = last_neuron_outputs[:, self.output_indices]
        
        return outputs, states, last_neuron_outputs
