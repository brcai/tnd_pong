Topological Neural Dynamics (TND) is a sequence modeling framework that shifts neural computation from **layer-wise dynamics** to **neuron-wise dynamics**.
In this repository, we provide a PyTorch implementation of TND together with a behavior cloning benchmark on a single-player Pong environment.

## Installation

Clone the repository

```bash
git clone https://github.com/brcai/tnd_pong.git
cd tnd_pong
```

## Dataset

The training data consists of human demonstrations collected from a single-player Pong environment.

Each observation contains

- Ball position
- Paddle position

and the target action is one of

- Left
- Right
- Stay

The model is trained using behavior cloning.

## Training

Train TND

```bash
python train_tnd.py
```

Evaluate

```bash
python test_tnd.py
```
