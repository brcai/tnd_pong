# -*- coding: utf-8 -*-
import pygame
import random
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import random
import argparse


from model.tnd import TND 
from fun import int_to_binary_list, detect_paddle_cycle_final, accuracy_test


seed = 0

torch.manual_seed(seed)
random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='cuda idx.')
parser.add_argument('--l', type=int, default=40, help='number of time stamps.')
parser.add_argument('--s', type=int, default=0, help='seed')
parser.add_argument('--n', type=int, default=200, help='number neurons.')
parser.add_argument('--p', type=float, default=0.2, help='connectivity probability')
parser.add_argument('--dt', type=float, default=0.2, help='connectivity probability')
parser.add_argument('--ll', type=int, default=3, help='number layers')
args = parser.parse_args()


n_neurons = args.n
p_connect = args.p
device = args.cuda
l = args.l
ll = args.ll
batch_size = 25
dt = args.dt

device = torch.device(device if torch.cuda.is_available() else "cpu")


model = TND(num_neurons=n_neurons, num_inputs=24, num_outputs=3, dt=dt, connection_prob=p_connect, device=device).to(device)
model.load_state_dict(torch.load("save/tnd_model_"+str(dt)+"_"+str(n_neurons)+"_"+str(p_connect)+"_"+str(batch_size)+"_"+str(l)+".pth", map_location=device))
model.eval()

seed = 989
print("the random seed is: ", seed)
tmp_best_catch = 0
torch.manual_seed(seed)
random.seed(seed) 
states = None
neuron_outputs = None

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 300
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle settings
PADDLE_WIDTH = 50
PADDLE_HEIGHT = 10
PADDLE_SPEED = 7

# Ball settings
BALL_SIZE = 15
BALL_SPEED_X = 4
BALL_SPEED_Y = -4

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Paddle Ball Game")

# Initialize paddle
paddle = pygame.Rect(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)

# Initialize ball
ball = pygame.Rect(random.choice([i*10 for i in range(3,17)]), random.choice([i*10 for i in range(3,25)]), BALL_SIZE, BALL_SIZE)
ball_dx = BALL_SPEED_X
ball_dy = BALL_SPEED_Y

# Game loop
clock = pygame.time.Clock()

d = []

idx = 0

succ = 0
fail = 0

V = []

with torch.no_grad():
    trail = 0
    while idx < 10000:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        x = ball.x
        y = ball.y
        p_x = paddle.x

        V.append(p_x)
        p_y = paddle.y
        press = 0

        tmp = []
        tmp.extend(int_to_binary_list(int(x) + 2))
        tmp.extend(int_to_binary_list(int(y) + 2))
        tmp.extend(int_to_binary_list(int(p_x) + 2))
        x_input = torch.tensor(tmp, dtype=torch.float32).reshape(1,-1,24).to(device)
    
    
        pred, s, o = model.pred(x_input, states, neuron_outputs)
        states = s
        neuron_outputs = o
        ctr = torch.argmax(pred).item()
        if ctr == 0:
            press = 1
        elif ctr == 1:
            press = -1

        if press == -1 and paddle.left > 0:
            paddle.x -= PADDLE_SPEED
        if press == 1 and paddle.right < SCREEN_WIDTH:
            paddle.x += PADDLE_SPEED

        # Ball movement
        ball.x += ball_dx
        ball.y += ball_dy

        # Ball collision with walls
        if ball.left <= 0 or ball.right >= SCREEN_WIDTH:
            ball_dx *= -1
        if ball.top <= 0:
            ball_dy *= -1
        if ball.bottom >= SCREEN_HEIGHT:
            fail += 1
            if trail > tmp_best_catch: tmp_best_catch = trail
            print("Game Over! Succeeded for ", trail, " times")
            trail = 0
            ball = pygame.Rect(random.choice([i*10 for i in range(3,17)]), random.choice([i*10 for i in range(3,25)]), BALL_SIZE, BALL_SIZE)
            ball_dx = BALL_SPEED_X
            ball_dy = BALL_SPEED_Y
            #pygame.quit()
            #sys.exit()

        # Ball collision with paddle
        if ball.colliderect(paddle) and ball_dy > 0:
            ball_dy *= -1
            succ += 1
            trail += 1

        # Drawing
        screen.fill(BLACK)
        pygame.draw.rect(screen, WHITE, paddle)
        pygame.draw.ellipse(screen, WHITE, ball)

        # Refresh screen
        pygame.display.flip()
        #clock.tick(60)
        idx += 1
    if trail > tmp_best_catch: tmp_best_catch = trail
    print("Game Over! Succeeded for ", trail, " times")

has_cycle = detect_paddle_cycle_final(V)["is_cycle"]
if has_cycle:
    print("Has cycle!!!!!")
else:
    print("success | fail times: ", succ, "|", fail, " has cycle: ", detect_paddle_cycle_final(V)["is_cycle"])


    
    
