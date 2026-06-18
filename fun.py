#encoding=utf-8
import numpy as np
import pygame
import random
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import random



def get_segments(paddle_x, W=10):
    return [paddle_x[i:i+W] for i in range(0, len(paddle_x)-W, W)]


def detect_paddle_cycle_final(V, max_period=None, min_repeats=2):
    n = len(V)
    if n < 4:
        return {"is_cycle": False}

    arr = V

    if max_period is None:
        max_period = n // min_repeats

    detected_m = -1
    min_m = 10000000
    for m in range(50, max_period + 1):
        if arr[-m:] == arr[-2*m : -m]:
            detected_m = m
            if detected_m < min_m: min_m = detected_m 
    
    if detected_m == -1:
        return {"is_cycle": False, "feq": 0}
    else:
        return {"is_cycle": True, "feq": min_m}


def int_to_binary_list(n, bits = 8):
    twos_complement = n & (2**bits - 1)
    return [int(bit) for bit in format(twos_complement, f'0{bits}b')]


def binary_list_to_int(binary_list):
    bits = len(binary_list)
    value = int(''.join(str(b) for b in binary_list), 2)
    if binary_list[0] == 1:
        value -= 2**bits
    return value


def accuracy_test(model, is_transformer=False, seq_len=40):
    cnt = 0
    succ_all = 0
    fail_all = 0
    best_catch = 0
    print("start testing!!!")
    device = model.device
    all_cnt = 0
    while cnt < 10 and all_cnt <50:
        seed = random.randint(-100000, 100000)
        print("the random seed is: ", seed)
        tmp_best_catch = 0
        torch.manual_seed(seed)
        random.seed(seed) 
        states = None
        neuron_outputs = None

        model.reset_state()

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

        if is_transformer:
            x_input = torch.zeros(1, seq_len, 24).to(device)
        else:
            x_input = None

        V = []

        with torch.no_grad():
            trail = 0
            while idx < 20000:
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
                if not is_transformer:
                    x_input = torch.tensor(tmp, dtype=torch.float32).reshape(1,-1,24).to(device)
                else:
                    new_in = torch.tensor(tmp, dtype=torch.float32).reshape(1,-1,24).to(device)
                    x_input = torch.cat((x_input[:,1:,:], new_in), dim=1).to(device)
    
    
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
        all_cnt += 1
            
        has_cycle = detect_paddle_cycle_final(V)["is_cycle"]
        
        if has_cycle:
            print("Has cycle!!!!!")
            continue
        else:
            succ_all += succ
            fail_all += fail
            cnt += 1
            if tmp_best_catch > best_catch: best_catch = tmp_best_catch
            print("success | fail times: ", succ, "|", fail, " has cycle: ", detect_paddle_cycle_final(V)["is_cycle"])

    return succ_all, fail_all, best_catch