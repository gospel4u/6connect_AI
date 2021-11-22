import numpy as np
import torch

import agents
import model
import utils

# env_small: 9x9, env_regular: 15x15
from env import env_connect6 as game

# Web API
import logging
import threading
import flask
# from webapi import web_api
from webapi import game_info
from webapi import player_agent_info
from webapi import enemy_agent_info
from info.agent_info import AgentInfo
from info.game_info import GameInfo

import pickle
from CONNSIX import connsix
import time

BOARD_SIZE = game.Return_BoardParams()[0]

N_BLOCKS_PLAYER = 10
# N_BLOCKS_ENEMY = 10

IN_PLANES_PLAYER = 5  # history * 2 + 1
# IN_PLANES_ENEMY = 5

OUT_PLANES_PLAYER = 128
# OUT_PLANES_ENEMY = 128

WIN_STONES = 6
N_MCTS_PLAYER = 100
# N_MCTS_ENEMY = 10
N_MCTS_MONITOR = 50

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# ==================== input model path ================= #
#       'human': human play       'puct': PUCB MCTS       #
#       'uct': UCB MCTS           'random': random        #
#       'web': web play                                   #
# ======================================================= #
# example)
# with open('./211109_1800_1495832_step_model.pickle', 'rb') as f:
#     our_model = pickle.load(f)

PATH = "./data/211117_1200_1027711_step_model.pth"
MONITOR_PATH = "./data/211117_1200_1027711_step_model.pth"

# 불러오기
our_model = torch.load(PATH, map_location=device)

print(our_model)

player_model_path = 'human'
#enemy_model_path = our_model#'./data/210926_10_0_step_model.pickle'
#monitor_model_path = our_model#'./data/210926_10_0_step_model.pickle'


class Evaluator(object):
    def __init__(self):
        self.player = None
        self.monitor = None
        pass

    def set_agents(self, model_path_a, model_path_b):
        self.env = game.GameState('text')

        self.player = agents.ZeroAgent(BOARD_SIZE,
                                          game.WIN_STONES,
                                          N_MCTS_PLAYER,
                                          IN_PLANES_PLAYER,
                                          noise=False)
        self.player.model = model.PVNet(N_BLOCKS_PLAYER,
                                           IN_PLANES_PLAYER,
                                           OUT_PLANES_PLAYER,
                                           BOARD_SIZE).to(device)
        
        state = self.player.model.state_dict()

        my_state = torch.load(
                model_path_a, map_location='cuda:0' if use_cuda else 'cpu')
        for k, v in my_state.state_dict().items():
            if k in state:
                state[k] = v
        self.player.model.load_state_dict(state)

        self.monitor = agents.ZeroAgent(BOARD_SIZE,
                                        game.WIN_STONES, 
                                        N_MCTS_MONITOR, 
                                        IN_PLANES_PLAYER, 
                                        noise = False)
        
        self.monitor.model = model.PVNet(N_BLOCKS_PLAYER,
                                        IN_PLANES_PLAYER,
                                        OUT_PLANES_PLAYER,
                                        BOARD_SIZE).to(device)

        state_m = self.monitor.model.state_dict()

        my_state_m = torch.load(
                model_path_b, map_location='cuda:0' if use_cuda else 'cpu')
        for k, v in my_state_m.state_dict().items():
            if k in state_m:
                state_m[k] = v
        self.monitor.model.load_state_dict(state_m)

        #self.player.model.eval()
        #self.monitor.model.eval()
    
    def get_action(self, root_id, board, turn, enemy_turn, count, state_arr):
        if isinstance(self.player, agents.ZeroAgent):
            pi = self.player.get_pi(root_id, tau=0)
        else:
            pi = self.player.get_pi(root_id, board, turn, tau=0)

        action, action_index = utils.argmax_onehot(pi)
        
        # action, action_index = utils.get_action(pi, 0, count=count, state=state_arr, board_size = BOARD_SIZE)

        return action, action_index

    def return_env(self):
        return self.env

    def reset(self):
        self.player.reset()


def invert(action_index, board_size=19):
    str = ''
    row = action_index // board_size
    col = action_index % board_size
    print('row', row)
    print('col', col)
    if 0 <= col <= 7:
        str = '{0}{1:02d}'.format(chr(ord('A')+col), (18-row)+1)
    elif 8 <= col <= 18:
        str = '{0}{1:02d}'.format(chr(ord('B')+col), (18-row)+1)
    return str

def response_split(str):
    
    response = []
    if len(str) > 3:
            response = str.split(':')
    else:
        response.append(str)
    r = []

    for i in range(len(response)):
        r.append(response[i][0])
        r.append(int(response[i][1] + response[i][2]))

    action_index = []

    for i in range(0, len(r), 2):
        tail = 0
        head = (19-r[i+1]) * 19
        if ord('A') <= ord(r[i]) < ord('I'):
            tail = ord(r[i]) - ord('A')
        if ord('I') < ord(r[i]) <= ord('T'):
            tail = ord(r[i]) - ord('B')
        action_index.append(int(head + tail))
    return action_index

def cordinate(index):
    row = index // BOARD_SIZE
    col = index % BOARD_SIZE
    return row, col

evaluator = Evaluator()

def main():
    ip = input("input ip: ")
    
    port = int(input("input port number: "))
    
    our_color = input("input BLACK or WHITE: ")
    
    board = np.zeros([BOARD_SIZE, BOARD_SIZE])
    lock = np.zeros([BOARD_SIZE, BOARD_SIZE])
    count = 0
    
    evaluator.set_agents(PATH, MONITOR_PATH)
    player_agent_info.agent = evaluator.player
    env = evaluator.return_env()
    #evaluator.player.model.eval()
    root_id = (0,)
    board = utils.get_board(root_id, BOARD_SIZE)
    
    red_stones = connsix.lets_connect(ip, port, our_color)
    
    red_indexes = []

    if len(red_stones):
        print("Received red stones from server: " + red_stones)
        red_indexes = response_split(red_stones)

        for red in red_indexes:
            print('red', red)
            action = np.zeros(19*19)
            row, col = cordinate(red)
            board[row, col] = 5
            action[red] = 5
            root_id += (red,)
            board, check_valid_pos, win_index, _, _ = env.step(action, 1)
    
    if our_color == "BLACK":
        turn = 0
        enemy_turn = 1
    else:
        turn = 1
        enemy_turn = 0
    
    
    if our_color == "BLACK":
        root_id += (180, )
        row, col = cordinate(180)
        board[row, col] = 1
        action = np.zeros(19*19)
        action[180] = 1
        board, check_valid_pos, win_index, turn, _ = env.step(action)
        
        away_move = connsix.draw_and_read("K10")
        print("Received first away move from server: " + away_move)
        
        init_indexes = response_split(away_move)
        row, col = cordinate(init_indexes[0])
        board[row, col] = -1
        action = np.zeros(19*19)
        action[init_indexes[0]] = 1
        root_id += (init_indexes[0],)
        board, check_valid_pos, win_index, turn, _ = env.step(action)
        
        row, col = cordinate(init_indexes[1])
        board[row, col] = -1
        action = np.zeros(19*19)
        action[init_indexes[1]] = 1
        root_id += (init_indexes[1],)
        board, check_valid_pos, win_index, turn, _ = env.step(action)
        
        count = 3
    else:
        away_move = connsix.draw_and_read("")
        print("Received first away move from server: " + away_move)
        
        init_indexes = response_split(away_move)
        row, col = cordinate(init_indexes[0])
        board[row, col] = -1
        action = np.zeros(19*19)
        action[init_indexes[0]] = 1
        root_id += (init_indexes[0],)
        board, check_valid_pos, win_index, turn, _ = env.step(action)
        
        count = 1

    player_list = []
    action_index = None
    away_move = ''
    
    check_enemy_iter = 0
    turn_iter = 0
    turn_change = False
    state_arr = utils.get_board(root_id, BOARD_SIZE)
    while 1:
        #utils.render_str(board, BOARD_SIZE, action_index)
        if turn_change == False or turn_iter == 0:
            if turn_iter == 1:  
                # player turn
                print("board1: ", board)
                state = utils.get_state_pt(root_id, BOARD_SIZE, 5, game.WIN_STONES)
                # state_input = torch.tensor([state]).to(device).float()
                # v = our_model.model(state_input)
                # pi = np.asarray(v)
                # pi = pi.tolist()
                # pi = pi[0][0]
                # print("pi", pi)
                # action, action_index = utils.argmax_onehot(pi)
                p, v = evaluator.player.get_pv(root_id)
                action, action_index = evaluator.get_action(root_id,
                                                        board,
                                                        enemy_turn,
                                                        turn,
                                                        count,
                                                        state_arr)

                action[action_index] = 1
                board, check_valid_pos, win_index, turn, _ = env.step(action)
                print("board1b: ", board)
                
                count += 1
                player_list.append(invert(action_index))
                print("Player list", player_list)
                if len(player_list) == 2:
                    print("TEST")
                    result_player = '{}:{}'.format(player_list[0], player_list[1])
                    player_list = []
                    away_move = connsix.draw_and_read(result_player)
                    print("awaymove1", away_move)
                root_id += (action_index,)
                
                turn_iter = 1
                turn_change = True
                evaluator.player.del_parents(root_id)
            else:
                # player turn
                print("board2: ", board)
                state = utils.get_state_pt(root_id, BOARD_SIZE, 5, game.WIN_STONES)
                # state_input = torch.tensor([state]).to(device).float()
                # v = our_model.model(state_input)
                # pi = np.asarray(v)
                # pi = pi.tolist()
                # pi = pi[0][0]
            
                # print("pi", pi)
                # action, action_index = utils.argmax_onehot(pi)
                                                        
                action, action_index = evaluator.get_action(root_id,
                                                        board,
                                                        enemy_turn,
                                                        turn,
                                                        count,
                                                        state_arr)
                action[action_index] = 1
                board, check_valid_pos, win_index, turn, _ = env.step(action)
                print("board2b: ", board)
                
                count += 1
                player_list.append(invert(action_index))
                print("Player list", player_list)
                if len(player_list) == 2:
                    print("TEST")
                    result_player = '{}:{}'.format(player_list[0], player_list[1])
                    player_list = []
                    away_move = connsix.draw_and_read(result_player)
                    print("awaymove1", away_move)
                root_id += (action_index,)
                
                turn_iter = 1
                turn_change = False
                evaluator.player.del_parents(root_id)

            move = np.count_nonzero(board)

        else:
            # enemy turn
            if check_enemy_iter % 2 == 1:
                print("board3: ", board)
                print("awaymove2",away_move)
                action_indexes = response_split(away_move)
                
                # if our_color == "BLACK":
                #     action[action_indexes[0]] = 1
                #     action[action_indexes[1]] = 1
                # else:
                action = np.zeros(19*19)
                action[action_indexes[0]] = 1
                board, check_valid_pos, win_index, turn, _ = env.step(action)
                root_id += (action_indexes[0],)
                
                action = np.zeros(19*19)
                action[action_indexes[1]] = 1
                board, check_valid_pos, win_index, turn, _ = env.step(action)
                root_id += (action_indexes[1],)
                print("board3b: ", board)
                check_enemy_iter = 0
                turn_change = False
                turn_iter = 0

                
            else:
                print("board4: ", board)
                check_enemy_iter += 1
                print("board4b: ", board)

            player_agent_info.add_value(move, v)
            
        print("turn_change: ", turn_change)
        print("turn_iter: ", turn_iter)
        state_arr = utils.get_board(root_id, BOARD_SIZE)
if __name__ == '__main__':
    print('cuda:', use_cuda)
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)
    
    main()