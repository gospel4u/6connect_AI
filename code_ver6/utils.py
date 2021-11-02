from collections import deque

import numpy as np
#from sklearn import preprocessing


ALPHABET = ' A B C D E F G H I J K L M N O P Q R S'


def valid_actions(board):
    '''
    Validate the actios be zero.
    Used for PUCTAgent, UCTAgent, and RandomAgent, not ZeroAgent.
    '''
    actions = []
    count = 0
    board_size = len(board)

    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                actions.append([(i, j), count])
            count += 1

    return actions


def legal_actions(node_id, board_size):
    '''
    Get the legal actions.
    Used for ZeroAgent.
    Return avaliable action list.
    '''
    all_action = {a for a in range(board_size**2)}
    action = set(node_id[1:])
    actions = all_action - action

    return list(actions)


def check_win(board, win_mark):
    board = board.copy()
    num_mark = np.count_nonzero(board)
    board_size = len(board)
    current_grid = np.zeros([win_mark, win_mark])
    for row in range(board_size - win_mark + 1):
        for col in range(board_size - win_mark + 1):
            current_grid = board[row: row + win_mark, col: col + win_mark]
            sum_horizontal = np.sum(current_grid, axis=1)
            sum_vertical = np.sum(current_grid, axis=0)
            sum_diagonal_1 = np.sum(current_grid.diagonal())
            sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())

            # Black wins! (Horizontal and Vertical)
            if win_mark in sum_horizontal or win_mark in sum_vertical:
                return 1
            # Black wins! (Diagonal)
            if win_mark == sum_diagonal_1 or win_mark == sum_diagonal_2:
                return 1
            # White wins! (Horizontal and Vertical)
            if -win_mark in sum_horizontal or -win_mark in sum_vertical:
                return 2
            # White wins! (Diagonal)
            if -win_mark == sum_diagonal_1 or -win_mark == sum_diagonal_2:
                return 2
    # Draw (board is full)
    if num_mark == board_size * board_size:
        return 3
    # If No winner or no draw
    return 0


def render_str(board, board_size, action_index):
    if action_index is not None:
        row = action_index // board_size
        col = action_index % board_size

    count = np.count_nonzero(board)

    board_str = '\n  {}\n'.format(ALPHABET[:board_size * 2])

    for i in range(board_size):
        for j in range(board_size):
            if j == 0:
                board_str += '{:2}'.format(i + 1)

            # Blank board
            if board[i][j] == 0:
                if count > 0:
                    if col + 1 < board_size:
                        if (i, j) == (row, col + 1):
                            board_str += '.'
                        else:
                            board_str += ' .'
                    else:
                        board_str += ' .'
                else:
                    board_str += ' .'

            # Black stone
            if board[i][j] == 1:
                if (i, j) == (row, col):
                    board_str += '(O)'
                elif (i, j) == (row, col + 1):
                    board_str += 'O'
                else:
                    board_str += ' O'

            # White stone
            if board[i][j] == -1:
                if (i, j) == (row, col):
                    board_str += '(X)'
                elif (i, j) == (row, col + 1):
                    board_str += 'X'
                else:
                    board_str += ' X'
            
            # # Red stone
            # if lock[i][j] == 1:
            #     if (i, j) == (row, col):
            #         board_str += '(R)'
            #     elif (i, j) == (row, col + 1):
            #         board_str += 'R'
            #     else:
            #         board_str += ' R'

            if j == board_size - 1:
                board_str += ' \n'

        if i == board_size - 1:
            board_str += '  ' + '-' * (board_size - 6) + \
                '  MOVE: {:2}  '.format(count) + '-' * (board_size - 6)

    print(board_str)


def get_state_tf(id, turn, board_size, channel_size):
    state = np.zeros([board_size, board_size, channel_size])
    length_game = len(id)

    state_1 = np.zeros([board_size, board_size])
    state_2 = np.zeros([board_size, board_size])

    channel_idx = channel_size - 1

    for i in range(length_game):
        row_idx = int(id[i] / board_size)
        col_idx = int(id[i] % board_size)

        if i != 0:
            if i % 2 == 0:
                state_1[row_idx, col_idx] = 1
            else:
                state_2[row_idx, col_idx] = 1

        if length_game - i < channel_size:
            channel_idx = length_game - i - 1
            if i % 2 == 0:
                state[:, :, channel_idx] = state_1
            else:
                state[:, :, channel_idx] = state_2

    if turn == 0:
        state[:, :, channel_size - 1] = 1
    else:
        state[:, :, channel_size - 1] = 0

    return state


def get_state_pt(node_id, board_size, channel_size, win_mark=5):
    '''
    state for evaluating
    '''
    state_b = np.zeros((board_size, board_size))
    state_w = np.zeros((board_size, board_size))
    color = np.ones((board_size, board_size))
    color_idx = 1
    history = deque(
        [np.zeros((board_size, board_size)) for _ in range(channel_size)],
        maxlen=channel_size)

    turn = 0
    for i, action_idx in enumerate(node_id):
        if win_mark == 6:
            turn = (turn + (i % 2)) % 2
        else:
            turn = i

        if turn == 0:
            history.append(state_b.copy())
            history.append(state_w.copy())
        else:
            row = action_idx // board_size
            col = action_idx % board_size

            if turn % 2 == 1:
                state_b[row, col] = 1
                history.append(state_b.copy())
                color_idx = 0
            else:
                state_w[row, col] = 1
                history.append(state_w.copy())
                color_idx = 1

    history.append(color * color_idx)
    state = np.stack(history)
    return state


def get_board(node_id, board_size, win_mark=5):
    '''
    used in selection for ZeroAgent
    used in expansion for PUCT
    cleanup board for reward    

    node_id: history of game. (0, a_1, ..., a_N)
    board_size: 3 or 9 or 15
    '''
    board = np.zeros(board_size**2)
    turn = 0
    for i, action_index in enumerate(node_id[1:]): # it is available from 1, initial root_id: (0,)
        # (i+1) means the number of stones
        if win_mark == 6:
            turn = (turn + (i % 2)) % 2
        else:
            turn = i
            
        if turn % 2 == 0:
            board[action_index] = 1
        else:
            board[action_index] = -1

    return board.reshape(board_size, board_size)


def get_turn(node_id):
    '''
    d
    '''
    if len(node_id) % 2 == 1:
        return 0
    else:
        return 1


def get_action(pi, idx, count, state, board_size):
    if idx == 0 and count < 20: # 7x7
        pi2 = []
        head = 64
        tail = 70
        for i in range(7):
            pi2.append(pi[head:tail+1])
            head += 15
            tail += 15
        pi2 = np.round_(np.array(pi2).reshape(-1), 4)
        print("pi2 : ", pi2)
        if not np.any(pi2) == True:
            print("in get_action : ", state)
            while True:
                idx = np.random.randint(64, 161)
                row = idx // board_size
                col = idx % board_size
                if state[row, col] == 0:
                    action_index = idx
                    action_size = len(pi)
                    action = np.zeros(action_size)
                    break
        else: 
            print("before else pi2 : ", pi2)
            pi2 /= np.nansum(pi2)
            np.nan_to_num(pi2, copy=False)
            action_size = len(pi)
            action = np.zeros(action_size)
            action_size = 7*7

            print("after else pi2 : ", pi2)
            action_index = np.random.choice(action_size, p=pi2, replace=False)
            print(action_index)

            head = 64
            if 0 <= action_index < 7:
                action_index = head + action_index
            elif 7 <= action_index < (7*2):
                action_index = head + 8 + action_index
            elif (7*2) <= action_index < (7*3):
                action_index = head + (8 * 2) + action_index
            elif (7*3) <= action_index < (7*4):
                action_index = head + (8 * 3) + action_index
            elif (7*4) <= action_index < (7*5):
                action_index = head + (8 * 4) + action_index
            elif (7*5) <= action_index < (7*6):
                action_index = head + (8 * 5) + action_index
            elif(7*6) <= action_index < (7*7):
                action_index = head + (8 * 6) + action_index
            print("max_idx", pi2.argmax())
        print("action_index", action_index)
        action[action_index] = 1
    elif idx == 0 and count < 40: # 9x9
        pi2 = []
        head = 48
        tail = 56
        for i in range(9):
            pi2.append(pi[head:tail+1])
            head += 15
            tail += 15
        pi2 = np.round_(np.array(pi2).reshape(-1), 4)
        if not np.any(pi2) == True:
            print("in get_action : ", state)
            while True:
                idx = np.random.randint(48, 177)
                row = idx // board_size
                col = idx % board_size
                if state[row, col] == 0:
                    action_index = idx
                    action_size = len(pi)
                    action = np.zeros(action_size)
                    break
        else:
            pi2 /= np.nansum(pi2)
            np.nan_to_num(pi2, copy=False)
            action_size = len(pi)
            action = np.zeros(action_size)
            action_size = 9*9
            
            print(pi2)
            action_index = np.random.choice(action_size, p=pi2, replace=False)
            print(action_index)

            head = 48
            if 0 <= action_index < 9:
                action_index = head + action_index
            elif 9 <= action_index < (9*2):
                action_index = head + 6 + action_index
            elif (9*2) <= action_index < (9*3):
                action_index = head + (6 * 2) + action_index
            elif (9*3) <= action_index < (9*4):
                action_index = head + (6 * 3) + action_index
            elif (9*4) <= action_index < (9*5):
                action_index = head + (6 * 4) + action_index
            elif (9*5) <= action_index < (9*6):
                action_index = head + (6 * 5) + action_index
            elif( 9*6) <= action_index < (9*7):
                action_index = head + (6 * 6) + action_index
            elif( 9*7) <= action_index < (9*8):
                action_index = head + (6 * 7) + action_index
            elif( 9*8 )<= action_index < (9*9):
                action_index = head + (6 * 8) + action_index

            print("max_idx", pi2.argmax())
        print("action_index", action_index)
        action[action_index] = 1
    elif idx == 0 and count < 60: #11x11
        pi2 = []
        head = 32
        tail = 42
        for i in range(11):
            pi2.append(pi[head:tail+1])
            head += 15
            tail += 15
        pi2 = np.round_(np.array(pi2).reshape(-1), 4)
        if not np.any(pi2) == True:
            print("in get_action : ", state)
            while True:
                idx = np.random.randint(32, 193)
                row = idx // board_size
                col = idx % board_size
                if state[row, col] == 0:
                    action_index = idx
                    action_size = len(pi)
                    action = np.zeros(action_size)
                    break
        else:
            pi2 /= np.nansum(pi2)
            np.nan_to_num(pi2, copy=False)
            action_size = len(pi)
            action = np.zeros(action_size)
            action_size = 11*11
            
            print(pi2)
            action_index = np.random.choice(action_size, p=pi2, replace=False)
            print(action_index)

            head = 32
            if 0 <= action_index < 11:
                action_index = head + action_index
            elif 11 <= action_index < (11*2):
                action_index = head + 4 + action_index
            elif (11*2) <= action_index < (11*3):
                action_index = head + (4 * 2) + action_index
            elif (11*3) <= action_index < (11*4):
                action_index = head + (4 * 3) + action_index
            elif (11*4) <= action_index < (11*5):
                action_index = head + (4 * 4) + action_index
            elif (11*5) <= action_index < (11*6):
                action_index = head + (4 * 5) + action_index
            elif(11*6) <= action_index < (11*7):
                action_index = head + (4 * 6) + action_index
            elif(11*7) <= action_index < (11*8):
                action_index = head + (4 * 7) + action_index
            elif(11*8)<= action_index < (11*9):
                action_index = head + (4 * 8) + action_index
            elif(11*9)<= action_index < (11*10):
                action_index = head + (4 * 9) + action_index
            elif(11*10)<= action_index < (11*11):
                action_index = head + (4 * 10) + action_index

            print("max_idx", pi2.argmax())
        print("action_index", action_index)
        action[action_index] = 1
    elif idx == 0 and count < 80: #13x13
        pi2 = []
        head = 16
        tail = 28
        for i in range(13):
            pi2.append(pi[head:tail+1])
            head += 15
            tail += 15
        pi2 = np.round_(np.array(pi2).reshape(-1), 4)
        if not np.any(pi2) == True:
            print("in get_action : ", state)
            while True:
                idx = np.random.randint(16, 209)
                row = idx // board_size
                col = idx % board_size
                if state[row, col] == 0:
                    action_index = idx
                    action_size = len(pi)
                    action = np.zeros(action_size)
                    break
        else:
            pi2 /= np.nansum(pi2)
            np.nan_to_num(pi2, copy=False)
            action_size = len(pi)
            action = np.zeros(action_size)
            action_size = 13*13
            
            print(pi2)
            action_index = np.random.choice(action_size, p=pi2, replace=False)
            print(action_index)

            head = 16
            if 0 <= action_index < 13:
                action_index = head + action_index
            elif 13 <= action_index < (13*2):
                action_index = head + 2 + action_index
            elif (13*2) <= action_index < (13*3):
                action_index = head + (2 * 2) + action_index
            elif (13*3) <= action_index < (13*4):
                action_index = head + (2 * 3) + action_index
            elif (13*4) <= action_index < (13*5):
                action_index = head + (2 * 4) + action_index
            elif (13*5) <= action_index < (13*6):
                action_index = head + (2 * 5) + action_index
            elif(13*6) <= action_index < (13*7):
                action_index = head + (2 * 6) + action_index
            elif(13*7) <= action_index < (13*8):
                action_index = head + (2 * 7) + action_index
            elif(13*8)<= action_index < (13*9):
                action_index = head + (2 * 8) + action_index
            elif(13*9)<= action_index < (13*10):
                action_index = head + (2 * 9) + action_index
            elif(13*10)<= action_index < (13*11):
                action_index = head + (2 * 10) + action_index
            elif(13*11)<= action_index < (13*12):
                action_index = head + (2 * 11) + action_index
            elif(13*12)<= action_index < (13*13):
                action_index = head + (2 * 12) + action_index

            print("max_idx", pi2.argmax())
        print("action_index", action_index)
        action[action_index] = 1
    elif idx == 0: # 15x15
        action_size = len(pi)
        action = np.zeros(action_size)
        if not np.any(pi) == True:
            while True:
                idx = np.random.randint(0, 255)
                row = idx // board_size
                col = idx % board_size
                if state[row, col] == 0:
                    action_index = idx
                    break
        else:
            pi /= np.nansum(pi)
            np.nan_to_num(pi, copy=False)
            action_index = np.random.choice(action_size, p=pi, replace=False)
            print("max_idx", pi.argmax())
        print("action_index", action_index)
        action[action_index] = 1
    else:
        action_size = len(pi)
        action = np.zeros(action_size)
        action_index = (15*15) // 2
        action[action_index] = 1

    return action, action_index


def argmax_onehot(pi):
    action_size = len(pi)
    action = np.zeros(action_size)
    max_idx = np.argwhere(pi == pi.max())
    action_index = max_idx[np.random.choice(len(max_idx))]
    action[action_index] = 1

    return action, action_index[0]


def get_reward(win_index, leaf_id):
    '''
    not used for ZeroAgent
    '''
    turn = get_turn(leaf_id)
    if win_index == 1:
        if turn == 1:
            reward = 1.
        else:
            reward = -1.
    elif win_index == 2:
        if turn == 1:
            reward = -1.
        else:
            reward = 1.
    else:
        reward = 0.

    return reward


def augment_dataset(memory, board_size):
    aug_dataset = []
    for (s, pi, z) in memory:
        for i in range(4):
            s_rot = np.rot90(s, i, axes=(1, 2)).copy()
            pi_rot = np.rot90(pi.reshape(board_size, board_size), i)
            pi_flat = pi_rot.flatten().copy()
            aug_dataset.append((s_rot, pi_flat, z))

            s_flip = np.fliplr(s_rot).copy()
            pi_flip = np.fliplr(pi_rot).flatten().copy()
            aug_dataset.append((s_flip, pi_flip, z))

    return aug_dataset