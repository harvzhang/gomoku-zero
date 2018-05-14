import enum
import numpy as np

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")


class GomokuEnv:
    def __init__(self):
        self.board = None
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False
        self.board_size = (15,15)
        self.pieces = self.board_size[0] * self.board_size[1]

    def reset(self):
        self.board = []

        for i in range(self.board_size[0]):
            self.board.append([])
            for j in range(self.board_size[1]):
                self.board[i].append(' ')

        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = np.copy(board)
        self.turn = self.turn_n()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def turn_n(self):
        turn = 0
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i][j] != ' ':
                    turn += 1

        return turn

    def player_turn(self):
        if self.turn % 2 == 0:
            return Player.white
        else:
            return Player.black

    def step(self, action):
        if action is None:
            self._resigned()
            return self.board, {}

        token = ('X' if self.player_turn() == Player.white else 'O')
        x = int(action / self.board_size[1])
        y = int(action %  self.board_size[1])
        self.board[x][y] = token

        self.turn += 1

        self.check_for_fives()

        if self.turn >= self.pieces:
            self.done = True
            if self.winner is None:
                self.winner = Winner.draw

        return self.board, {}

    def legal_moves(self):
        legal = []
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i][j] == ' ':
                    legal.append(1)
                else:   
                    legal.append(0)
    
        return legal

    def check_for_fives(self):
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i][j] != ' ':
                    # check if a vertical five-in-a-row starts at (i, j)
                    if self.vertical_check(i, j):
                        self.done = True
                        return

                    # check if a horizontal five-in-a-row starts at (i, j)
                    if self.horizontal_check(i, j):
                        self.done = True
                        return

                    # check if a diagonal (either way) starts at (i, j)
                    diag_five = self.diagonal_check(i, j)
                    if diag_five:
                        self.done = True
                        return

    def vertical_check(self, row, col):
        # print("checking vert")
        five_in_a_row = False
        consecutive_count = 0
        for i in range(row, self.board_size[0]):
            if self.board[i][col].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 5:
            five_in_a_row = True
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        return five_in_a_row

    def horizontal_check(self, row, col):
        five_in_a_row = False
        consecutive_count = 0

        for j in range(col, self.board_size[1]):
            if self.board[row][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 5:
            five_in_a_row = True
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        return five_in_a_row

    def diagonal_check(self, row, col):
        five_in_a_row = False
        count = 0

        consecutive_count = 0
        j = col
        for i in range(row, self.board_size[0]):
            #print(i, j, ' ', self.board[i][j].lower())
            if j >= self.board_size[1]:
                break
            elif self.board[i][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= 5:
            count += 1
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        consecutive_count = 0
        j = col
        for i in range(row, -1, -1):
            #print(i, j, ' ', self.board[i][j].lower() )
            if j >= self.board_size[1]:
                break
            elif self.board[i][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= 5:
            count += 1
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        if count > 0:
            five_in_a_row = True

        return five_in_a_row

    def _resigned(self):
        if self.player_turn() == Player.white:
            self.winner = Winner.white
        else:
            self.winner = Winner.black
        self.done = True
        self.resigned = True

    def black_and_white_plane(self):
        board_white = np.copy(self.board)
        board_black = np.copy(self.board)
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                if self.board[i][j] == ' ':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                elif self.board[i][j] == 'X':
                    board_white[i][j] = 1
                    board_black[i][j] = 0
                else:
                    board_white[i][j] = 0
                    board_black[i][j] = 1

        return np.array(board_white), np.array(board_black)

    def render(self):
        print("\nRound: " + str(self.turn))

        #this is super hacky
        print ("\t", end="")
        for j in range(self.board_size[1]):
            if j+1 >= 10:
                print (j+1, "", end = "")
            else:
                print (j+1, " ",end = "")
        print("")
            
        for i in range(self.board_size[0]):
            print(i + 1, "\t", end="")
            
            for j in range(self.board_size[1]):
                if self.board[i][j] == ' ':
                    print ('*', end="")
                else:
                    print(str(self.board[i][j]), end="")
                print("  ", end="")
            print("")

        if self.done:
            print("Game Over!")
            if self.winner == Winner.white:
                print("X is the winner")
            elif self.winner == Winner.black:
                print("O is the winner")
            else:
                print("Game was a draw")

    @property
    def observation(self):
        return ''.join(''.join(x for x in y) for y in self.board)
