import numpy as np

class TrainingSet64Inputs:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.load_data()
        
    def fen_to_board(self, fen: str):
        # k = 1, q = 2, r = 3, b = 4, n = 5, p = 6
        # K = 7, Q = 8, R = 9, B = 10, N = 11, P = 12
        board = np.zeros((8, 8))
        fen = fen.split()
        fen = fen[0].split('/')
        for i in range(8):
            j = 0
            for c in fen[i]:
                if c.isdigit():
                    j += int(c)
                else:
                    if c == 'k':
                        board[i][j] = 1
                    elif c == 'q':
                        board[i][j] = 2
                    elif c == 'r':
                        board[i][j] = 3
                    elif c == 'b':
                        board[i][j] = 4
                    elif c == 'n':
                        board[i][j] = 5
                    elif c == 'p':
                        board[i][j] = 6
                    elif c == 'K':
                        board[i][j] = 7
                    elif c == 'Q':
                        board[i][j] = 8
                    elif c == 'R':
                        board[i][j] = 9
                    elif c == 'B':
                        board[i][j] = 10
                    elif c == 'N':
                        board[i][j] = 11
                    elif c == 'P':
                        board[i][j] = 12
                    j += 1
        return board
    
    def load_data(self):
        data = {
            "result": None,
            "checkmate": None,
            "fen": None,
        }
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith("RES:"):
                    data["result"] = lines[i].split()[1]
                if lines[i].startswith("CHECKMATE:"):
                    data["checkmate"] = lines[i].split()[1]
                if lines[i].startswith("FEN:"):
                    data["fen"] = lines[i].split()[1]
                
                if data["fen"] != None and lines[i] == "\n":
                    result = [0, 0, 0, 0]
                    # result[0] = 1 if it's a checkmate, 0 otherwise
                    # result[1] = 1 if white wins, 0 otherwise
                    # result[2] = 1 if black wins, 0 otherwise
                    # result[3] = 1 if it's a draw, 0 otherwise
                    if data["result"] == '1/2-1/2':
                        result = [0, 0, 0, 1]
                    elif data["result"] == '1-0' or data["result"] == '1-O':
                        result = [0, 1, 0, 0]
                    elif data["result"] == '0-1' or data["result"] == 'O-1':
                        result = [0, 0, 1, 0]
                    elif data["result"] == '*':
                        result = [0, 0, 0, 0]
                    elif data["result"] == "1/2":
                        result = [0, 0, 0, 1]
                    elif data["result"] == "-":
                        result = [0, 0, 0, 0]
                    elif data["result"] == "0-0":
                        result = [0, 0, 0, 0]
                    elif data["result"] == "":
                        result = None
                    else:
                        print(data["result"])
                        raise Exception("Invalid result")
                    if data["checkmate"] == 'True':
                        result[0] = 1
                    elif data["checkmate"] == 'False':
                        result[0] = 0
                    board = self.fen_to_board(data["fen"])
                    board = board.astype(int).flatten()
                    self.data.append((board, result))
                    data = {
                        "result": None,
                        "checkmate": None,
                        "fen": None,
                    }
                
    def get_formatted_data(self):
        X = []
        y = []
        for board, result in self.data:
            X.append(board)
            y.append(result)
        return np.array(X), np.array(y)

class TrainingSet:
    
    def __init__(self, filename):
        
        self.filename = filename
        self.data = []
        self.load_data()
        
    def fen_to_board(self, fen):
        # k = king, q = queen, r = rook, b = bishop, n = knight, p = pawn
        board = np.zeros((6, 8, 8))
        fen = fen.split()
        fen = fen[0].split('/')
        for i in range(8):
            j = 0
            for c in fen[i]:
                if c.isdigit():
                    j += int(c)
                else:
                    if c == 'k':
                        board[0][i][j] = -1
                    elif c == 'q':
                        board[1][i][j] = -1
                    elif c == 'r':
                        board[2][i][j] = -1
                    elif c == 'b':
                        board[3][i][j] = -1
                    elif c == 'n':
                        board[4][i][j] = -1
                    elif c == 'p':
                        board[5][i][j] = -1
                    elif c == 'K':
                        board[0][i][j] = 1
                    elif c == 'Q':
                        board[1][i][j] = 1
                    elif c == 'R':
                        board[2][i][j] = 1
                    elif c == 'B':
                        board[3][i][j] = 1
                    elif c == 'N':
                        board[4][i][j] = 1
                    elif c == 'P':
                        board[5][i][j] = 1
                    j += 1
        return board
        
    def load_data(self):
        data = {
            "result": None,
            "checkmate": None,
            "fen": None,
        }
        with open(self.filename, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if lines[i].startswith("RES:"):
                    data["result"] = lines[i].split()[1]
                elif lines[i].startswith("CHECKMATE:"):
                    data["checkmate"] = lines[i].split()[1]
                elif lines[i].startswith("FEN:"):
                    data["fen"] = lines[i].split()[1]
                elif data["fen"] != None and lines[i] == "\n":
                    result = [0, 0, 0, 0]
                    # result[0] = 1 if it's a checkmate, 0 otherwise
                    # result[1] = 1 if white wins, 0 otherwise
                    # result[2] = 1 if black wins, 0 otherwise
                    # result[3] = 1 if it's a draw, 0 otherwise
                    if data["result"] == '1/2-1/2':
                        result = [0, 0, 0, 1]
                    elif data["result"] == '1-0' or data["result"] == '1-O' or data["result"] == '1-00' or data["result"] == '+/-':
                        result = [0, 1, 0, 0]
                    elif data["result"] == '0-1' or data["result"] == 'O-1' or data["result"] == '00-1' or data["result"] == '-/+':
                        result = [0, 0, 1, 0]
                    elif data["result"] == '*':
                        result = [0, 0, 0, 0]
                    elif data["result"] == "1/2":
                        result = [0, 0, 0, 1]
                    elif data["result"] == "-":
                        result = [0, 0, 0, 0]
                    elif data["result"] == "0-0":
                        result = [0, 0, 0, 1]
                    elif data["result"] == "":
                        result = None
                    else:
                        print(data["result"])
                        raise Exception("Invalid result")
                    if data["checkmate"] == 'True':
                        result[0] = 1
                    elif data["checkmate"] == 'False':
                        result[0] = 0
                    board = self.fen_to_board(data["fen"])
                    board = board.astype(int).flatten()
                    self.data.append((board, result))
                    data = {
                        "result": None,
                        "checkmate": None,
                        "fen": None,
                    }
                i += 1
    
    def get_formatted_data(self):
        X = []
        y = []
        for board, result in self.data:
            X.append(board)
            y.append(result)
        return np.array(X), np.array(y)
