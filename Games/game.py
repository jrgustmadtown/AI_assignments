import random
import copy

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko. """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color. """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.cache = {}  # Cache for storing evaluated states

    def run_challenge_test(self):
        """ Set to True if you would like to run gradescope against the challenge AI! """
        return True

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. """
        # Count pieces to determine game phase
        flat_state = [cell for row in state for cell in row]
        b_count = flat_state.count('b')
        r_count = flat_state.count('r')
        drop_phase = (b_count + r_count < 8)

        # Reset cache for each new move
        self.cache = {}
        
        # Use a moderate search depth for better performance
        depth = 3
        
        # Get best move using minimax with alpha-beta pruning
        _, best_state = self.max_value(state, depth, float('-inf'), float('inf'))
        
        # Find the differences between current state and best state
        if drop_phase:
            # In drop phase, find where the new piece is placed
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ' and best_state[r][c] == self.my_piece:
                        return [(r, c)]
            
            # Fallback: find any empty space
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        return [(r, c)]
        else:
            # In move phase, find source and destination
            source = None
            dest = None
            for r in range(5):
                for c in range(5):
                    if state[r][c] == self.my_piece and best_state[r][c] == ' ':
                        source = (r, c)
                    elif state[r][c] == ' ' and best_state[r][c] == self.my_piece:
                        dest = (r, c)
                    if source and dest:
                        return [dest, source]
            
            # Fallback: make any valid move
            for r in range(5):
                for c in range(5):
                    if state[r][c] == self.my_piece:
                        # Try all 8 directions
                        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                                return [(nr, nc), (r, c)]
        
        # Emergency fallback
        for r in range(5):
            for c in range(5):
                if state[r][c] == self.my_piece:
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 5 and 0 <= nc < 5 and state[nr][nc] == ' ':
                            return [(nr, nc), (r, c)]
        return [(0, 0)]  # Should never reach here

    def state_to_string(self, state):
        """Convert state to string for cache key"""
        return ''.join(''.join(row) for row in state)

    def heuristic_game_value(self, state):
        """Fast heuristic evaluation"""
        # Terminal state check
        val = self.game_value(state)
        if val != 0:
            return val
        
        my = self.my_piece
        opp = self.opp
        
        score = 0
        
        # Check for potential wins (horizontal, vertical, diagonal, box)
        patterns = self.get_potential_winning_patterns(state)
        
        for pattern in patterns:
            my_count = pattern.count(my)
            opp_count = pattern.count(opp)
            empty_count = pattern.count(' ')
            
            # My potential wins
            if my_count > 0 and opp_count == 0:
                if my_count == 3 and empty_count == 1:  # Near win
                    score += 0.8
                elif my_count == 2 and empty_count == 2:
                    score += 0.4
                elif my_count == 1:
                    score += 0.1
            
            # Opponent potential wins - block these
            if opp_count > 0 and my_count == 0:
                if opp_count == 3 and empty_count == 1:  # Near win for opponent
                    score -= 0.9  # High priority to block
                elif opp_count == 2 and empty_count == 2:
                    score -= 0.4
        
        # Center control
        center_positions = [(2,2), (1,2), (2,1), (2,3), (3,2)]
        for r, c in center_positions:
            if state[r][c] == my:
                score += 0.1
            elif state[r][c] == opp:
                score -= 0.1
                
        # Piece proximity - basic clustering evaluation
        my_positions = [(r, c) for r in range(5) for c in range(5) if state[r][c] == my]
        if len(my_positions) >= 2:
            min_dist = float('inf')
            for i in range(len(my_positions)):
                for j in range(i+1, len(my_positions)):
                    r1, c1 = my_positions[i]
                    r2, c2 = my_positions[j]
                    dist = abs(r1-r2) + abs(c1-c2)
                    min_dist = min(min_dist, dist)
            
            # Closer is better (scaled)
            if min_dist < float('inf'):
                score += (4 - min_dist) * 0.05
        
        return max(-0.99, min(0.99, score))
    
    def get_potential_winning_patterns(self, state):
        """Returns all possible winning patterns without creating full line lists"""
        patterns = []
        
        # Sample horizontal, vertical, diagonal, and box patterns
        # Only check a subset for performance
        
        # Horizontal samples
        for r in range(5):
            for c in range(2):
                patterns.append([state[r][c+i] for i in range(4)])
        
        # Vertical samples
        for c in range(5):
            for r in range(2):
                patterns.append([state[r+i][c] for i in range(4)])
        
        # Diagonal (↘) samples
        for r in range(2):
            for c in range(2):
                patterns.append([state[r+i][c+i] for i in range(4)])
        
        # Diagonal (↗) samples
        for r in range(3, 5):
            for c in range(2):
                patterns.append([state[r-i][c+i] for i in range(4)])
        
        # Box samples - critical for Teeko
        for r in range(4):
            for c in range(4):
                patterns.append([state[r][c], state[r][c+1], state[r+1][c], state[r+1][c+1]])
        
        return patterns
    
    def max_value(self, state, d, alpha, beta):
        """Maximizing player (AI)"""
        # Use efficient string representation
        state_key = self.state_to_string(state) + str(d)
        
        if state_key in self.cache:
            return self.cache[state_key]
        
        # Check for terminal state
        val = self.game_value(state)
        if val != 0:
            return val, state
    
        # Depth limit reached
        if d == 0:
            h_val = self.heuristic_game_value(state)
            self.cache[state_key] = (h_val, state)
            return h_val, state
        
        best_v = float('-inf')
        best_states = []

        # Generate successors efficiently
        for suc in self.succ(state, self.my_piece):
            val, _ = self.min_value(suc, d - 1, alpha, beta)
            
            if val > best_v:
                best_v = val
                best_states = [suc]
            elif val == best_v:
                best_states.append(suc)
                
            alpha = max(alpha, best_v)
            if beta <= alpha:
                break
    
        # Choose a random state from the best ones
        chosen = random.choice(best_states) if best_states else state
        self.cache[state_key] = (best_v, chosen)
        return best_v, chosen
        
    def min_value(self, state, d, alpha, beta):
        """Minimizing player (opponent)"""
        # Use efficient string representation
        state_key = self.state_to_string(state) + str(d)
        
        if state_key in self.cache:
            return self.cache[state_key]
                
        # Check for terminal state
        val = self.game_value(state)
        if val != 0:
            return val, state
        
        # Depth limit reached
        if d == 0:
            h_val = self.heuristic_game_value(state)
            self.cache[state_key] = (h_val, state)
            return h_val, state
    
        best_v = float('inf')
        best_states = []

        # Generate successors efficiently
        for suc in self.succ(state, self.opp):
            val, _ = self.max_value(suc, d - 1, alpha, beta)
            
            if val < best_v:
                best_v = val
                best_states = [suc]
            elif val == best_v:
                best_states.append(suc)
                
            beta = min(beta, best_v)
            if beta <= alpha:
                break

        # Choose a random state from the best ones
        chosen = random.choice(best_states) if best_states else state
        self.cache[state_key] = (best_v, chosen)
        return best_v, chosen
    
    def succ(self, state, piece):
        """Generate successor states efficiently"""
        successors = []
        
        # Check if we're in the drop phase - count pieces directly
        piece_count = sum(row.count('b') + row.count('r') for row in state)
        dropping = (piece_count < 8)

        if dropping:
            # Drop phase: place a piece in any empty cell
            # Optimize by starting with center and working outward
            check_order = []
            
            # Center positions first (better positions)
            center_positions = [(2,2), (1,2), (2,1), (2,3), (3,2), 
                                (1,1), (1,3), (3,1), (3,3)]
            check_order.extend(center_positions)
            
            # Then other positions
            for r in range(5):
                for c in range(5):
                    if (r, c) not in center_positions:
                        check_order.append((r, c))
            
            # Generate successors in priority order
            for row, col in check_order:
                if state[row][col] == ' ':
                    new_state = copy.deepcopy(state)
                    new_state[row][col] = piece
                    successors.append(new_state)
                    # Limit number of successors for performance
                    if len(successors) >= 12:
                        return successors
            
            return successors
        else:
            # Move phase: move any piece to an adjacent empty cell
            piece_positions = [(r, c) for r in range(5) for c in range(5) if state[r][c] == piece]
            
            # Order by center proximity
            center = (2, 2)
            piece_positions.sort(key=lambda pos: abs(pos[0]-center[0]) + abs(pos[1]-center[1]))
            
            for row, col in piece_positions:
                # Check all 8 adjacent cells
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < 5 and 0 <= new_col < 5 and state[new_row][new_col] == ' ':
                        new_state = copy.deepcopy(state)
                        new_state[row][col] = ' '
                        new_state[new_row][new_col] = piece
                        successors.append(new_state)
                        
                # Limit successors per piece for performance
                if len(successors) >= 15:
                    return successors
            
        return successors

    def opponent_move(self, move):
        """Validates the opponent's next move"""
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """Modifies the board representation using the specified move and piece"""
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """Formatted printing for the board"""
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """Checks the current board status for a win condition"""
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1
                
        # box wins
        for row in range(4):
            for i in range(4):
                if state[row][i] != ' ' and state[row][i] == state[row][i+1] == state[row+1][i] == state[row+1][i+1]:
                    return 1 if state[row][i]==self.my_piece else -1
                
        # \ diagonal wins
        for row in range(2):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row+1][i+1] == state[row+2][i+2] == state[row+3][i+3]:
                    return 1 if state[row][i]==self.my_piece else -1
                
        # / diagonal wins
        for row in range(3,5):
            for i in range(2):
                if state[row][i] != ' ' and state[row][i] == state[row-1][i+1] == state[row-2][i+2] == state[row-3][i+3]:
                    return 1 if state[row][i]==self.my_piece else -1

        return 0 # no winner yet

def main():
    print('Hello, this is Teeko AI')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:
        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:
        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                      (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")

if __name__ == "__main__":
    main()