import os
import uuid
import time
import json
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from case_closed_game import Game, Direction, GameResult

# Data logging configuration
LOG_EPISODES = os.getenv("LOG_EPISODES", "0") == "1"
DATA_DIR = os.getenv("DATA_DIR", "data/episodes")
# Convert to absolute path to avoid issues with working directory
if DATA_DIR and not os.path.isabs(DATA_DIR):
    DATA_DIR = os.path.abspath(DATA_DIR)
if LOG_EPISODES:
    os.makedirs(DATA_DIR, exist_ok=True)
    # Verify directory is writable
    if not os.access(DATA_DIR, os.W_OK):
        raise RuntimeError(f"Data directory is not writable: {DATA_DIR}")
    # Create a log file for debugging
    LOG_FILE = os.path.join(DATA_DIR, "agent_log.txt")
    def log_debug(msg):
        try:
            with open(LOG_FILE, "a") as f:
                import datetime
                f.write(f"[{datetime.datetime.now()}] {msg}\n")
                f.flush()
        except:
            pass
    # Log initialization
    import datetime
    player_num = os.getenv("PLAYER_NUMBER", "?")
    log_debug(f"Agent initialized: LOG_EPISODES=True, DATA_DIR={DATA_DIR}, PLAYER_NUMBER={player_num}")
else:
    def log_debug(msg):
        pass

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

# Global model instance (loaded once)
MODEL = None

# Episode buffers for data logging
EP_STATES = []
EP_ACTIONS = []
EP_SAFE = []
EP_TURNS = []
EP_META = {}


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    # Log state receipt for debugging (only first few to avoid spam)
    if LOG_EPISODES and data.get("turn_count", 0) < 3:
        log_debug(f"Received state: turn={data.get('turn_count', '?')}, has_board={'board' in data}")
    return jsonify({"status": "state received"}), 200


def encode_state_numpy(state: dict, player_number: int):
    """
    Return a numpy array of shape (6, 18, 20) for data logging.
    Works without PyTorch.

    Channels:
      0: empty cells (1 if empty)
      1: my trail
      2: opponent trail
      3: my head (one-hot)
      4: opponent head (one-hot)
      5: walls/occupied (1 if nonempty)
    """
    my_id = player_number
    opp_id = 2 if my_id == 1 else 1
    
    board = state.get("board", [])
    if not board:
        return None
    
    H, W = len(board), len(board[0]) if board else 0
    if H != 18 or W != 20:
        return None
    
    # Get trails
    my_trail_key = f"agent{my_id}_trail"
    opp_trail_key = f"agent{opp_id}_trail"
    my_trail = state.get(my_trail_key, [])
    opp_trail = state.get(opp_trail_key, [])
    
    # Initialize channels: (6, 18, 20)
    channels = np.zeros((6, H, W), dtype=np.float32)
    
    # Convert board to numpy array for fast operations
    board_arr = np.array(board, dtype=np.int32)
    
    # Channel 0: empty cells (1 if empty, 0 otherwise)
    channels[0] = (board_arr == 0).astype(np.float32)
    
    # Channel 5: walls/occupied (1 if nonempty, 0 otherwise)
    channels[5] = (board_arr != 0).astype(np.float32)
    
    # Channel 1: my trail (all positions including head)
    if my_trail:
        my_trail_arr = np.array(my_trail, dtype=np.int32)
        if len(my_trail_arr) > 0:
            # Use numpy advanced indexing to set all trail positions at once
            # Trail positions are (x, y) but numpy indexing is (y, x)
            x_coords = my_trail_arr[:, 0]
            y_coords = my_trail_arr[:, 1]
            # Wrap coordinates for torus
            x_coords = np.mod(x_coords, W)
            y_coords = np.mod(y_coords, H)
            channels[1, y_coords, x_coords] = 1.0
    
    # Channel 2: opponent trail (all positions including head)
    if opp_trail:
        opp_trail_arr = np.array(opp_trail, dtype=np.int32)
        if len(opp_trail_arr) > 0:
            # Use numpy advanced indexing to set all trail positions at once
            x_coords = opp_trail_arr[:, 0]
            y_coords = opp_trail_arr[:, 1]
            # Wrap coordinates for torus
            x_coords = np.mod(x_coords, W)
            y_coords = np.mod(y_coords, H)
            channels[2, y_coords, x_coords] = 1.0
    
    # Channel 3: my head (one-hot)
    if my_trail:
        my_head = tuple(my_trail[-1])
        x, y = int(my_head[0]) % W, int(my_head[1]) % H
        channels[3, y, x] = 1.0
    
    # Channel 4: opponent head (one-hot)
    if opp_trail:
        opp_head = tuple(opp_trail[-1])
        x, y = int(opp_head[0]) % W, int(opp_head[1]) % H
        channels[4, y, x] = 1.0
    
    return channels


def encode_state(state: dict, player_number: int):
    """
    Return a CPU torch.FloatTensor of shape (1, C, 18, 20).

    Channels:
      0: empty cells (1 if empty)
      1: my trail
      2: opponent trail
      3: my head (one-hot)
      4: opponent head (one-hot)
      5: walls/occupied (1 if nonempty)

    Uses numpy for fast fills; torch.from_numpy at the end.
    """
    try:
        import torch
    except ImportError:
        return None
    
    # Use the numpy encoding function
    channels = encode_state_numpy(state, player_number)
    if channels is None:
        return None
    
    # Convert to torch tensor and add batch dimension
    tensor = torch.from_numpy(channels).unsqueeze(0)  # (1, 6, 18, 20)
    return tensor


def get_safe_moves(board, head_xy, W, H):
    """Return list of safe dirs among ['UP','DOWN','LEFT','RIGHT'] using torus wrap."""
    def wrap_add(x, dx, size):
        x += dx
        if x < 0:
            x += size
        elif x >= size:
            x -= size
        return x
    
    safe_dirs = []
    directions = [
        ("UP", (0, -1)),
        ("DOWN", (0, 1)),
        ("LEFT", (-1, 0)),
        ("RIGHT", (1, 0))
    ]
    
    x, y = head_xy
    for dir_name, (dx, dy) in directions:
        nx = wrap_add(x, dx, W)
        ny = wrap_add(y, dy, H)
        # Check if cell is blocked (nonzero)
        if board[ny][nx] == 0:
            safe_dirs.append(dir_name)
    
    return safe_dirs


def safe_mask_from_moves(safe_moves):
    """Convert list of safe move strings to a binary mask array."""
    idx = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    m = np.zeros(4, dtype=np.int8)
    for d in safe_moves:
        if d in idx:
            m[idx[d]] = 1
    return m


if torch is not None and nn is not None:
    class TronNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 18 * 20, 128), nn.ReLU(),
                nn.Linear(128, 4)  # UP, DOWN, LEFT, RIGHT
            )
        
        def forward(self, x):
            return self.net(x)
else:
    TronNet = None


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
   
    # Encode state to tensor (no-op if torch unavailable)
    torch_available = False
    try:
        import torch
        torch_available = True
        state_tensor = encode_state(state, player_number)
    except Exception:
        state_tensor = None
    
    # Load and initialize model once
    global MODEL
    if MODEL is None and torch_available:
        if TronNet is not None:
            try:
                model_path = os.getenv("MODEL_PATH", "model.pt")
                MODEL = TronNet()
                MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
                MODEL.eval()
            except FileNotFoundError:
                MODEL = None
            except Exception:
                MODEL = None
    
    # --- YOUR CODE GOES HERE ---
    # Hybrid control: rules + PyTorch policy
    board = state.get("board", [])
    safe_moves = []  # Initialize for data logging
    
    if not board:
        move = "UP"
    else:
        H, W = len(board), len(board[0]) if board else 0
        occ = {(x, y) for y in range(H) for x in range(W) if board[y][x] != 0}
        my_head = tuple(my_agent.trail[-1])
        cur_dir = my_agent.direction.name if hasattr(my_agent.direction, 'name') else None
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        opp_head = tuple(opp_agent.trail[-1]) if opp_agent.trail else None
        turn_count = state.get("turn_count", 0)
        boosts_remaining = my_agent.boosts_remaining
        
        # Get safe moves first
        safe_moves = get_safe_moves(board, my_head, W, H)
        if not safe_moves:
            move = "UP"
        else:
            # Helper functions with bound methods for performance
            def wrap_add(x, dx, size):
                x += dx
                if x < 0:
                    x += size
                elif x >= size:
                    x -= size
                return x
            
            neighbors = ((0, -1), (0, 1), (1, 0), (-1, 0))
            
            def flood_count(start, occ_set, W, H):
                if start in occ_set:
                    return 0
                occ_has = occ_set.__contains__
                seen = set()
                seen_add = seen.add
                dq = deque([start])
                dq_append = dq.append
                dq_popleft = dq.popleft
                count = 0
                while dq:
                    x, y = dq_popleft()
                    if (x, y) in seen:
                        continue
                    seen_add((x, y))
                    count += 1
                    for dx, dy in neighbors:
                        nx = wrap_add(x, dx, W)
                        ny = wrap_add(y, dy, H)
                        np = (nx, ny)
                        if np not in seen and not occ_has(np):
                            dq_append(np)
                return count
            
            # Get flood-fill score for a specific move direction
            def get_move_score(move_dir, my_head, occ_set, W, H):
                dir_to_delta = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
                if move_dir not in dir_to_delta:
                    return 0
                dx, dy = dir_to_delta[move_dir]
                nx = wrap_add(my_head[0], dx, W)
                ny = wrap_add(my_head[1], dy, H)
                next_pos = (nx, ny)
                if (nx, ny) in occ_set:
                    return 0
                return flood_count(next_pos, occ_set, W, H)
            
            # Rule-based move selection (best_move_by_space)
            def best_move_by_space(my_head, occ_set, W, H, safe_dirs, prefer_dir=None):
                best_size = -1
                best_moves = []
                dirs = [("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0))]
                occ_has = occ_set.__contains__
                for d, (dx, dy) in dirs:
                    if d not in safe_dirs:
                        continue
                    nx = wrap_add(my_head[0], dx, W)
                    ny = wrap_add(my_head[1], dy, H)
                    next_pos = (nx, ny)
                    if occ_has(next_pos):
                        continue
                    size = flood_count(next_pos, occ_set, W, H)
                    if size > best_size:
                        best_size = size
                        best_moves = [d]
                    elif size == best_size:
                        best_moves.append(d)
                if prefer_dir and prefer_dir in best_moves:
                    return prefer_dir, best_size
                return (best_moves[0] if best_moves else safe_dirs[0] if safe_dirs else "UP"), best_size
            
            # Rule move
            rule_move, rule_score = best_move_by_space(my_head, occ, W, H, safe_moves, cur_dir)
            
            # Model inference (masked to safe moves only)
            model_move = None
            if MODEL is not None and state_tensor is not None and torch_available:
                try:
                    with torch.no_grad():
                        logits = MODEL(state_tensor)[0]  # Shape [4]
                        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
                        safe_set = set(safe_moves)
                        # Mask unsafe moves
                        for i, move_name in enumerate(moves):
                            if move_name not in safe_set:
                                logits[i] = -1e9
                        # Get model move
                        model_idx = torch.argmax(logits).item()
                        if moves[model_idx] in safe_set:
                            model_move = moves[model_idx]
                except Exception:
                    model_move = None
            
            # Blend model and rule moves
            MARGIN = 3
            if model_move is None:
                move = rule_move
            else:
                model_score = get_move_score(model_move, my_head, occ, W, H)
                # Use model move only if it's within MARGIN of rule move score
                if model_score >= rule_score - MARGIN:
                    move = model_move
                else:
                    move = rule_move
            
            # If current direction is safe and scores are close, prefer it for consistency
            if cur_dir and cur_dir in safe_moves:
                cur_score = get_move_score(cur_dir, my_head, occ, W, H)
                # Prefer current direction if its score is within MARGIN of the chosen move's score
                chosen_score = get_move_score(move, my_head, occ, W, H)
                if cur_score >= chosen_score - MARGIN:
                    move = cur_dir
            
            # Deterministic fallback: ensure move is safe
            if move not in safe_moves:
                move = safe_moves[0] if safe_moves else "UP"
    
    # Data logging: log state, action, safe mask, and turn
    # Always log if LOG_EPISODES is enabled, using numpy encoding (works without PyTorch)
    # This is OUTSIDE the board check so it always runs when state is available
    if LOG_EPISODES:
        try:
            # Ensure state has required fields
            if not state or "board" not in state:
                log_debug(f"Warning: State missing board in send_move, player={player_number}, state keys: {list(state.keys()) if state else 'None'}")
            else:
                # Encode state as numpy array for logging (works without PyTorch)
                state_array = encode_state_numpy(state, player_number)
                if state_array is not None:
                    EP_STATES.append(state_array)  # (6,18,20)
                    action_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
                    move_base = move.split(":")[0]  # Remove :BOOST if present
                    EP_ACTIONS.append(action_map.get(move_base, 0))
                    EP_SAFE.append(safe_mask_from_moves(safe_moves))
                    EP_TURNS.append(state.get("turn_count", 0))
                else:
                    log_debug(f"Warning: encode_state_numpy returned None, player={player_number}, turn={state.get('turn_count', '?')}, board shape: {len(state.get('board', []))}")
        except Exception as e:
            # Log error to file for debugging
            import traceback
            log_debug(f"Error in data logging: {e}\n{traceback.format_exc()}")
    # --- END YOUR CODE ---

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    
    # Data logging: save episode data if logging is enabled
    if LOG_EPISODES:
        try:
            res = data.get("result", "DRAW") if data else "DRAW"
            # Map to +1/0/-1 for this process
            me = int(os.getenv("PLAYER_NUMBER", "1"))
            win = (res == "AGENT1_WIN" and me == 1) or (res == "AGENT2_WIN" and me == 2)
            lose = (res == "AGENT1_WIN" and me == 2) or (res == "AGENT2_WIN" and me == 1)
            reward = 1 if win else (-1 if lose else 0)
            EP_META.update({"result": res, "reward": reward})
            
            log_debug(f"End game called: player={me}, result={res}, EP_STATES len={len(EP_STATES)}")
            
            # Persist NPZ if we have data
            if EP_STATES:
                # Ensure DATA_DIR exists
                os.makedirs(DATA_DIR, exist_ok=True)
                
                ts = int(time.time() * 1000)
                out = os.path.join(DATA_DIR, f"episode_{ts}_p{me}.npz")
                
                # Save the data
                np.savez_compressed(
                    out,
                    states=np.stack(EP_STATES, 0),  # (T,6,18,20)
                    actions=np.array(EP_ACTIONS, dtype=np.int64),
                    safe=np.stack(EP_SAFE, 0),  # (T,4)
                    turns=np.array(EP_TURNS, dtype=np.int32),
                    meta=np.frombuffer(json.dumps(EP_META).encode("utf-8"), dtype=np.uint8),
                )
                
                log_debug(f"Saved episode data: {out} ({len(EP_STATES)} steps, {len(EP_ACTIONS)} actions)")
            else:
                # Log warning if no data was collected
                log_debug(f"WARNING: No episode data to save (EP_STATES is empty) for player {me}")
                log_debug(f"State keys available: {list(data.keys()) if data else 'None'}")
            
            # Clear buffers (always clear, even if no data was saved)
            EP_STATES.clear()
            EP_ACTIONS.clear()
            EP_SAFE.clear()
            EP_TURNS.clear()
            EP_META.clear()
        except Exception as e:
            # Log error to file for debugging
            import traceback
            log_debug(f"Error saving episode data: {e}\n{traceback.format_exc()}")
            # Clear buffers even on error to prevent memory leaks
            EP_STATES.clear()
            EP_ACTIONS.clear()
            EP_SAFE.clear()
            EP_TURNS.clear()
            EP_META.clear()
    
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
