"""
Sample agent for Case Closed Challenge - Works with Judge Protocol
This agent runs as a Flask server and responds to judge requests.
"""

import os
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "SampleParticipant")
AGENT_NAME = os.getenv("AGENT_NAME", "SampleAgent")


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
    return jsonify({"status": "state received"}), 200


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
   
    # --- YOUR CODE GOES HERE ---
    board = state.get("board", [])
    if not board:
        move = "UP"
    else:
        H, W = len(board), len(board[0]) if board else 0
        occ = {(x, y) for y in range(H) for x in range(W) if board[y][x] != 0}
        my_head = tuple(my_agent.trail[-1])
        cur_dir = my_agent.direction.name
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        opp_head = tuple(opp_agent.trail[-1]) if opp_agent.trail else None
        dirs = [("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0))]
        neighbors = ((0, -1), (0, 1), (1, 0), (-1, 0))  # Tuple for efficiency, reused across functions
        
        # Wrap function optimized (avoid %)
        def wrap_add(x, dx, W):
            x += dx
            if x < 0:
                x += W
            elif x >= W:
                x -= W
            return x
        
        # Flood count with bound methods for performance
        def flood_count(start, occ_set, W, H, cutoff=None):
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
                if cutoff and count > cutoff:
                    return count
                for dx, dy in neighbors:
                    nx = wrap_add(x, dx, W)
                    ny = wrap_add(y, dy, H)
                    np = (nx, ny)
                    if np not in seen and not occ_has(np):
                        dq_append(np)
            return count
        
        # Check if two points are in same component (early-exit BFS)
        def same_component(a, b, occ_set, W, H):
            if a in occ_set or b in occ_set:
                return False
            if a == b:
                return True
            occ_has = occ_set.__contains__
            seen = set()
            seen_add = seen.add
            dq = deque([a])
            dq_append = dq.append
            dq_popleft = dq.popleft
            while dq:
                x, y = dq_popleft()
                if (x, y) in seen:
                    continue
                seen_add((x, y))
                for dx, dy in neighbors:
                    nx = wrap_add(x, dx, W)
                    ny = wrap_add(y, dy, H)
                    np = (nx, ny)
                    if np == b:
                        return True
                    if np not in seen and not occ_has(np):
                        dq_append(np)
            return False
        
        # Best move by maximizing my space
        def best_move_by_space(my_head, occ_set, W, H, dirs, cur_dir):
            best_size = -1
            best_moves = []
            occ_has = occ_set.__contains__
            for d, (dx, dy) in dirs:
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
            if cur_dir in best_moves:
                return cur_dir
            return best_moves[0] if best_moves else "UP"
        
        # Minimax-style move selection for same component
        def maximin_move(my_head, opp_head, occ, W, H, dirs, cur_dir):
            best_score = float('-inf')
            best_moves = []
            my_size_before = flood_count(my_head, occ, W, H)
            opp_size_before = flood_count(opp_head, occ, W, H)
            occ_has = occ.__contains__
            for md, (mdx, mdy) in dirs:
                nx = wrap_add(my_head[0], mdx, W)
                ny = wrap_add(my_head[1], mdy, H)
                my_next = (nx, ny)
                if occ_has(my_next):
                    continue
                occ_after_me = {my_next} | occ
                my_size_after = flood_count(my_next, occ_after_me, W, H)
                opp_size_after = flood_count(opp_head, occ_after_me, W, H)
                seal_bonus = 0.0
                if my_size_after >= my_size_before * 0.9:
                    if opp_size_after < opp_size_before * 0.7:
                        seal_bonus = (opp_size_before - opp_size_after) * 0.5
                min_score = float('inf')
                opp_valid = False
                occ_after_me_has = occ_after_me.__contains__
                for od, (odx, ody) in dirs:
                    ox = wrap_add(opp_head[0], odx, W)
                    oy = wrap_add(opp_head[1], ody, H)
                    opp_next = (ox, oy)
                    if occ_after_me_has(opp_next):
                        continue
                    opp_valid = True
                    occ_both = {opp_next} | occ_after_me
                    my_reachable = flood_count(my_next, occ_both, W, H)
                    opp_reachable = flood_count(opp_next, occ_both, W, H)
                    score = my_reachable - opp_reachable
                    if score < min_score:
                        min_score = score
                if not opp_valid:
                    min_score = my_size_after
                final_score = min_score + seal_bonus
                if final_score > best_score:
                    best_score = final_score
                    best_moves = [md]
                elif final_score == best_score:
                    best_moves.append(md)
            if cur_dir in best_moves:
                return cur_dir
            return best_moves[0] if best_moves else "UP"
        
        # Choose base direction with region analysis
        if not opp_head:
            base_dir = best_move_by_space(my_head, occ, W, H, dirs, cur_dir)
        else:
            in_same_comp = same_component(my_head, opp_head, occ, W, H)
            if not in_same_comp:
                base_dir = best_move_by_space(my_head, occ, W, H, dirs, cur_dir)
            else:
                base_dir = maximin_move(my_head, opp_head, occ, W, H, dirs, cur_dir)
        
        # Boost decision logic
        boosts_remaining = my_agent.boosts_remaining
        turn_count = state.get("turn_count", 0)
        move = base_dir
        
        if boosts_remaining > 0:
            dir_to_delta = {d: (dx, dy) for d, (dx, dy) in dirs}
            if base_dir in dir_to_delta:
                dx, dy = dir_to_delta[base_dir]
                step1 = (wrap_add(my_head[0], dx, W), wrap_add(my_head[1], dy, H))
                step2 = (wrap_add(step1[0], dx, W), wrap_add(step1[1], dy, H))
                occ_has = occ.__contains__
                
                if not occ_has(step1) and not occ_has(step2):
                    space_after_one = flood_count(step1, occ, W, H)
                    occ_after_one = {step1} | occ
                    space_after_two = flood_count(step2, occ_after_one, W, H)
                    
                    CRAMPED = 15
                    is_early = turn_count < 50
                    is_late = turn_count > 120
                    should_boost = False
                    
                    if space_after_two > space_after_one:
                        should_boost = True
                    elif space_after_one < CRAMPED:
                        should_boost = True
                    
                    if opp_head and space_after_one >= CRAMPED:
                        occ_after_two = {step2} | occ_after_one
                        opp_size_before = flood_count(opp_head, occ, W, H)
                        opp_size_after_two = flood_count(opp_head, occ_after_two, W, H)
                        if opp_size_after_two < opp_size_before - 8:
                            should_boost = True
                    
                    if should_boost:
                        if is_early and boosts_remaining >= 2:
                            if space_after_one < CRAMPED or space_after_two > space_after_one + 3:
                                move = f"{base_dir}:BOOST"
                        elif is_late or space_after_one < 12:
                            if boosts_remaining > 1 or space_after_one < CRAMPED:
                                move = f"{base_dir}:BOOST"
                        elif boosts_remaining > 1 and space_after_two > space_after_one + 2:
                            move = f"{base_dir}:BOOST"
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
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    # For development only. Port can be overridden with the PORT env var.
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
