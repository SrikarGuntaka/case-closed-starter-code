#!/usr/bin/env python3
"""
Evaluation and benchmarking script for Tron agent.
Systematically tests trained agents against baselines and collects metrics.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import requests
import signal
from pathlib import Path
from typing import Dict, List

from judge_engine import Judge, GameResult

# Agent URLs
P1_URL = "http://localhost:5008"
P2_URL = "http://localhost:5009"

# Timeout for health checks
HEALTH_CHECK_TIMEOUT = 5.0
MAX_HEALTH_CHECK_ATTEMPTS = 30


def wait_for_agent(url: str, timeout: float = HEALTH_CHECK_TIMEOUT, max_attempts: int = MAX_HEALTH_CHECK_ATTEMPTS) -> bool:
    """Wait for an agent to be ready by checking the health endpoint.
    
    Args:
        url: Agent URL to check
        timeout: Timeout per request
        max_attempts: Maximum number of attempts
    
    Returns:
        True if agent is ready, False otherwise
    """
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True
        except (requests.RequestException, requests.Timeout):
            pass
        time.sleep(0.5)
    return False


def run_match(judge: Judge) -> tuple:
    """Run a single match using the Judge.
    
    This is extracted from judge_engine.main() but returns the result and latency data.
    
    Args:
        judge: Initialized Judge instance
    
    Returns:
        Tuple of (GameResult, p1_latencies, p2_latencies) where latencies are lists in seconds
    """
    from case_closed_game import Direction
    from judge_engine import RandomPlayer
    import requests
    from judge_engine import TIMEOUT
    
    # Latency tracking
    p1_latencies = []
    p2_latencies = []
    
    # Send initial state to both players
    if not judge.send_state(1) or not judge.send_state(2):
        raise RuntimeError("Failed to send initial state")
    
    # Random moves left for p1 and p2
    p1_random = 5
    p2_random = 5
    
    # Direction to string mapping
    dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
    
    # Game loop
    while True:
        # Get moves from both players
        p1_move = None
        p2_move = None
        p1_boost = False
        p2_boost = False
        p1_direction = None
        p2_direction = None
        p1_validation = None
        p2_validation = None
        
        # Player 1 move (with latency tracking)
        for attempt in range(1, 3):  # 2 attempts
            # Track latency
            url = judge.p1_url
            params = {
                "player_number": 1,
                "attempt_number": attempt,
                "random_moves_left": p1_random,
                "turn_count": judge.game.turns,
            }
            try:
                start_time = time.time()
                response = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
                end_time = time.time()
                latency = end_time - start_time
                p1_latencies.append(latency)
                
                if response.status_code == 200:
                    move_data = response.json()
                    p1_move = move_data.get('move')
                    if p1_move:
                        p1_validation = judge.handle_move(p1_move, 1, is_random=False)
                        if p1_validation == "forfeit":
                            judge.end_game(GameResult.AGENT2_WIN)
                            return (GameResult.AGENT2_WIN, p1_latencies, p2_latencies)
                        elif p1_validation:
                            p1_boost = p1_validation[1]  # Extract boost flag
                            p1_direction = p1_validation[2]  # Extract direction
                            break
            except (requests.RequestException, requests.Timeout):
                pass
        
        # If both attempts failed, use random move or forfeit
        if not p1_move or not p1_validation:
            if p1_random > 0:
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                p1_validation = judge.handle_move(dir_to_str[p1_direction], 1, is_random=True)
                p1_boost = False
            else:
                judge.end_game(GameResult.AGENT2_WIN)
                return (GameResult.AGENT2_WIN, p1_latencies, p2_latencies)
        
        # Player 2 move (with latency tracking)
        for attempt in range(1, 3):  # 2 attempts
            # Track latency
            url = judge.p2_url
            params = {
                "player_number": 2,
                "attempt_number": attempt,
                "random_moves_left": p2_random,
                "turn_count": judge.game.turns,
            }
            try:
                start_time = time.time()
                response = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
                end_time = time.time()
                latency = end_time - start_time
                p2_latencies.append(latency)
                
                if response.status_code == 200:
                    move_data = response.json()
                    p2_move = move_data.get('move')
                    if p2_move:
                        p2_validation = judge.handle_move(p2_move, 2, is_random=False)
                        if p2_validation == "forfeit":
                            judge.end_game(GameResult.AGENT1_WIN)
                            return (GameResult.AGENT1_WIN, p1_latencies, p2_latencies)
                        elif p2_validation:
                            p2_boost = p2_validation[1]  # Extract boost flag
                            p2_direction = p2_validation[2]  # Extract direction
                            break
            except (requests.RequestException, requests.Timeout):
                pass
        
        # If both attempts failed, use random move or forfeit
        if not p2_move or not p2_validation:
            if p2_random > 0:
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                p2_validation = judge.handle_move(dir_to_str[p2_direction], 2, is_random=True)
                p2_boost = False
            else:
                judge.end_game(GameResult.AGENT1_WIN)
                return (GameResult.AGENT1_WIN, p1_latencies, p2_latencies)
        
        # Execute both moves simultaneously
        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        
        # Send updated state to both players
        judge.send_state(1)
        judge.send_state(2)
        
        # Check for game end
        if result is not None:
            judge.end_game(result)
            return (result, p1_latencies, p2_latencies)
        
        # Check for max turns (safety)
        if judge.game.turns >= 500:
            judge.end_game(GameResult.DRAW)
            return (GameResult.DRAW, p1_latencies, p2_latencies)


def cleanup_processes(p1_process: subprocess.Popen, p2_process: subprocess.Popen):
    """Clean up agent processes.
    
    Args:
        p1_process: Player 1 process handle
        p2_process: Player 2 process handle
    """
    if p1_process is not None:
        try:
            p1_process.terminate()
            p1_process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                p1_process.kill()
            except ProcessLookupError:
                pass
    
    if p2_process is not None:
        try:
            p2_process.terminate()
            p2_process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                p2_process.kill()
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained agent against baseline')
    parser.add_argument('--games', type=int, default=50,
                       help='Number of games to run')
    parser.add_argument('--trained', type=str, default='agent.py',
                       help='Path to trained agent script')
    parser.add_argument('--baseline', type=str, default='sample_agent.py',
                       help='Path to baseline agent script')
    parser.add_argument('--model', type=str, default='model.pt',
                       help='Path to trained model file')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Path to save results JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress per-game output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.trained):
        raise FileNotFoundError(f"Trained agent script not found: {args.trained}")
    if not os.path.exists(args.baseline):
        raise FileNotFoundError(f"Baseline agent script not found: {args.baseline}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Metrics storage
    metrics = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "turns": [],
        "p1_latency": [],
        "p2_latency": []
    }
    
    # Process handles
    p1_process = None
    p2_process = None
    
    def signal_handler(sig, frame):
        """Handle interrupt signal."""
        print("\nInterrupted! Cleaning up...")
        cleanup_processes(p1_process, p2_process)
        sys.exit(1)
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Run matches
        for game_num in range(1, args.games + 1):
            if not args.quiet:
                print(f"\n{'='*60}")
                print(f"Game {game_num}/{args.games}")
                print(f"{'='*60}")
            
            try:
                # Set up environment for trained agent (Player 1)
                p1_env = os.environ.copy()
                p1_env['PORT'] = '5008'
                p1_env['PLAYER_NUMBER'] = '1'
                p1_env['MODEL_PATH'] = args.model
                p1_env['LOG_EPISODES'] = '0'  # Disable logging during evaluation
                
                # Set up environment for baseline agent (Player 2)
                p2_env = os.environ.copy()
                p2_env['PORT'] = '5009'
                p2_env['PLAYER_NUMBER'] = '2'
                p2_env['LOG_EPISODES'] = '0'  # Disable logging during evaluation
                
                # Start Player 1 (trained agent)
                p1_process = subprocess.Popen(
                    [sys.executable, args.trained],
                    env=p1_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Start Player 2 (baseline agent)
                p2_process = subprocess.Popen(
                    [sys.executable, args.baseline],
                    env=p2_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Wait for both agents to be ready
                if not args.quiet:
                    print("Waiting for agents to be ready...")
                if not wait_for_agent(P1_URL):
                    raise RuntimeError(f"Player 1 agent failed to start at {P1_URL}")
                if not wait_for_agent(P2_URL):
                    raise RuntimeError(f"Player 2 agent failed to start at {P2_URL}")
                
                # Create judge and check connectivity
                judge = Judge(P1_URL, P2_URL)
                if not judge.check_latency():
                    raise RuntimeError("Failed to connect to agents")
                
                # Run the match
                result, p1_latencies, p2_latencies = run_match(judge)
                
                # Record metrics
                if result == GameResult.AGENT1_WIN:
                    metrics["wins"] += 1
                elif result == GameResult.AGENT2_WIN:
                    metrics["losses"] += 1
                else:
                    metrics["draws"] += 1
                
                metrics["turns"].append(judge.game.turns)
                
                # Record latency (average over all moves in the match)
                if p1_latencies:
                    avg_p1_latency = sum(p1_latencies) / len(p1_latencies)
                    metrics["p1_latency"].append(avg_p1_latency * 1000)  # Convert to ms
                if p2_latencies:
                    avg_p2_latency = sum(p2_latencies) / len(p2_latencies)
                    metrics["p2_latency"].append(avg_p2_latency * 1000)  # Convert to ms
                
                if not args.quiet:
                    print(f"Game {game_num} completed: {result.name} (turns: {judge.game.turns})")
                
                # Clean up processes
                cleanup_processes(p1_process, p2_process)
                p1_process = None
                p2_process = None
                
                # Wait before next game (except after last game)
                if game_num < args.games:
                    time.sleep(2)
                    
            except Exception as e:
                print(f"Error during game {game_num}: {e}")
                # Clean up processes on error
                cleanup_processes(p1_process, p2_process)
                p1_process = None
                p2_process = None
                # Continue to next game
                if game_num < args.games:
                    time.sleep(2)
                continue
        
        # Calculate summary statistics
        total_games = metrics["wins"] + metrics["losses"] + metrics["draws"]
        avg_turns = sum(metrics["turns"]) / len(metrics["turns"]) if metrics["turns"] else 0
        avg_p1_latency = sum(metrics["p1_latency"]) / len(metrics["p1_latency"]) if metrics["p1_latency"] else 0
        avg_p2_latency = sum(metrics["p2_latency"]) / len(metrics["p2_latency"]) if metrics["p2_latency"] else 0
        
        # Print summary
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"Trained agent: {args.trained}")
        print(f"Baseline agent: {args.baseline}")
        print(f"Model: {args.model}")
        print(f"\nTotal games: {total_games}")
        print(f"Wins: {metrics['wins']} ({metrics['wins']/total_games*100:.1f}%)" if total_games > 0 else "Wins: 0")
        print(f"Losses: {metrics['losses']} ({metrics['losses']/total_games*100:.1f}%)" if total_games > 0 else "Losses: 0")
        print(f"Draws: {metrics['draws']} ({metrics['draws']/total_games*100:.1f}%)" if total_games > 0 else "Draws: 0")
        print(f"\nAverage turns: {avg_turns:.1f}")
        print(f"Average latency (Trained): {avg_p1_latency:.2f} ms")
        print(f"Average latency (Baseline): {avg_p2_latency:.2f} ms")
        print(f"{'='*60}")
        
        # Save results to JSON
        results = {
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "draws": metrics["draws"],
            "total_games": total_games,
            "win_rate": metrics["wins"] / total_games if total_games > 0 else 0,
            "avg_turns": avg_turns,
            "avg_latency_ms": avg_p1_latency,
            "avg_latency_baseline_ms": avg_p2_latency,
            "trained_agent": args.trained,
            "baseline_agent": args.baseline,
            "model": args.model
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    finally:
        # Final cleanup
        cleanup_processes(p1_process, p2_process)


if __name__ == '__main__':
    main()

