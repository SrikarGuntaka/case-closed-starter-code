#!/usr/bin/env python3
"""
Self-play data generator for Tron agent.
Launches two local agent servers and automatically records matches for offline training.
"""

import os
import sys
import time
import argparse
import subprocess
import requests
import signal
from pathlib import Path

from judge_engine import Judge, GameResult, RandomPlayer
from case_closed_game import Direction

# Agent URLs
P1_URL = "http://localhost:5008"
P2_URL = "http://localhost:5009"

# Timeout for health checks
HEALTH_CHECK_TIMEOUT = 2.0  # Reduced from 5.0
MAX_HEALTH_CHECK_ATTEMPTS = 20  # Reduced from 30

# Constants moved outside functions for performance
DIR_TO_STR = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}


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
        # No sleep - immediate retry for maximum speed
    return False


def run_match(judge: Judge, p1_random_agent: RandomPlayer, p2_random_agent: RandomPlayer) -> GameResult:
    """Run a single match using the Judge.
    
    This is extracted from judge_engine.main() but returns the result.
    
    Args:
        judge: Initialized Judge instance (will be reused)
        p1_random_agent: Reusable RandomPlayer for player 1
        p2_random_agent: Reusable RandomPlayer for player 2
    
    Returns:
        GameResult indicating match outcome
    """
    # Reset game state and game string
    judge.game.reset()
    judge.game_str = ""
    
    # Send initial state to both players
    if not judge.send_state(1) or not judge.send_state(2):
        raise RuntimeError("Failed to send initial state")
    
    # Random moves left for p1 and p2
    p1_random = 5
    p2_random = 5
    
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
        
        # Player 1 move - reduced to 1 attempt for speed
        p1_move = judge.get_move(1, 1, p1_random)
        if p1_move:
            p1_validation = judge.handle_move(p1_move, 1, is_random=False)
            if p1_validation == "forfeit":
                judge.end_game(GameResult.AGENT2_WIN)
                return GameResult.AGENT2_WIN
            elif p1_validation:
                p1_boost = p1_validation[1]
                p1_direction = p1_validation[2]
        
        # If move failed, use random move or forfeit
        if not p1_move or not p1_validation:
            if p1_random > 0:
                p1_direction = p1_random_agent.get_best_move()
                p1_random -= 1
                p1_validation = judge.handle_move(DIR_TO_STR[p1_direction], 1, is_random=True)
                p1_boost = False
            else:
                judge.end_game(GameResult.AGENT2_WIN)
                return GameResult.AGENT2_WIN
        
        # Player 2 move - reduced to 1 attempt for speed
        p2_move = judge.get_move(2, 1, p2_random)
        if p2_move:
            p2_validation = judge.handle_move(p2_move, 2, is_random=False)
            if p2_validation == "forfeit":
                judge.end_game(GameResult.AGENT1_WIN)
                return GameResult.AGENT1_WIN
            elif p2_validation:
                p2_boost = p2_validation[1]
                p2_direction = p2_validation[2]
        
        # If move failed, use random move or forfeit
        if not p2_move or not p2_validation:
            if p2_random > 0:
                p2_direction = p2_random_agent.get_best_move()
                p2_random -= 1
                p2_validation = judge.handle_move(DIR_TO_STR[p2_direction], 2, is_random=True)
                p2_boost = False
            else:
                judge.end_game(GameResult.AGENT1_WIN)
                return GameResult.AGENT1_WIN
        
        # Execute both moves simultaneously
        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        
        # Send updated state to both players
        judge.send_state(1)
        judge.send_state(2)
        
        # Check for game end
        if result is not None:
            judge.end_game(result)
            return result
        
        # Check for max turns (safety)
        if judge.game.turns >= 500:
            judge.end_game(GameResult.DRAW)
            return GameResult.DRAW


def main():
    parser = argparse.ArgumentParser(description='Self-play data generator for Tron agent')
    parser.add_argument('--games', type=int, default=50,
                       help='Number of matches to generate')
    parser.add_argument('--wait', type=float, default=0.0,
                       help='Delay between games (seconds) - kept for compatibility but not used')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel games (currently unused, kept for future)')
    parser.add_argument('--data-dir', type=str, default='data/selfplay',
                       help='Directory to save episode data')
    parser.add_argument('--agent-script', type=str, default='agent.py',
                       help='Path to agent script')
    
    args = parser.parse_args()
    
    # Create data directory (use absolute path)
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Case A: Make directory writable (set permissions to 0o777)
    try:
        os.chmod(data_dir, 0o777)
    except Exception as e:
        print(f"Warning: Could not set directory permissions: {e}")
    
    # Verify directory is writable
    if not os.access(data_dir, os.W_OK):
        raise RuntimeError(f"Data directory is not writable: {data_dir}")
    
    # Ensure agent script exists
    if not os.path.exists(args.agent_script):
        raise FileNotFoundError(f"Agent script not found: {args.agent_script}")
    
    # Process handles for cleanup
    p1_process = None
    p2_process = None
    
    def cleanup():
        """Clean up agent processes."""
        nonlocal p1_process, p2_process
        if p1_process is not None:
            try:
                p1_process.terminate()
                p1_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    p1_process.kill()
                except ProcessLookupError:
                    pass
            p1_process = None
        
        if p2_process is not None:
            try:
                p2_process.terminate()
                p2_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    p2_process.kill()
                except ProcessLookupError:
                    pass
            p2_process = None
    
    def signal_handler(sig, frame):
        """Handle interrupt signal."""
        print("\nInterrupted! Cleaning up...")
        cleanup()
        sys.exit(1)
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Case B: Set up environment for agent processes with explicit env variables
        # Start agent processes for all games (they will persist)
        print("Starting agent processes...")
        
        # Start Player 1 agent with explicit environment
        p1_env = os.environ.copy()
        p1_env["LOG_EPISODES"] = "1"
        p1_env["DATA_DIR"] = str(data_dir)
        p1_env["PLAYER_NUMBER"] = "1"
        p1_env["PORT"] = "5008"
        p1_process = subprocess.Popen(
            [sys.executable, args.agent_script],
            env=p1_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Start Player 2 agent with explicit environment
        p2_env = os.environ.copy()
        p2_env["LOG_EPISODES"] = "1"
        p2_env["DATA_DIR"] = str(data_dir)
        p2_env["PLAYER_NUMBER"] = "2"
        p2_env["PORT"] = "5009"
        p2_process = subprocess.Popen(
            [sys.executable, args.agent_script],
            env=p2_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for both agents to be ready
        print("Waiting for agents to be ready...")
        if not wait_for_agent(P1_URL):
            raise RuntimeError(f"Player 1 agent failed to start at {P1_URL}")
        if not wait_for_agent(P2_URL):
            raise RuntimeError(f"Player 2 agent failed to start at {P2_URL}")
        print("Both agents are ready!")
        
        # Create Judge instance once and reuse it
        judge = Judge(P1_URL, P2_URL)
        
        # Check connectivity once (only needed for first game)
        if not judge.check_latency():
            raise RuntimeError(f"Failed to connect to agents")
        
        print(f"Player 1: {judge.p1_agent.agent_name} ({judge.p1_agent.participant})")
        print(f"Player 2: {judge.p2_agent.agent_name} ({judge.p2_agent.participant})")
        
        # Create reusable RandomPlayer instances
        p1_random_agent = RandomPlayer(1)
        p2_random_agent = RandomPlayer(2)
        
        # Run matches
        results = {
            GameResult.AGENT1_WIN: 0,
            GameResult.AGENT2_WIN: 0,
            GameResult.DRAW: 0
        }
        
        start_time = time.time()
        for game_num in range(1, args.games + 1):
            try:
                # Run the match (judge is reused, game state is reset inside run_match)
                result = run_match(judge, p1_random_agent, p2_random_agent)
                results[result] = results.get(result, 0) + 1
                
                # Print progress every 10 games or on last game
                if game_num % 10 == 0 or game_num == args.games:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / game_num
                    remaining = (args.games - game_num) * avg_time
                    print(f"Game {game_num}/{args.games} completed: {result.name} "
                          f"(Avg: {avg_time:.2f}s/game, Est. remaining: {remaining:.1f}s)")
                    
            except Exception as e:
                print(f"Error during game {game_num}: {e}")
                # Continue immediately without delay
                continue
        
        # Print summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("Self-play session completed!")
        print(f"{'='*60}")
        print(f"Total games: {args.games}")
        print(f"Agent 1 wins: {results.get(GameResult.AGENT1_WIN, 0)}")
        print(f"Agent 2 wins: {results.get(GameResult.AGENT2_WIN, 0)}")
        print(f"Draws: {results.get(GameResult.DRAW, 0)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per game: {total_time/args.games:.2f}s")
        print(f"Episode data directory: {data_dir}")
        
        # Count saved episode files
        try:
            episode_files = list(data_dir.glob("episode_*.npz"))
            print(f"Episode files saved: {len(episode_files)}")
            if len(episode_files) == 0:
                print("WARNING: No episode files found in data directory!")
                print("This may indicate that data logging is not working correctly.")
                print("Check that LOG_EPISODES=1 is set and agents can write to the directory.")
                # Check for log files from agents
                log_files = list(data_dir.glob("agent_log.txt"))
                if log_files:
                    print(f"Agent log files found: {[str(f) for f in log_files]}")
                    print("Check these log files for debugging information.")
                else:
                    print("No agent log files found - agents may not have LOG_EPISODES enabled.")
        except Exception as e:
            print(f"Could not count episode files: {e}")
        
    finally:
        # Clean up agent processes
        print("\nCleaning up agent processes...")
        cleanup()
        print("Done!")


if __name__ == "__main__":
    main()

