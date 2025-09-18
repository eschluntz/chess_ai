import csv
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import chess

from core.eval import (
    linear_1m_alpha10_no_mates_eval,
    linear_1m_alpha10_with_mates_eval,
    piece_position_eval,
    piece_value_eval,
    rf_1m_estimators500_depth25_no_mates_eval,
    rf_330k_estimators1000_unlimited_no_mates_eval,
)
from core.search import negamax


class AIPlayer:
    def __init__(self, name, eval_func, depth):
        self.name = name
        self.eval_func = eval_func
        self.depth = depth

    def get_move(self, board):
        _, move = negamax(board, self.eval_func, self.depth)
        return move

    def __str__(self):
        return f"{self.name}_depth{self.depth}"


def create_ai_players():
    players = []

    # Classical evaluators with depths 2-5
    for depth in range(2, 6):
        players.append(AIPlayer("piece_value", piece_value_eval, depth))
        players.append(AIPlayer("piece_position", piece_position_eval, depth))

    # # RF estimators with depth 1 only
    # players.append(
    #     AIPlayer("rf_500trees", rf_1m_estimators500_depth25_no_mates_eval, 1)
    # )
    # players.append(
    #     AIPlayer("rf_1000trees", rf_330k_estimators1000_unlimited_no_mates_eval, 1)
    # )

    # Linear models with depths 1-5
    for depth in range(1, 6):
        players.append(
            AIPlayer("linear_no_mates", linear_1m_alpha10_no_mates_eval, depth)
        )
        players.append(
            AIPlayer("linear_with_mates", linear_1m_alpha10_with_mates_eval, depth)
        )

    return players


def play_game(white_player, black_player):
    board = chess.Board()
    start_time = time.time()
    move_count = 0
    white_move_times = []
    black_move_times = []

    while not board.is_game_over():
        move_start = time.time()

        if board.turn:  # White's turn
            move = white_player.get_move(board)
            move_end = time.time()
            white_move_times.append(move_end - move_start)
        else:  # Black's turn
            move = black_player.get_move(board)
            move_end = time.time()
            black_move_times.append(move_end - move_start)

        board.push(move)
        move_count += 1

    end_time = time.time()
    game_duration = end_time - start_time

    result = board.result()
    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = "draw"

    white_avg_time = (
        round(sum(white_move_times) / len(white_move_times), 3)
        if white_move_times
        else 0
    )
    black_avg_time = (
        round(sum(black_move_times) / len(black_move_times), 3)
        if black_move_times
        else 0
    )

    return {
        "white_player": str(white_player),
        "black_player": str(black_player),
        "winner": winner,
        "move_count": move_count,
        "duration_seconds": round(game_duration, 2),
        "white_avg_move_time": white_avg_time,
        "black_avg_move_time": black_avg_time,
        "timestamp": datetime.now().isoformat(),
    }


# Thread lock for CSV writing
csv_lock = threading.Lock()


def save_result_to_csv(result, filename="tournament_results.csv"):
    with csv_lock:
        file_exists = True
        try:
            with open(filename, "r"):
                pass
        except FileNotFoundError:
            file_exists = False

        with open(filename, "a", newline="") as csvfile:
            fieldnames = [
                "white_player",
                "black_player",
                "winner",
                "move_count",
                "duration_seconds",
                "white_avg_move_time",
                "black_avg_move_time",
                "timestamp",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(result)


def load_completed_games(filename="tournament_results.csv"):
    completed_games = set()
    try:
        with open(filename, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                game_key = (row["white_player"], row["black_player"])
                completed_games.add(game_key)
    except FileNotFoundError:
        pass
    return completed_games


def play_game_with_params(
    white_name, white_eval_name, white_depth, black_name, black_eval_name, black_depth
):
    # Map eval function names to actual functions
    eval_map = {
        "piece_value": piece_value_eval,
        "piece_position": piece_position_eval,
        "rf_500trees": rf_1m_estimators500_depth25_no_mates_eval,
        "rf_1000trees": rf_330k_estimators1000_unlimited_no_mates_eval,
        "linear_no_mates": linear_1m_alpha10_no_mates_eval,
        "linear_with_mates": linear_1m_alpha10_with_mates_eval,
    }

    white_player = AIPlayer(white_name, eval_map[white_eval_name], white_depth)
    black_player = AIPlayer(black_name, eval_map[black_eval_name], black_depth)

    return play_game(white_player, black_player)


def play_and_save_game_process(game_info):
    (
        game_num,
        total_games,
        white_eval_name,
        white_depth,
        black_eval_name,
        black_depth,
        completed_count,
    ) = game_info

    try:
        result = play_game_with_params(
            white_eval_name,
            white_eval_name,
            white_depth,
            black_eval_name,
            black_eval_name,
            black_depth,
        )

        # Save to CSV (each process writes independently)
        save_result_to_csv(result)

        white_player_str = f"{white_eval_name}_depth{white_depth}"
        black_player_str = f"{black_eval_name}_depth{black_depth}"
        print(
            f"Game {completed_count + game_num}/{total_games}: {white_player_str} (White) vs {black_player_str} (Black) - {result['winner']} wins in {result['move_count']} moves ({result['duration_seconds']}s)"
        )
        return result

    except Exception as e:
        white_player_str = f"{white_eval_name}_depth{white_depth}"
        black_player_str = f"{black_eval_name}_depth{black_depth}"
        print(
            f"Game {completed_count + game_num}/{total_games}: {white_player_str} vs {black_player_str} - Error: {e}"
        )
        return None


def run_tournament():
    players = create_ai_players()

    print(f"Created {len(players)} AI players:")
    for player in players:
        print(f"  - {player}")

    # Generate all combinations of players, with each pair playing twice (once each as white)
    game_combinations = []
    for player1 in players:
        for player2 in players:
            if player1 != player2:  # Don't play against self
                game_combinations.append((player1, player2))  # player1 as white
                game_combinations.append((player2, player1))  # player2 as white
    total_games = len(game_combinations)

    # Load already completed games
    completed_games = load_completed_games()
    print(f"Found {len(completed_games)} already completed games")

    # Filter out completed games
    remaining_games = [
        (w, b) for w, b in game_combinations if (str(w), str(b)) not in completed_games
    ]

    print(
        f"\nStarting tournament: {len(remaining_games)} remaining games out of {total_games} total"
    )
    print("Running 8 games in parallel with separate processes...")

    # Prepare game info for process pool - extract player parameters
    game_infos = []
    for i, (white_player, black_player) in enumerate(remaining_games):
        game_info = (
            i + 1,
            total_games,
            white_player.name,
            white_player.depth,  # white eval_name, depth
            black_player.name,
            black_player.depth,  # black eval_name, depth
            len(completed_games),
        )
        game_infos.append(game_info)

    # Run games in parallel with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(play_and_save_game_process, game_infos))

    successful_games = [r for r in results if r is not None]
    print(
        f"\nTournament complete! {len(successful_games)}/{len(remaining_games)} games completed successfully."
    )
    print("Results saved to tournament_results.csv")


if __name__ == "__main__":
    run_tournament()
