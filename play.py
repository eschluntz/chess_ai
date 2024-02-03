#!/usr/bin/env python

"""
Plays a game against the computer!
"""
import chess

from eval import piece_value_eval
from search import negamax


def human_vs_ai(board):
    while not board.is_game_over():
        print(board)
        if board.turn:
            print("Human's Turn (White):")
            move = input("Enter your move: ")
            try:
                board.push_san(move)
            except ValueError:
                print("Invalid move. Please try again.")
                continue
        else:
            print("AI's Turn (Black):")
            _, move = negamax(board, piece_value_eval, 4)
            board.push(move)
            print(f"AI played: {move}")

    print("Game Over")
    result = board.result()
    if result == "1-0":
        print("White wins!")
    elif result == "0-1":
        print("Black wins!")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    board = chess.Board()
    human_vs_ai(board)
