#!/usr/bin/env python
"""
Flask server for chess policy visualizer.

Run from learning_policy directory:
    python interface/server.py --checkpoint checkpoints/<run_id>/latest.pt
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

from data import build_move_vocabulary, get_index_to_move
from features import extract_features_piece_square
from mlp_model import SimplePolicyMLP

app = Flask(__name__, static_folder="static")

# Global model and vocab (loaded once at startup)
model = None
device = None
vocab = None
idx_to_move = None


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    global model, device, vocab, idx_to_move

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build vocabulary
    vocab = build_move_vocabulary()
    idx_to_move = get_index_to_move()
    num_moves = len(vocab)

    # Determine input size from features
    dummy_board = chess.Board()
    input_size = len(extract_features_piece_square(dummy_board))

    # Create and load model
    model = SimplePolicyMLP(
        input_size=input_size,
        num_moves=num_moves,
        hidden_size=256,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Vocabulary size: {num_moves}")


def get_move_probabilities(fen: str, top_n: int = 5) -> list[dict]:
    """Get top N moves with probabilities for a position (including illegal moves)."""
    board = chess.Board(fen)
    legal_moves = set(move.uci() for move in board.legal_moves)

    # Extract features and run model
    features = extract_features_piece_square(board)
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Get all moves sorted by probability
    all_moves = []
    for idx, prob in enumerate(probs):
        move = idx_to_move[idx]
        all_moves.append({
            "move": move,
            "probability": float(prob),
            "legal": move in legal_moves,
        })

    all_moves.sort(key=lambda x: x["probability"], reverse=True)
    return all_moves[:top_n]


def sample_move(fen: str, temperature: float) -> str:
    """Sample a move from the policy distribution."""
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    legal_uci = [move.uci() for move in legal_moves]

    # Extract features and run model
    features = extract_features_piece_square(board)
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(x).squeeze(0)

        # Get probabilities for legal moves only
        legal_indices = [vocab[m] for m in legal_uci]
        legal_logits = logits[legal_indices].cpu().numpy()

        if temperature == 0:
            # Greedy: pick highest probability
            best_idx = np.argmax(legal_logits)
        else:
            # Sample from softmax distribution
            scaled_logits = legal_logits / temperature
            probs = np.exp(scaled_logits - np.max(scaled_logits))
            probs = probs / probs.sum()
            best_idx = np.random.choice(len(legal_uci), p=probs)

    return legal_uci[best_idx]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/predict")
def predict():
    fen = request.args.get("fen", chess.STARTING_FEN)
    top_n = int(request.args.get("top_n", 5))

    moves = get_move_probabilities(fen, top_n)
    return jsonify({"moves": moves})


@app.route("/api/sample", methods=["POST"])
def sample():
    data = request.json
    fen = data.get("fen", chess.STARTING_FEN)
    temperature = float(data.get("temperature", 1.0))

    move = sample_move(fen, temperature)
    return jsonify({"move": move})


def main():
    parser = argparse.ArgumentParser(description="Chess Policy Visualizer Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    load_model(str(checkpoint_path))
    print(f"\nStarting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
