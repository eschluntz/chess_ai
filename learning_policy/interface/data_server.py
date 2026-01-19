#!/usr/bin/env python
"""
Flask server for chess training data explorer.

Visualizes precomputed training data with soft labels.

Run:
    python data_server.py --data cache/planes/1M
"""

import argparse
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# Global data (loaded once at startup)
planes = None  # (N, 13, 8, 8)
meta = None  # (N, 5)
label_indices = None  # (total_entries,)
label_probs = None  # (total_entries,)
label_offsets = None  # (N+1,)
vocab = None  # (num_moves, 3)
num_positions = 0


def sq_to_algebraic(sq: int) -> str:
    """Convert square index (0-63) to algebraic notation."""
    file_idx = sq % 8
    rank_idx = sq // 8
    return chr(ord('a') + file_idx) + str(rank_idx + 1)


def move_tuple_to_uci(from_sq: int, to_sq: int, promo: int) -> str:
    """Convert (from_sq, to_sq, promotion) to UCI string."""
    uci = sq_to_algebraic(from_sq) + sq_to_algebraic(to_sq)
    if promo > 0:
        promo_chars = ['', 'n', 'b', 'r', 'q']
        uci += promo_chars[promo]
    return uci


def planes_to_fen(p: np.ndarray, m: np.ndarray) -> str:
    """Convert planes (13, 8, 8) and meta (5,) back to FEN."""
    piece_chars = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            piece = None
            for plane_idx in range(12):
                if p[plane_idx, rank, file]:
                    piece = piece_chars[plane_idx]
                    break
            if piece:
                if empty > 0:
                    row += str(empty)
                    empty = 0
                row += piece
            else:
                empty += 1
        if empty > 0:
            row += str(empty)
        rows.append(row)

    piece_placement = '/'.join(rows)
    active_color = 'w' if m[0] == 1 else 'b'

    castling = ''
    if m[1]: castling += 'K'
    if m[2]: castling += 'Q'
    if m[3]: castling += 'k'
    if m[4]: castling += 'q'
    if not castling:
        castling = '-'

    # En passant from plane 12
    en_passant = '-'
    ep_plane = p[12]
    ep_pos = np.argwhere(ep_plane)
    if len(ep_pos) > 0:
        rank_idx, file_idx = ep_pos[0]
        en_passant = chr(ord('a') + file_idx) + str(rank_idx + 1)

    return f"{piece_placement} {active_color} {castling} {en_passant} 0 1"


def load_data(data_path: Path, split: str = "train"):
    """Load precomputed planes data."""
    global planes, meta, label_indices, label_probs, label_offsets, vocab, num_positions

    print(f"Loading data from {data_path}...")

    planes = np.load(data_path / f"{split}_planes.npy")
    meta = np.load(data_path / f"{split}_meta.npy")
    label_indices = np.load(data_path / f"{split}_label_indices.npy")
    label_probs = np.load(data_path / f"{split}_label_probs.npy")
    label_offsets = np.load(data_path / f"{split}_label_offsets.npy")

    vocab_path = data_path.parent / "vocab.npy"
    vocab = np.load(vocab_path)

    num_positions = len(planes)
    print(f"Loaded {num_positions:,} positions")
    print(f"Vocab size: {len(vocab)}")


@app.route("/")
def index():
    return send_from_directory("static", "data_explorer.html")


@app.route("/api/position/<int:idx>")
def get_position(idx: int):
    if idx < 0 or idx >= num_positions:
        return jsonify({"error": "Index out of range"}), 404

    fen = planes_to_fen(planes[idx], meta[idx])

    start = label_offsets[idx]
    end = label_offsets[idx + 1]
    move_indices = label_indices[start:end]
    probs = label_probs[start:end]

    moves = []
    for move_idx, prob in zip(move_indices, probs):
        from_sq, to_sq, promo = vocab[move_idx]
        uci = move_tuple_to_uci(from_sq, to_sq, promo)
        moves.append({
            "move": uci,
            "probability": float(prob),
        })

    moves.sort(key=lambda x: x["probability"], reverse=True)

    return jsonify({
        "index": idx,
        "total": num_positions,
        "fen": fen,
        "moves": moves,
        "num_moves": len(moves),
    })


@app.route("/api/stats")
def stats():
    return jsonify({
        "total": num_positions,
    })


def main():
    parser = argparse.ArgumentParser(description="Chess Training Data Explorer Server")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent.parent / "cache" / "planes" / "1M",
        help="Path to precomputed data directory",
    )
    parser.add_argument("--split", type=str, default="train", help="Data split (train/eval)")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        return 1

    load_data(args.data, args.split)
    print(f"\nStarting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
