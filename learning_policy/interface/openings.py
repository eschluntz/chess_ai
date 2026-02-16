"""Curated common opening positions for engine evaluation and inspection."""

from __future__ import annotations

from typing import TypedDict

import chess


class OpeningDef(TypedDict):
    id: str
    name: str
    eco: str
    moves_uci: list[str]


class OpeningPosition(TypedDict):
    id: str
    name: str
    eco: str
    moves_uci: list[str]
    moves_san: list[str]
    ply: int
    fen: str


COMMON_OPENINGS: list[OpeningDef] = [
    {
        "id": "ruy_lopez",
        "name": "Ruy Lopez",
        "eco": "C60",
        "moves_uci": ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    },
    {
        "id": "italian_game",
        "name": "Italian Game (Giuoco Piano)",
        "eco": "C50",
        "moves_uci": ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
    },
    {
        "id": "scotch_game",
        "name": "Scotch Game",
        "eco": "C45",
        "moves_uci": ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"],
    },
    {
        "id": "petrov_defense",
        "name": "Petrov Defense",
        "eco": "C42",
        "moves_uci": ["e2e4", "e7e5", "g1f3", "g8f6"],
    },
    {
        "id": "sicilian_najdorf",
        "name": "Sicilian Defense: Najdorf",
        "eco": "B90",
        "moves_uci": [
            "e2e4",
            "c7c5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c5d4",
            "f3d4",
            "g8f6",
            "b1c3",
            "a7a6",
        ],
    },
    {
        "id": "sicilian_dragon_setup",
        "name": "Sicilian Defense: Dragon Setup",
        "eco": "B70",
        "moves_uci": [
            "e2e4",
            "c7c5",
            "g1f3",
            "d7d6",
            "d2d4",
            "c5d4",
            "f3d4",
            "g8f6",
            "b1c3",
            "g7g6",
        ],
    },
    {
        "id": "french_advance",
        "name": "French Defense: Advance",
        "eco": "C02",
        "moves_uci": ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5"],
    },
    {
        "id": "caro_kann_classical",
        "name": "Caro-Kann Defense: Classical",
        "eco": "B19",
        "moves_uci": ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"],
    },
    {
        "id": "scandinavian_qa5",
        "name": "Scandinavian Defense",
        "eco": "B01",
        "moves_uci": ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"],
    },
    {
        "id": "pirc_defense",
        "name": "Pirc Defense",
        "eco": "B07",
        "moves_uci": ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
    },
    {
        "id": "qgd_orthodox",
        "name": "Queen's Gambit Declined",
        "eco": "D35",
        "moves_uci": ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
    },
    {
        "id": "slav_defense",
        "name": "Slav Defense",
        "eco": "D10",
        "moves_uci": ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3"],
    },
    {
        "id": "kings_indian_defense",
        "name": "King's Indian Defense",
        "eco": "E60",
        "moves_uci": ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"],
    },
    {
        "id": "nimzo_indian_defense",
        "name": "Nimzo-Indian Defense",
        "eco": "E20",
        "moves_uci": ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    },
    {
        "id": "queens_gambit_accepted",
        "name": "Queen's Gambit Accepted",
        "eco": "D20",
        "moves_uci": ["d2d4", "d7d5", "c2c4", "d5c4"],
    },
    {
        "id": "english_opening",
        "name": "English Opening",
        "eco": "A10",
        "moves_uci": ["c2c4", "e7e5", "b1c3", "g8f6"],
    },
    {
        "id": "reti_opening",
        "name": "Réti Opening",
        "eco": "A04",
        "moves_uci": ["g1f3", "d7d5", "c2c4"],
    },
    {
        "id": "dutch_defense",
        "name": "Dutch Defense",
        "eco": "A80",
        "moves_uci": ["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6"],
    },
]


def _build_opening_position(entry: OpeningDef) -> OpeningPosition:
    board = chess.Board()
    san_moves: list[str] = []
    for uci in entry["moves_uci"]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move {uci} in opening '{entry['id']}'")
        san_moves.append(board.san(move))
        board.push(move)

    return {
        "id": entry["id"],
        "name": entry["name"],
        "eco": entry["eco"],
        "moves_uci": list(entry["moves_uci"]),
        "moves_san": san_moves,
        "ply": len(entry["moves_uci"]),
        "fen": board.fen(),
    }


def get_common_opening_positions() -> list[OpeningPosition]:
    return [_build_opening_position(entry) for entry in COMMON_OPENINGS]
