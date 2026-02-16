#!/usr/bin/env python
"""Flask server for policy-vs-policy (or policy-vs-negamax) chess matches."""

from __future__ import annotations

import argparse
import sys
import threading
import uuid
from pathlib import Path

import chess
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Ensure imports work when running as `python interface/server.py` from learning_policy/
THIS_FILE = Path(__file__).resolve()
LEARNING_POLICY_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(LEARNING_POLICY_DIR))
sys.path.insert(0, str(REPO_ROOT))

from interface.match_runner import (  # noqa: E402
    ConfigError,
    EngineFactory,
    NegamaxEngine,
    default_vocab_path,
    normalize_engine_config,
    run_series_with_schedule,
)
from interface.openings import get_common_opening_positions

app = Flask(__name__, static_folder="static")

factory: EngineFactory | None = None
global_rng = np.random.default_rng(0)
server_defaults: dict[str, object] = {}
match_sessions: dict[str, dict[str, object]] = {}
match_sessions_lock = threading.Lock()
opening_catalog: list[dict[str, object]] = []
opening_fen_by_id: dict[str, str] = {}
series_jobs: dict[str, dict[str, object]] = {}
series_jobs_lock = threading.Lock()


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _apply_default_checkpoint(raw: dict[str, object], default_ckpt: str) -> dict[str, object]:
    cfg = dict(raw or {})
    if cfg.get("kind", "policy") == "policy" and not str(cfg.get("checkpoint", "")).strip():
        cfg["checkpoint"] = default_ckpt
    return cfg


def _empty_frame(board: chess.Board, engine_name: str | None, temperature: float | None) -> dict[str, object]:
    return {
        "ply": board.ply(),
        "fen": board.fen(),
        "to_move": "white" if board.turn == chess.WHITE else "black",
        "engine": engine_name,
        "temperature": temperature,
        "suggestions": [],
        "top_moves": [],
        "chosen_move": None,
        "chosen_san": None,
    }


def _frame_from_distribution(
    board: chess.Board,
    engine_name: str | None,
    temperature: float | None,
    distribution: list[tuple[str, float]],
    min_arrow_probability: float,
) -> dict[str, object]:
    return {
        "ply": board.ply(),
        "fen": board.fen(),
        "to_move": "white" if board.turn == chess.WHITE else "black",
        "engine": engine_name,
        "temperature": temperature,
        "suggestions": [
            {"move": move, "probability": float(prob)}
            for move, prob in distribution
            if prob >= min_arrow_probability
        ],
        "top_moves": [
            {"move": move, "probability": float(prob)}
            for move, prob in distribution[:10]
        ],
        "chosen_move": None,
        "chosen_san": None,
    }


def _is_done(board: chess.Board, max_plies: int) -> bool:
    return board.is_game_over(claim_draw=True) or board.ply() >= max_plies


def _finalize_session(session: dict[str, object]):
    board: chess.Board = session["board"]  # type: ignore[assignment]
    max_plies = int(session["max_plies"])

    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            result, termination = "1/2-1/2", "UNKNOWN"
        else:
            result, termination = outcome.result(), outcome.termination.name
    elif board.ply() >= max_plies:
        result, termination = "1/2-1/2", "MAX_PLIES"
    else:
        return

    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = "draw"

    session["result"] = result
    session["termination"] = termination
    session["winner"] = winner
    session["done"] = True
    session["pending_distribution"] = []

    trace: list[dict[str, object]] = session["trace"]  # type: ignore[assignment]
    if not trace or trace[-1].get("chosen_move") is not None:
        trace.append(_empty_frame(board, None, None))


def _public_session(session_id: str, session: dict[str, object]) -> dict[str, object]:
    white_meta: dict[str, object] = session["white_meta"]  # type: ignore[assignment]
    black_meta: dict[str, object] = session["black_meta"]  # type: ignore[assignment]
    board: chess.Board = session["board"]  # type: ignore[assignment]

    return {
        "session_id": session_id,
        "status": "done" if session["done"] else "ready",
        "white": white_meta,
        "black": black_meta,
        "settings": {
            "max_plies": session["max_plies"],
            "min_arrow_probability": session["min_arrow_probability"],
            "start_fen": session["start_fen"],
        },
        "match": {
            "result": session["result"],
            "winner": session["winner"],
            "termination": session["termination"],
            "plies": board.ply(),
            "final_fen": board.fen(),
            "moves_uci": session["moves_uci"],
            "moves_san": session["moves_san"],
            "trace": session["trace"],
            "done": session["done"],
        },
    }


def _set_series_job(job_id: str, **updates):
    with series_jobs_lock:
        if job_id not in series_jobs:
            return
        series_jobs[job_id].update(updates)


def _build_series_schedule(
    *,
    use_diverse_openings: bool,
    games: int,
    swap_colors: bool,
    start_fen: str,
) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    if use_diverse_openings:
        for op in opening_catalog:
            op_fen = str(op["fen"])
            op_id = str(op["id"])
            op_name = str(op["name"])
            op_eco = str(op.get("eco", ""))
            schedule.append(
                {
                    "white": "a",
                    "start_fen": op_fen,
                    "opening_id": op_id,
                    "opening_name": op_name,
                    "opening_eco": op_eco,
                }
            )
            schedule.append(
                {
                    "white": "b",
                    "start_fen": op_fen,
                    "opening_id": op_id,
                    "opening_name": op_name,
                    "opening_eco": op_eco,
                }
            )
        return schedule

    for i in range(games):
        white = "b" if (swap_colors and i % 2 == 1) else "a"
        schedule.append(
            {
                "white": white,
                "start_fen": start_fen,
                "opening_id": "custom",
                "opening_name": "Custom",
                "opening_eco": "",
            }
        )
    return schedule


def _run_series_job(
    job_id: str,
    model_a,
    model_b,
    *,
    schedule: list[dict[str, object]],
    max_plies: int,
    min_arrow_probability: float,
):
    def on_progress(done: int, total: int, game: dict[str, object], _result: dict[str, object]):
        _set_series_job(
            job_id,
            games_done=done,
            games_total=total,
            current_opening=str(game.get("opening_name", "")),
        )

    try:
        series_rng = np.random.default_rng(global_rng.integers(0, 2**63 - 1, dtype=np.int64))
        summary = run_series_with_schedule(
            model_a,
            model_b,
            schedule=schedule,
            max_plies=max_plies,
            min_arrow_probability=min_arrow_probability,
            rng=series_rng,
            progress_callback=on_progress,
        )
        _set_series_job(job_id, status="done", summary=summary)
    except Exception as exc:
        _set_series_job(job_id, status="error", error=str(exc))


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def config():
    return jsonify(
        {
            "defaults": server_defaults,
            "classical_eval_options": NegamaxEngine.eval_options(),
            "openings": opening_catalog,
        }
    )


@app.route("/api/run_match", methods=["POST"])
def run_match_api():
    if factory is None:
        return _json_error("Server is not initialized", 500)

    payload = request.get_json(silent=True) or {}

    try:
        default_ckpt = str(server_defaults.get("default_policy_checkpoint", ""))
        white_raw = _apply_default_checkpoint(payload.get("white", {}) or {}, default_ckpt)
        black_raw = _apply_default_checkpoint(payload.get("black", {}) or {}, default_ckpt)

        white_cfg = normalize_engine_config(white_raw, "white")
        black_cfg = normalize_engine_config(black_raw, "black")

        white_engine = factory.create(white_cfg)
        black_engine = factory.create(black_cfg)

        max_plies = int(payload.get("max_plies", server_defaults["max_plies"]))
        min_arrow_probability = float(
            payload.get(
                "min_arrow_probability",
                server_defaults["min_arrow_probability"],
            )
        )
        opening_id = str(payload.get("opening_id", "") or "").strip()
        if opening_id:
            start_fen = opening_fen_by_id.get(opening_id)
            if start_fen is None:
                return _json_error(f"Unknown opening_id: {opening_id}", 400)
        else:
            start_fen = str(payload.get("start_fen", server_defaults["start_fen"]))

        board = chess.Board(start_fen)
        session_id = uuid.uuid4().hex
        side_engine = white_engine if board.turn == chess.WHITE else black_engine
        initial_distribution: list[tuple[str, float]] = []
        if not _is_done(board, max_plies):
            initial_distribution = side_engine.distribution(board)

        session: dict[str, object] = {
            "id": session_id,
            "lock": threading.Lock(),
            "board": board,
            "white_engine": white_engine,
            "black_engine": black_engine,
            "white_meta": {"name": white_engine.name, "kind": white_engine.kind},
            "black_meta": {"name": black_engine.name, "kind": black_engine.kind},
            "rng": np.random.default_rng(global_rng.integers(0, 2**63 - 1, dtype=np.int64)),
            "max_plies": max_plies,
            "min_arrow_probability": min_arrow_probability,
            "start_fen": start_fen,
            "trace": [
                _frame_from_distribution(
                    board,
                    side_engine.name,
                    side_engine.temperature,
                    initial_distribution,
                    min_arrow_probability,
                )
            ],
            "moves_uci": [],
            "moves_san": [],
            "pending_distribution": initial_distribution,
            "result": None,
            "winner": None,
            "termination": None,
            "done": False,
        }

        if _is_done(board, max_plies):
            _finalize_session(session)

        with match_sessions_lock:
            match_sessions[session_id] = session

        return jsonify(_public_session(session_id, session))

    except ConfigError as exc:
        return _json_error(str(exc), 400)
    except ValueError as exc:
        return _json_error(str(exc), 400)


@app.route("/api/match_state/<session_id>")
def match_state_api(session_id: str):
    with match_sessions_lock:
        session = match_sessions.get(session_id)
    if session is None:
        return _json_error("Unknown session_id", 404)

    with session["lock"]:
        return jsonify(_public_session(session_id, session))


@app.route("/api/match_step/<session_id>", methods=["POST"])
def match_step_api(session_id: str):
    with match_sessions_lock:
        session = match_sessions.get(session_id)
    if session is None:
        return _json_error("Unknown session_id", 404)

    with session["lock"]:
        board: chess.Board = session["board"]  # type: ignore[assignment]
        if session["done"]:
            return jsonify(_public_session(session_id, session))

        max_plies = int(session["max_plies"])
        if _is_done(board, max_plies):
            _finalize_session(session)
            return jsonify(_public_session(session_id, session))

        white_engine = session["white_engine"]
        black_engine = session["black_engine"]
        engine = white_engine if board.turn == chess.WHITE else black_engine

        distribution: list[tuple[str, float]] = session.get("pending_distribution", [])  # type: ignore[assignment]
        if not distribution:
            distribution = engine.distribution(board)
            session["pending_distribution"] = distribution

        chosen_move = engine.choose_move(board, distribution, session["rng"])
        if chosen_move not in board.legal_moves:
            chosen_move = list(board.legal_moves)[0]

        trace: list[dict[str, object]] = session["trace"]  # type: ignore[assignment]
        frame = trace[-1]
        frame["chosen_move"] = chosen_move.uci()
        frame["chosen_san"] = board.san(chosen_move)

        board.push(chosen_move)
        session["moves_uci"].append(chosen_move.uci())
        session["moves_san"].append(frame["chosen_san"])

        if _is_done(board, max_plies):
            _finalize_session(session)
        else:
            next_engine = white_engine if board.turn == chess.WHITE else black_engine
            next_distribution = next_engine.distribution(board)
            session["pending_distribution"] = next_distribution
            trace.append(
                _frame_from_distribution(
                    board,
                    next_engine.name,
                    next_engine.temperature,
                    next_distribution,
                    float(session["min_arrow_probability"]),
                )
            )

        return jsonify(_public_session(session_id, session))


@app.route("/api/run_match_status/<session_id>")
def run_match_status_api(session_id: str):
    # Backward-compatible alias for older frontend polling code.
    return match_state_api(session_id)


@app.route("/api/run_series", methods=["POST"])
def run_series_api():
    if factory is None:
        return _json_error("Server is not initialized", 500)

    payload = request.get_json(silent=True) or {}

    try:
        default_ckpt = str(server_defaults.get("default_policy_checkpoint", ""))
        model_a_raw = _apply_default_checkpoint(payload.get("model_a", {}) or {}, default_ckpt)
        model_b_raw = _apply_default_checkpoint(payload.get("model_b", {}) or {}, default_ckpt)

        model_a_cfg = normalize_engine_config(model_a_raw, "model_a")
        model_b_cfg = normalize_engine_config(model_b_raw, "model_b")

        model_a = factory.create(model_a_cfg)
        model_b = factory.create(model_b_cfg)

        games = int(payload.get("games", 20))
        swap_colors = bool(payload.get("swap_colors", True))
        use_diverse_openings = bool(payload.get("use_diverse_openings", False))
        max_plies = int(payload.get("max_plies", server_defaults["max_plies"]))
        min_arrow_probability = float(
            payload.get(
                "min_arrow_probability",
                server_defaults["min_arrow_probability"],
            )
        )
        opening_id = str(payload.get("opening_id", "") or "").strip()
        if opening_id:
            start_fen = opening_fen_by_id.get(opening_id)
            if start_fen is None:
                return _json_error(f"Unknown opening_id: {opening_id}", 400)
        else:
            start_fen = str(payload.get("start_fen", server_defaults["start_fen"]))

        schedule = _build_series_schedule(
            use_diverse_openings=use_diverse_openings,
            games=games,
            swap_colors=swap_colors,
            start_fen=start_fen,
        )
        job_id = uuid.uuid4().hex

        with series_jobs_lock:
            series_jobs[job_id] = {
                "status": "running",
                "error": None,
                "games_done": 0,
                "games_total": len(schedule),
                "current_opening": "",
                "summary": None,
                "model_a": {"name": model_a.name, "kind": model_a.kind},
                "model_b": {"name": model_b.name, "kind": model_b.kind},
                "settings": {
                    "games_requested": games,
                    "games_total": len(schedule),
                    "swap_colors": swap_colors,
                    "use_diverse_openings": use_diverse_openings,
                    "max_plies": max_plies,
                    "start_fen": start_fen,
                },
            }

        thread = threading.Thread(
            target=_run_series_job,
            kwargs={
                "job_id": job_id,
                "model_a": model_a,
                "model_b": model_b,
                "schedule": schedule,
                "max_plies": max_plies,
                "min_arrow_probability": min_arrow_probability,
            },
            daemon=True,
        )
        thread.start()

        return jsonify(
            {
                "job_id": job_id,
                "status": "running",
                "games_total": len(schedule),
            }
        )

    except ConfigError as exc:
        return _json_error(str(exc), 400)
    except ValueError as exc:
        return _json_error(str(exc), 400)


@app.route("/api/run_series_status/<job_id>")
def run_series_status_api(job_id: str):
    with series_jobs_lock:
        job = series_jobs.get(job_id)
        if job is None:
            return _json_error("Unknown series job_id", 404)
        payload = dict(job)
    return jsonify(payload)


def main() -> int:
    global factory, server_defaults, opening_catalog, opening_fen_by_id

    parser = argparse.ArgumentParser(description="Policy head-to-head web server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--vocab",
        type=Path,
        default=default_vocab_path(),
        help="Path to vocab.npy (default: learning_policy/cache/planes/vocab.npy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--max-plies-default",
        type=int,
        default=300,
        help="Default max plies in each game",
    )
    parser.add_argument(
        "--min-arrow-probability-default",
        type=float,
        default=0.2,
        help="Default minimum probability for drawing arrows",
    )
    parser.add_argument(
        "--start-fen-default",
        type=str,
        default=chess.STARTING_FEN,
        help="Default starting FEN for matches",
    )
    parser.add_argument(
        "--default-policy-checkpoint",
        type=str,
        default="/Users/erik/code/chess_ai/learning_policy/checkpoints/scale-40M.pt",
        help="Default checkpoint used when a policy engine does not specify one",
    )

    args = parser.parse_args()

    vocab_path = args.vocab.expanduser().resolve()
    if not vocab_path.exists():
        print(f"Error: vocab not found at {vocab_path}")
        print("Pass --vocab /path/to/vocab.npy")
        return 1

    factory = EngineFactory(vocab_path=vocab_path, device=args.device)
    opening_catalog = get_common_opening_positions()
    opening_fen_by_id = {str(op["id"]): str(op["fen"]) for op in opening_catalog}

    server_defaults = {
        "vocab": str(vocab_path),
        "device": str(factory.device),
        "max_plies": args.max_plies_default,
        "min_arrow_probability": args.min_arrow_probability_default,
        "start_fen": args.start_fen_default,
        "default_policy_checkpoint": args.default_policy_checkpoint,
    }

    print("=" * 72)
    print("Policy Head-to-Head Server")
    print("=" * 72)
    print(f"Device: {factory.device}")
    print(f"Vocab: {vocab_path}")
    print(f"Loaded {len(opening_catalog)} common opening positions")
    print(f"URL: http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=True, use_reloader=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
