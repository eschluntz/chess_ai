"""Utilities for head-to-head chess matches between policy checkpoints and classical search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from board_repr import fen_to_planes
from cnn_model import PolicyCNN
from core.eval import piece_position_eval, piece_value_eval
from core.search import negamax

_SCORE_TEMP_CP = 200.0


class ConfigError(ValueError):
    """Raised when a user-provided engine config is invalid."""


@dataclass(frozen=True)
class EngineConfig:
    kind: str
    temperature: float
    name: str
    checkpoint: str | None = None
    eval_name: str | None = None
    depth: int | None = None


class MoveCodec:
    """Bidirectional mapping between vocab index and UCI strings."""

    def __init__(self, vocab_path: Path):
        self.vocab_path = vocab_path
        moves = np.load(vocab_path)
        if moves.ndim != 2 or moves.shape[1] != 3:
            raise ConfigError(f"Expected vocab shape (N, 3), got {moves.shape}")

        self.idx_to_tuple = [tuple(int(v) for v in row.tolist()) for row in moves]
        self.idx_to_uci = [self._tuple_to_uci(m) for m in moves]
        self.uci_to_idx = {uci: idx for idx, uci in enumerate(self.idx_to_uci)}

    @staticmethod
    def _tuple_to_uci(move: np.ndarray | tuple[int, int, int]) -> str:
        from_sq, to_sq, promo = [int(v) for v in move]
        from_file, from_rank = from_sq % 8, from_sq // 8
        to_file, to_rank = to_sq % 8, to_sq // 8

        uci = chr(ord("a") + from_file) + str(from_rank + 1)
        uci += chr(ord("a") + to_file) + str(to_rank + 1)
        if promo > 0:
            uci += ["", "n", "b", "r", "q"][promo]
        return uci

    @staticmethod
    def _uci_to_tuple(uci: str) -> tuple[int, int, int]:
        from_file = ord(uci[0]) - ord("a")
        from_rank = int(uci[1]) - 1
        to_file = ord(uci[2]) - ord("a")
        to_rank = int(uci[3]) - 1
        promo = {"": 0, "n": 1, "b": 2, "r": 3, "q": 4}.get(uci[4:] if len(uci) > 4 else "", 0)

        from_sq = from_rank * 8 + from_file
        to_sq = to_rank * 8 + to_file
        return from_sq, to_sq, promo

    @staticmethod
    def _flip_sq(sq: int) -> int:
        return (7 - sq // 8) * 8 + (sq % 8)

    def flip_uci_vertical(self, uci: str) -> str:
        from_sq, to_sq, promo = self._uci_to_tuple(uci)
        return self._tuple_to_uci(
            (self._flip_sq(from_sq), self._flip_sq(to_sq), promo)
        )


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    return probs / np.sum(probs)


def _distribution_with_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return out

    if abs(temperature - 1.0) < 1e-8:
        return probs

    logp = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    return _softmax(logp)


class BaseEngine:
    kind: str
    name: str
    temperature: float

    def distribution(self, board: chess.Board) -> list[tuple[str, float]]:
        raise NotImplementedError

    def choose_move(
        self,
        board: chess.Board,
        distribution: list[tuple[str, float]],
        rng: np.random.Generator,
    ) -> chess.Move:
        if not distribution:
            legal = list(board.legal_moves)
            return legal[0]

        moves = [uci for uci, _ in distribution]
        probs = np.array([p for _, p in distribution], dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0.0:
            probs = np.full(len(moves), 1.0 / len(moves), dtype=np.float64)
        else:
            probs = probs / total

        if self.temperature <= 0:
            idx = int(np.argmax(probs))
        else:
            sampled = _distribution_with_temperature(probs, self.temperature)
            sampled = np.clip(sampled, 0.0, None)
            sampled_total = float(np.sum(sampled))
            if not np.isfinite(sampled_total) or sampled_total <= 0.0:
                sampled = np.full(len(moves), 1.0 / len(moves), dtype=np.float64)
            else:
                sampled = sampled / sampled_total

            # Make numpy.random.choice happy with an exact total of 1.
            if len(sampled) > 1:
                sampled[-1] = 1.0 - float(np.sum(sampled[:-1]))
            idx = int(rng.choice(len(moves), p=sampled))

        return chess.Move.from_uci(moves[idx])


class PolicyEngine(BaseEngine):
    """Policy checkpoint engine (CNN) that samples legal moves from logits."""

    def __init__(
        self,
        config: EngineConfig,
        codec: MoveCodec,
        device: torch.device,
    ):
        self.kind = "policy"
        self.name = config.name
        self.temperature = config.temperature
        self.device = device
        self.codec = codec

        if not config.checkpoint:
            raise ConfigError("Policy engine requires a checkpoint path")

        ckpt_path = Path(config.checkpoint).expanduser().resolve()
        if not ckpt_path.exists():
            raise ConfigError(f"Checkpoint not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ConfigError("Unsupported checkpoint format")

        state_dict = _strip_compile_prefix(state_dict)

        if "layers.0.weight" not in state_dict or "output.weight" not in state_dict:
            raise ConfigError(
                "Checkpoint does not look like a PolicyCNN model (missing layers.0.weight/output.weight)"
            )

        first_conv = state_dict["layers.0.weight"]
        out_linear = state_dict["output.weight"]

        hidden_channels = int(first_conv.shape[0])
        input_channels = int(first_conv.shape[1])
        kernel_size = int(first_conv.shape[2])
        num_moves = int(out_linear.shape[0])

        block_pat = re.compile(r"^layers\.(\d+)\.conv1\.weight$")
        block_indices = {
            int(m.group(1))
            for key in state_dict
            if (m := block_pat.match(key)) is not None
        }
        num_layers = len(block_indices)

        flip_board = input_channels == 17

        if num_layers == 0:
            raise ConfigError("Failed to infer residual depth from checkpoint state dict")

        if num_moves != len(codec.idx_to_uci):
            raise ConfigError(
                f"Vocab size mismatch: checkpoint has {num_moves} moves, vocab has {len(codec.idx_to_uci)}"
            )

        model = PolicyCNN(
            num_moves=num_moves,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            flip_board=flip_board,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.model = model
        self.flip_board = bool(model.flip_board)

    def distribution(self, board: chess.Board) -> list[tuple[str, float]]:
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            return []

        known_moves: list[str] = []
        known_indices: list[int] = []
        for move in legal_moves:
            lookup_move = move
            if self.flip_board and board.turn == chess.BLACK:
                # flip_board=True models are trained in current-player coordinates,
                # so black moves must be looked up in vertically flipped coordinates.
                lookup_move = self.codec.flip_uci_vertical(move)

            idx = self.codec.uci_to_idx.get(lookup_move)
            if idx is None:
                continue
            known_moves.append(move)
            known_indices.append(idx)

        if not known_indices:
            uniform = 1.0 / len(legal_moves)
            return [(m, uniform) for m in legal_moves]

        planes, meta = fen_to_planes(board.fen())
        planes_t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        meta_t = torch.from_numpy(meta).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(planes_t, meta_t).squeeze(0).float().cpu().numpy()

        known_logits = logits[np.array(known_indices, dtype=np.int64)]
        probs = _softmax(known_logits)

        dist = list(zip(known_moves, probs.tolist()))
        dist.sort(key=lambda x: x[1], reverse=True)
        return dist


class NegamaxEngine(BaseEngine):
    """Classical search engine wrapped into a probabilistic move policy."""

    _evals: dict[str, Any] = {
        "piece_value": piece_value_eval,
        "piece_position": piece_position_eval,
    }

    def __init__(self, config: EngineConfig):
        self.kind = "negamax"
        self.name = config.name
        self.temperature = config.temperature

        if not config.eval_name:
            raise ConfigError("Negamax engine requires an eval_name")
        if config.eval_name not in self._evals:
            valid = ", ".join(sorted(self._evals))
            raise ConfigError(f"Unknown eval_name '{config.eval_name}'. Valid values: {valid}")

        self.eval_name = config.eval_name
        self.eval_fn = self._evals[config.eval_name]
        self.depth = int(config.depth or 1)
        if self.depth < 1:
            raise ConfigError("Negamax depth must be >= 1")

    @classmethod
    def eval_options(cls) -> list[str]:
        return sorted(cls._evals)

    def distribution(self, board: chess.Board) -> list[tuple[str, float]]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        side = 1 if board.turn == chess.WHITE else -1
        scores: list[float] = []
        ucis: list[str] = []

        for move in legal_moves:
            board.push(move)
            score, _ = negamax(board, self.eval_fn, max(self.depth - 1, 0))
            board.pop()
            scores.append(float(score * side))
            ucis.append(move.uci())

        arr = np.array(scores, dtype=np.float64)
        logits = (arr - np.max(arr)) / _SCORE_TEMP_CP
        probs = _softmax(logits)

        dist = list(zip(ucis, probs.tolist()))
        dist.sort(key=lambda x: x[1], reverse=True)
        return dist


class EngineFactory:
    """Caches loaded policy models so repeated games do not reload checkpoints."""

    def __init__(self, vocab_path: Path, device: str = "auto"):
        self.vocab_path = vocab_path
        self.codec = MoveCodec(vocab_path)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._policy_cache: dict[tuple[str, str, float, str], PolicyEngine] = {}

    def create(self, config: EngineConfig) -> BaseEngine:
        if config.kind == "policy":
            if not config.checkpoint:
                raise ConfigError("Policy config is missing checkpoint")
            ckpt_path = str(Path(config.checkpoint).expanduser().resolve())
            key = (ckpt_path, config.name, config.temperature, str(self.device))
            if key not in self._policy_cache:
                cached_config = EngineConfig(
                    kind="policy",
                    checkpoint=ckpt_path,
                    temperature=config.temperature,
                    name=config.name,
                )
                self._policy_cache[key] = PolicyEngine(cached_config, self.codec, self.device)
            return self._policy_cache[key]

        if config.kind == "negamax":
            return NegamaxEngine(config)

        raise ConfigError(f"Unsupported engine kind: {config.kind}")


def normalize_engine_config(raw: dict[str, Any], side: str) -> EngineConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"Missing engine config for {side}")

    kind = str(raw.get("kind", "policy")).strip().lower()
    temperature = float(raw.get("temperature", 0.0))
    if temperature < 0:
        raise ConfigError(f"Temperature must be >= 0 for {side}")

    if kind == "classical":
        kind = "negamax"

    if kind == "policy":
        checkpoint = raw.get("checkpoint")
        if checkpoint is None or not str(checkpoint).strip():
            raise ConfigError(f"Missing checkpoint path for {side}")
        checkpoint = str(checkpoint).strip()
        default_name = Path(checkpoint).stem
        name = str(raw.get("name", default_name)).strip() or default_name
        return EngineConfig(
            kind="policy",
            checkpoint=checkpoint,
            temperature=temperature,
            name=name,
        )

    if kind == "negamax":
        eval_name = str(raw.get("eval_name", "piece_value")).strip()
        depth = int(raw.get("depth", 2))
        default_name = f"negamax-{eval_name}-d{depth}"
        name = str(raw.get("name", default_name)).strip() or default_name
        return EngineConfig(
            kind="negamax",
            eval_name=eval_name,
            depth=depth,
            temperature=temperature,
            name=name,
        )

    raise ConfigError(f"Unsupported engine kind '{kind}' for {side}")


def _suggestions(
    distribution: list[tuple[str, float]],
    min_arrow_probability: float,
) -> list[dict[str, float | str]]:
    return [
        {"move": move, "probability": float(prob)}
        for move, prob in distribution
        if prob >= min_arrow_probability
    ]


def play_match(
    white: BaseEngine,
    black: BaseEngine,
    *,
    max_plies: int,
    min_arrow_probability: float,
    start_fen: str,
    rng: np.random.Generator,
    collect_trace: bool = True,
    progress_callback: Any = None,
) -> dict[str, Any]:
    if max_plies < 1:
        raise ConfigError("max_plies must be >= 1")
    if not 0.0 <= min_arrow_probability <= 1.0:
        raise ConfigError("min_arrow_probability must be in [0, 1]")

    if start_fen.strip().lower() == "startpos":
        start_fen = chess.STARTING_FEN

    board = chess.Board(start_fen)
    trace: list[dict[str, Any]] = []
    uci_moves: list[str] = []
    san_moves: list[str] = []

    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        engine = white if board.turn == chess.WHITE else black
        distribution = engine.distribution(board)
        chosen_move = engine.choose_move(board, distribution, rng)
        if chosen_move not in board.legal_moves:
            chosen_move = list(board.legal_moves)[0]

        if collect_trace:
            trace.append(
                {
                    "ply": board.ply(),
                    "fen": board.fen(),
                    "to_move": "white" if board.turn == chess.WHITE else "black",
                    "engine": engine.name,
                    "temperature": engine.temperature,
                    "suggestions": _suggestions(distribution, min_arrow_probability),
                    "top_moves": [
                        {"move": move, "probability": float(prob)}
                        for move, prob in distribution[:10]
                    ],
                    "chosen_move": chosen_move.uci(),
                    "chosen_san": board.san(chosen_move),
                }
            )

        board.push(chosen_move)
        uci_moves.append(chosen_move.uci())
        san_moves.append(trace[-1]["chosen_san"] if collect_trace else "")
        if progress_callback is not None:
            progress_callback(board.ply())

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = outcome.result()
        termination = outcome.termination.name
    else:
        result = "1/2-1/2"
        termination = "MAX_PLIES"

    if result == "1-0":
        winner = "white"
    elif result == "0-1":
        winner = "black"
    else:
        winner = "draw"

    if collect_trace:
        trace.append(
            {
                "ply": board.ply(),
                "fen": board.fen(),
                "to_move": "white" if board.turn == chess.WHITE else "black",
                "engine": None,
                "temperature": None,
                "suggestions": [],
                "top_moves": [],
                "chosen_move": None,
                "chosen_san": None,
            }
        )

    return {
        "result": result,
        "winner": winner,
        "termination": termination,
        "plies": board.ply(),
        "final_fen": board.fen(),
        "moves_uci": uci_moves,
        "moves_san": [m for m in san_moves if m],
        "trace": trace,
    }


def run_series(
    model_a: BaseEngine,
    model_b: BaseEngine,
    *,
    games: int,
    swap_colors: bool,
    max_plies: int,
    min_arrow_probability: float,
    start_fen: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if games < 1:
        raise ConfigError("games must be >= 1")

    model_a_label = model_a.name
    model_b_label = model_b.name
    if model_a_label == model_b_label:
        model_a_label = f"{model_a_label} [A]"
        model_b_label = f"{model_b_label} [B]"

    stats = {
        model_a_label: {"wins": 0, "losses": 0, "draws": 0},
        model_b_label: {"wins": 0, "losses": 0, "draws": 0},
    }

    color_stats = {"white_wins": 0, "black_wins": 0, "draws": 0}

    for idx in range(games):
        if swap_colors and idx % 2 == 1:
            white, black = model_b, model_a
            white_label, black_label = model_b_label, model_a_label
        else:
            white, black = model_a, model_b
            white_label, black_label = model_a_label, model_b_label

        game_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
        result = play_match(
            white,
            black,
            max_plies=max_plies,
            min_arrow_probability=min_arrow_probability,
            start_fen=start_fen,
            rng=game_rng,
            collect_trace=False,
        )

        if result["winner"] == "draw":
            stats[white_label]["draws"] += 1
            stats[black_label]["draws"] += 1
            color_stats["draws"] += 1
            continue

        if result["winner"] == "white":
            winner_name, loser_name = white_label, black_label
            color_stats["white_wins"] += 1
        else:
            winner_name, loser_name = black_label, white_label
            color_stats["black_wins"] += 1

        stats[winner_name]["wins"] += 1
        stats[loser_name]["losses"] += 1

    summary = {
        "games": games,
        "swap_colors": swap_colors,
        "start_fen": start_fen,
        "by_model": {},
        "by_color": color_stats,
    }

    for name, counts in stats.items():
        summary["by_model"][name] = {
            **counts,
            "win_rate": counts["wins"] / games,
            "loss_rate": counts["losses"] / games,
            "draw_rate": counts["draws"] / games,
        }

    return summary


def run_series_with_schedule(
    model_a: BaseEngine,
    model_b: BaseEngine,
    *,
    schedule: list[dict[str, Any]],
    max_plies: int,
    min_arrow_probability: float,
    rng: np.random.Generator,
    progress_callback: Any = None,
) -> dict[str, Any]:
    if not schedule:
        raise ConfigError("schedule must contain at least one game")

    model_a_label = model_a.name
    model_b_label = model_b.name
    if model_a_label == model_b_label:
        model_a_label = f"{model_a_label} [A]"
        model_b_label = f"{model_b_label} [B]"

    stats = {
        model_a_label: {"wins": 0, "losses": 0, "draws": 0},
        model_b_label: {"wins": 0, "losses": 0, "draws": 0},
    }
    color_stats = {"white_wins": 0, "black_wins": 0, "draws": 0}
    opening_stats: dict[str, dict[str, Any]] = {}

    total_games = len(schedule)

    for idx, game in enumerate(schedule):
        white_side = str(game.get("white", "a")).lower()
        white = model_a if white_side == "a" else model_b
        black = model_b if white_side == "a" else model_a
        white_label = model_a_label if white_side == "a" else model_b_label
        black_label = model_b_label if white_side == "a" else model_a_label

        start_fen = str(game.get("start_fen", chess.STARTING_FEN))
        opening_id = str(game.get("opening_id", "custom"))
        opening_name = str(game.get("opening_name", "Custom"))
        opening_eco = str(game.get("opening_eco", ""))

        game_rng = np.random.default_rng(rng.integers(0, 2**63 - 1, dtype=np.int64))
        result = play_match(
            white,
            black,
            max_plies=max_plies,
            min_arrow_probability=min_arrow_probability,
            start_fen=start_fen,
            rng=game_rng,
            collect_trace=False,
        )

        op = opening_stats.setdefault(
            opening_id,
            {
                "id": opening_id,
                "name": opening_name,
                "eco": opening_eco,
                "games": 0,
                "white_wins": 0,
                "black_wins": 0,
                "draws": 0,
            },
        )
        op["games"] += 1

        if result["winner"] == "draw":
            stats[white_label]["draws"] += 1
            stats[black_label]["draws"] += 1
            color_stats["draws"] += 1
            op["draws"] += 1
        else:
            if result["winner"] == "white":
                winner_name, loser_name = white_label, black_label
                color_stats["white_wins"] += 1
                op["white_wins"] += 1
            else:
                winner_name, loser_name = black_label, white_label
                color_stats["black_wins"] += 1
                op["black_wins"] += 1
            stats[winner_name]["wins"] += 1
            stats[loser_name]["losses"] += 1

        if progress_callback is not None:
            progress_callback(idx + 1, total_games, game, result)

    summary = {
        "games": total_games,
        "by_model": {},
        "by_color": color_stats,
        "by_opening": sorted(opening_stats.values(), key=lambda x: x["id"]),
    }

    for name, counts in stats.items():
        summary["by_model"][name] = {
            **counts,
            "win_rate": counts["wins"] / total_games,
            "loss_rate": counts["losses"] / total_games,
            "draw_rate": counts["draws"] / total_games,
        }

    return summary


def default_vocab_path() -> Path:
    return Path(__file__).resolve().parents[1] / "cache" / "planes" / "vocab.npy"
