import importlib
import json
import os
import random
import time
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from board import Board
from genetic_helpers import bool_to_np
from piece import BODIES2, Piece


MODEL_FILE = Path(__file__).resolve().parent / "data" / "custom_model_best.json"
POP_SIZE = 100
T_RATIO = 0.1
O_RATIO = 0.3
M_CHANCE = 0.05
M_DELTA = 0.2
GAMES = 25
LOOK_DEPTH = 0
MAX_MOVES = 500
TEST_GAMES = 1

FEATS = 4
DEFAULT_W: Tuple[float, float, float, float] = (
    -0.5110110669045389,
    0.14150862207010803,
    -0.810748088689264,
    -0.2480534943668594,
)
LOOK_TEMPLATES: Tuple[Tuple[Tuple[Tuple[int, int], ...], Tuple[int, int, int]], ...] = tuple(
    BODIES2
)
_PATCHED = False


def _grid_board(grid: List[List[bool]]) -> Board:
    bd = Board()
    bd.board = [list(row) for row in grid]
    bd.colors = [[False] * bd.width for _ in range(len(bd.board))]
    bd.widths = [sum(row) for row in bd.board]
    bd.heights = [
        max((r + 1 for r, row in enumerate(bd.board) if row[col]), default=0)
        for col in range(bd.width)
    ]
    return bd


def _reset_game(gm: "Game") -> None:
    gm.board = Board()
    gm.curr_piece = Piece()
    gm.y = 20
    gm.x = 5
    gm.top = 0
    gm.pieces_dropped = 0
    gm.rows_cleared = 0


def _patch_game() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        mod = importlib.import_module("game")
    except Exception:
        return
    Game = getattr(mod, "Game", None)
    if Game is None:
        return

    base_run = Game.run_no_visual

    def run_patch(self, *args, **kwargs):
        ai = getattr(self, "ai", None)
        if ai is None:
            return base_run(self, *args, **kwargs)
        games = getattr(ai, "test_games", 1)
        limit = getattr(ai, "max_moves", MAX_MOVES)
        inf = limit is None
        if inf:
            limit = float("inf")
        res: List[Tuple[int, int]] = []
        for idx in range(max(1, games)):
            _reset_game(self)
            mv = 0
            while True:
                try:
                    x, pc = ai.get_best_move(self.board, self.curr_piece)
                except Exception:
                    return -1, 0
                self.curr_piece = pc
                try:
                    y = self.board.drop_height(self.curr_piece, x)
                except Exception:
                    break
                self.drop(y, x=x)
                mv += 1
                if self.board.top_filled() or mv >= limit:
                    break
            res.append((self.pieces_dropped, self.rows_cleared))
            suffix = " (no limit)" if inf else ""
            print(f"Game {idx + 1}: pieces={self.pieces_dropped} rows={self.rows_cleared}{suffix}")
        final_res = res[-1] if res else (0, 0)
        self.pieces_dropped, self.rows_cleared = final_res
        if games == 1:
            print(self.pieces_dropped, self.rows_cleared)
        else:
            for i, (pieces, rows) in enumerate(res, 1):
                print(f"Game {i}: pieces={pieces} rows={rows}")
        return final_res

    Game.run_no_visual = run_patch

    if not hasattr(Game, "_drop_patched_custom"):
        base_drop = Game.drop

        def drop_patch(self, y, x=None):
            prev = getattr(self, "rows_cleared", 0)
            out = base_drop(self, y, x=x)
            ai = getattr(self, "ai", None)
            if ai is not None and getattr(ai, "log_piece_clears", False):
                cleared = getattr(self, "rows_cleared", 0) - prev
                print(f"Piece {self.pieces_dropped}: cleared {cleared} lines (total {self.rows_cleared})")
            return out

        Game.drop = drop_patch
        Game._drop_patched_custom = True

    _PATCHED = True


def _col_h(grid: np.ndarray) -> np.ndarray:
    h = np.zeros(grid.shape[1], dtype=int)
    for c in range(grid.shape[1]):
        filled = np.flatnonzero(grid[:, c])
        h[c] = int(filled[-1] + 1) if filled.size else 0
    return h


def _holes(grid: np.ndarray) -> int:
    cnt = 0
    for col in grid.T:
        seen = False
        for cell in col[::-1]:
            if cell:
                seen = True
            elif seen:
                cnt += 1
    return cnt


def _agg_h(h: np.ndarray) -> float:
    return float(np.sum(h))


def _full_rows(grid: np.ndarray) -> float:
    if grid.size == 0:
        return 0.0
    row_sum = np.sum(grid, axis=1)
    w = grid.shape[1]
    return float(np.count_nonzero(row_sum == w))


def _bump(h: np.ndarray) -> float:
    if h.size <= 1:
        return 0.0
    return float(np.sum(np.abs(np.diff(h))))


def _norm(vals: Sequence[float]) -> List[float]:
    vec = np.array(vals, dtype=float)
    nrm = float(np.linalg.norm(vec))
    if np.isclose(nrm, 0.0):
        return list(DEFAULT_W)
    return list(vec / nrm)


def _rand_w() -> List[float]:
    vec = np.random.normal(size=FEATS)
    return list(DEFAULT_W) if not np.any(vec) else _norm(vec)


def _mut_w(wts: Sequence[float], delta: float = M_DELTA) -> List[float]:
    mut = list(wts)
    idx = random.randrange(FEATS)
    mut[idx] += random.uniform(-delta, delta)
    return _norm(mut)


def _mix_w(
    a: Sequence[float],
    b: Sequence[float],
    fitness_a: float,
    fitness_b: float,
) -> List[float]:
    fa = max(float(fitness_a), 0.0)
    fb = max(float(fitness_b), 0.0)
    tot = fa + fb
    if tot == 0.0:
        mix = [(wa + wb) / 2.0 for wa, wb in zip(a, b)]
    else:
        mix = [(wa * fa + wb * fb) / tot for wa, wb in zip(a, b)]
    return _norm(mix)


@contextmanager
def _mute():
    with open(os.devnull, "w") as null, redirect_stdout(null):
        yield


def load_model(path: Path = MODEL_FILE) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if "weights" not in data or len(data["weights"]) != FEATS:
        return None
    return data


def save_model(
    weights: Sequence[float],
    lines_cleared: float,
    pieces_dropped: float,
    generation: int,
    path: Path = MODEL_FILE,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "weights": [float(w) for w in weights],
        "lines_cleared": float(lines_cleared),
        "pieces_dropped": float(pieces_dropped),
        "generation": int(generation),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


@dataclass
class PlacementResult:
    board: List[List[bool]]
    rows_cleared: int
    landing_height: float
    eroded_cells: int
    line_score: int
    board_state: Optional[Board] = None
    pre_clear_board: Optional[List[List[bool]]] = None


class HeuristicAgent:
    def __init__(self, weights: Sequence[float], lookahead_depth: int = 0):
        self.w = np.array(weights, dtype=float)
        self.depth = max(0, int(lookahead_depth))
        self.max_moves = MAX_MOVES
        self.templates = LOOK_TEMPLATES

    def get_best_move(self, board: Board, piece: Piece) -> Tuple[int, Piece]:
        _, best_x, best_piece = self._search(board, piece, self.depth)
        return best_x, best_piece

    def _search(
        self, board: Board, piece: Piece, depth: int
    ) -> Tuple[float, int, Piece]:
        best_val = float("-inf")
        best_x = 0
        best_piece = piece

        for rotated in self._rotations(piece):
            max_x = board.width - HeuristicAgent._pw(rotated)
            if max_x < 0:
                continue
            for x in range(max_x + 1):
                try:
                    y = board.drop_height(rotated, x)
                except Exception:
                    continue
                if y < 0:
                    continue
                pl = self._place(board, rotated, x, y, include_board_state=depth > 0)
                if pl is None:
                    continue
                val = self._eval(pl, depth)
                if val > best_val:
                    best_val = val
                    best_x = x
                    best_piece = rotated

        return best_val, best_x, best_piece

    def _eval(self, pl: PlacementResult, depth: int) -> float:
        val = self._score(pl)
        if depth <= 0:
            return val
        nxt = pl.board_state
        if nxt is None:
            nxt = _grid_board(pl.board)
            pl.board_state = nxt
        fut = []
        for body, color in self.templates:
            nxt_piece = Piece(body, color)
            nxt_val, _, _ = self._search(nxt, nxt_piece, depth - 1)
            if nxt_val != float("-inf"):
                fut.append(nxt_val)
        if not fut:
            return val
        return val + float(np.mean(fut))

    def _score(self, pl: PlacementResult) -> float:
        grid = pl.pre_clear_board or pl.board
        np_grid = bool_to_np(grid).astype(int)
        h = _col_h(np_grid)
        agg = _agg_h(h)
        lines = _full_rows(np_grid)
        holes = float(_holes(np_grid))
        bump = _bump(h)
        feats = np.array(
            [
                agg,
                lines,
                holes,
                bump,
            ],
            dtype=float,
        )
        return float(np.dot(self.w, feats))

    @staticmethod
    def _place(
        board: Board, piece: Piece, x: int, y: int, include_board_state: bool = False
    ) -> Optional[PlacementResult]:
        grid = [row[:] for row in board.board]
        w = board.width
        h = len(grid)
        cells: List[Tuple[int, int]] = []
        for px, py in piece.body:
            row = y + py
            col = x + px
            if row < 0 or row >= h or col < 0 or col >= w:
                return None
            if grid[row][col]:
                return None
            grid[row][col] = True
            cells.append((row, col))

        before = [row[:] for row in grid]
        after, rows = HeuristicAgent._clear_rows(grid)
        row_set = set(rows)
        eroded = sum(1 for row, _ in cells if row in row_set)
        land = y + HeuristicAgent._ph(piece) / 2.0
        score = HeuristicAgent._line_score(len(rows))
        state = _grid_board(after) if include_board_state else None

        return PlacementResult(
            board=after,
            rows_cleared=len(rows),
            landing_height=land,
            eroded_cells=eroded,
            line_score=score,
            board_state=state,
            pre_clear_board=before,
        )

    @staticmethod
    def _rotations(piece: Piece) -> Iterable[Piece]:
        seen = set()
        cur = piece
        for _ in range(4):
            sig = tuple(sorted(cur.body))
            if sig in seen:
                break
            seen.add(sig)
            yield cur
            cur = cur.get_next_rotation()

    @staticmethod
    def _clear_rows(
        grid: List[List[bool]],
    ) -> Tuple[List[List[bool]], List[int]]:
        if not grid:
            return grid, []
        w = len(grid[0])
        full_rows = [idx for idx, row in enumerate(grid) if all(row)]
        if not full_rows:
            return grid, []
        remain = [row for idx, row in enumerate(grid) if idx not in full_rows]
        new_rows = [[False] * w for _ in full_rows]
        return remain + new_rows, full_rows

    @staticmethod
    def _ph(pc: Piece) -> int:
        return (max(py for _, py in pc.body) + 1) if pc.body else 0

    @staticmethod
    def _pw(pc: Piece) -> int:
        return (max(px for px, _ in pc.body) + 1) if pc.body else 0

    @staticmethod
    def _line_score(rows: int) -> int:
        score_table = {0: 0, 1: 2, 2: 5, 3: 15, 4: 60}
        return score_table.get(rows, 0)


class CUSTOM_AI_MODEL:
    def __init__(self):
        self.w = _norm(DEFAULT_W)
        _patch_game()
        self.max_moves = None
        self.test_games = TEST_GAMES
        self.log_piece_clears = True
        self.agent = HeuristicAgent(self.w, lookahead_depth=1)

    def get_best_move(self, board: Board, piece: Piece) -> Tuple[int, Piece]:
        return self.agent.get_best_move(board, piece)


@dataclass
class Individual:
    w: List[float]
    lines: float = 0.0
    pieces: float = 0.0
    fit: float = 0.0

    def clone(self) -> "Individual":
        return Individual(w=list(self.w))


class GeneticTrainer:
    def __init__(
        self,
        population_size: int = POP_SIZE,
        games_per_agent: int = GAMES,
        tournament_ratio: float = T_RATIO,
        offspring_ratio: float = O_RATIO,
        mutation_chance: float = M_CHANCE,
        mutation_delta: float = M_DELTA,
        lookahead_depth: int = LOOK_DEPTH,
        model_path: Path = MODEL_FILE,
    ):
        self.size = population_size
        self.games = games_per_agent
        self.depth = max(0, int(lookahead_depth))
        self.t_ratio = max(0.01, min(0.5, tournament_ratio))
        self.o_ratio = max(0.05, min(0.9, offspring_ratio))
        self.mut_chance = max(0.0, min(1.0, mutation_chance))
        self.delta = mutation_delta
        self.path = Path(model_path)
        self.best = load_model(self.path)
        self.pop = self._init_population()
        _patch_game()

    def _init_population(self) -> List[Individual]:
        pop: List[Individual] = []
        base: Optional[Sequence[float]] = None
        if self.best:
            base = self.best["weights"]
            pop.append(Individual(w=list(base)))
        else:
            base = list(DEFAULT_W)
            pop.append(Individual(w=list(base)))

        while len(pop) < self.size:
            if base and len(pop) < self.size // 2:
                pop.append(Individual(w=_mut_w(base, self.delta)))
            else:
                pop.append(Individual(w=_rand_w()))
        return pop

    def train(self, generations: Optional[int] = None) -> None:
        generation = 1
        try:
            while generations is None or generation <= generations:
                evaluated = [
                    self._evaluate(idx + 1, ind)
                    for idx, ind in enumerate(self.pop)
                ]
                evaluated.sort(key=lambda ind: ind.fit, reverse=True)
                best = evaluated[0]
                print(
                    f"Generation {generation}: "
                    f"fitness={best.fit:.4f} "
                    f"lines={best.lines:.2f} pieces={best.pieces:.2f}"
                )
                self._maybe_save(best, generation)
                self.pop = self._breed(evaluated)
                generation += 1
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Latest best preserved.")

    def _evaluate(self, agent_index: int, individual: Individual) -> Individual:
        from game import Game

        total_lines = 0.0
        total_pieces = 0.0
        for _ in range(self.games):
            agent = HeuristicAgent(individual.w, lookahead_depth=self.depth)
            game = Game("genetic", agent=agent)
            with _mute():
                pieces_dropped, rows_cleared = game.run_no_visual()
            total_lines += rows_cleared
            total_pieces += pieces_dropped
        individual.lines = total_lines / self.games
        individual.pieces = total_pieces / self.games
        individual.fit = individual.lines
        print(
            f"  Agent {agent_index:03d}: "
            f"lines={individual.lines:.2f} "
            f"pieces={individual.pieces:.2f} "
            f"fitness={individual.fit:.2f}"
        )
        return individual

    def _breed(self, evaluated: List[Individual]) -> List[Individual]:
        offspring_target = max(1, int(self.size * self.o_ratio))
        survivor_count = max(1, self.size - offspring_target)
        survivors = [ind.clone() for ind in evaluated[:survivor_count]]
        t_size = max(2, int(self.size * self.t_ratio))
        t_size = min(t_size, len(evaluated))
        if t_size <= 1:
            t_size = 1
        offspring: List[Individual] = []
        while len(offspring) < offspring_target:
            competitors = random.sample(evaluated, t_size)
            competitors.sort(key=lambda ind: ind.fit, reverse=True)
            parent_a = competitors[0]
            parent_b = competitors[1] if len(competitors) > 1 else competitors[0]
            child_weights = _mix_w(
                parent_a.w,
                parent_b.w,
                parent_a.fit,
                parent_b.fit,
            )
            if random.random() < self.mut_chance:
                child_weights = _mut_w(child_weights, self.delta)
            offspring.append(Individual(w=child_weights))
        next_gen = survivors + offspring
        return next_gen[: self.size]

    def _maybe_save(self, candidate: Individual, generation: int) -> None:
        best_lines = float(self.best["lines_cleared"]) if self.best else None
        if best_lines is None or candidate.lines > best_lines:
            save_model(
                candidate.w,
                candidate.lines,
                candidate.pieces,
                generation,
                self.path,
            )
            self.best = {
                "weights": list(candidate.w),
                "lines_cleared": candidate.lines,
                "pieces_dropped": candidate.pieces,
                "generation": generation,
            }
            print(f"  New best saved with {candidate.lines:.2f} average lines cleared.")


def train_loop(
    generations: Optional[int] = None,
    games_per_agent: int = GAMES,
    mutation_delta: float = M_DELTA,
    lookahead_depth: int = LOOK_DEPTH,
) -> None:
    trainer = GeneticTrainer(
        population_size=POP_SIZE,
        games_per_agent=games_per_agent,
        model_path=MODEL_FILE,
        mutation_delta=mutation_delta,
        lookahead_depth=lookahead_depth,
    )
    trainer.train(generations=generations)


if __name__ == "__main__":
    train_loop()
