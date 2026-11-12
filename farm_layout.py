"""
farm_layout.py -- schematic tile layout of the Hanford tank farms
(framework-agnostic: no Qt, no plotting).

One pure function turns a tank table (WasteSiteId such as "241-AW-106",
optionally TankFarm and TankSystem) into tile positions, farm label
anchors, area titles and the canvas extent. The arrangement is a SCHEMATIC
grouped by farm, not a geographic map and not to scale: two areas side by
side, farms in bands inside each area, tanks in rows by tank number inside
each farm, and the small 200-series tanks (numbers 201 to 204) in a row of
their own beneath the farm they belong to.

Farm membership comes from the data. A farm that is not in the constants
below still gets its tiles, in an extra block at the right, so a dataset
with an unexpected tank identifier never loses a tank from the map.

Units: one tile cell is 1.0 wide and 1.0 tall; x grows to the right and y
grows downward (screen convention).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# Area -> bands (top to bottom) -> farms (left to right).
AREAS: Dict[str, List[List[str]]] = {
    "200 West Area": [["T", "TX", "TY"], ["U", "S", "SX"], ["SY"]],
    "200 East Area": [["B", "BX", "BY", "C"], ["A", "AX", "AY", "AZ"], ["AN", "AP", "AW"]],
}
OTHER_AREA = "Other tanks"
UNASSIGNED_FARM = "Unassigned"

# Farms built as double-shell tanks. Used only when the table carries no
# TankSystem column; with one, the data decides.
DOUBLE_SHELL_FARMS = frozenset({"AN", "AP", "AW", "AY", "AZ", "SY"})

# Tiles per row inside a farm (default 3).
FARM_COLUMNS: Dict[str, int] = {"AX": 2, "AY": 2, "AZ": 2}
DEFAULT_COLUMNS = 3

AREA_GAP = 2.0          # between the two areas
FARM_GAP = 0.8          # between farms of one band
BAND_GAP = 0.8          # between bands of one area
TITLE_HEIGHT = 1.3      # area title and its rule
LABEL_HEIGHT = 1.0      # farm name above its tiles
TANK_RADIUS = 0.43      # of a full-size tank, in tile units
SMALL_SCALE = 0.62      # 200-series tanks are drawn smaller
SMALL_PITCH = 0.75      # centre-to-centre spacing of the small tanks
OTHER_FARMS_PER_BAND = 4

_NUMBER_RE = re.compile(r"(\d+)\s*$")
_FARM_RE = re.compile(r"^\s*\d+\s*-\s*([A-Za-z]+)\s*-")


def tank_number(tank_id: object) -> Optional[int]:
    """The trailing tank number of an identifier ("241-AW-106" -> 106)."""
    match = _NUMBER_RE.search(str(tank_id or ""))
    return int(match.group(1)) if match else None


def farm_from_id(tank_id: object) -> Optional[str]:
    """The farm letters of an identifier ("241-AW-106" -> "AW")."""
    match = _FARM_RE.match(str(tank_id or ""))
    return match.group(1).upper() if match else None


def is_small_tank(tank_id: object) -> bool:
    """The 200-series tanks (201 to 204) are the small ones of a farm."""
    number = tank_number(tank_id)
    return number is not None and 200 <= number < 300


@dataclass(frozen=True)
class Tile:
    tank: str
    farm: str
    area: str
    number: Optional[int]
    x: float                  # centre
    y: float                  # centre
    radius: float             # outline radius, in tile units
    small: bool
    double_shell: bool


@dataclass(frozen=True)
class FarmBlock:
    name: str
    area: str
    x0: float                 # tile block, left
    y0: float                 # tile block, top
    x1: float
    y1: float
    label_x: float            # centre of the label
    label_y: float            # baseline band centre of the label
    tanks: Tuple[str, ...]
    double_shell: bool


@dataclass(frozen=True)
class AreaBlock:
    title: str
    x0: float
    y0: float                 # top of the title band
    x1: float
    y1: float                 # bottom of the lowest farm
    rule_y: float             # the hairline under the title


@dataclass(frozen=True)
class FarmLayout:
    tiles: Dict[str, Tile] = field(default_factory=dict)
    farms: List[FarmBlock] = field(default_factory=list)
    areas: List[AreaBlock] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

    def tanks(self) -> List[str]:
        return list(self.tiles)

    def farm(self, name: str) -> Optional[FarmBlock]:
        for block in self.farms:
            if block.name == name:
                return block
        return None

    def neighbour(self, tank: str, dx: int, dy: int) -> Optional[str]:
        """The nearest tile in a direction (for arrow-key navigation).

        Candidates lie strictly on the requested side; among them the one
        closest along the direction wins, with sideways drift penalised
        so a move stays in its row or column when it can."""
        origin = self.tiles.get(tank)
        if origin is None:
            return None
        best, best_score = None, math.inf
        for other in self.tiles.values():
            if other.tank == tank:
                continue
            ox, oy = other.x - origin.x, other.y - origin.y
            along = ox * dx + oy * dy
            across = abs(ox * dy) + abs(oy * dx)
            if along <= 1e-6:
                continue
            score = along + 2.5 * across
            if score < best_score:
                best, best_score = other.tank, score
        return best


def _tank_records(tanks: object) -> List[Tuple[str, str, Optional[bool]]]:
    """(tank, farm, double_shell or None) for every distinct tank."""
    if isinstance(tanks, pd.DataFrame):
        if "WasteSiteId" not in tanks.columns:
            raise ValueError("farm_layout needs a WasteSiteId column.")
        ids = tanks["WasteSiteId"].tolist()
        farms = tanks["TankFarm"].tolist() if "TankFarm" in tanks.columns else [None] * len(ids)
        systems = tanks["TankSystem"].tolist() if "TankSystem" in tanks.columns else [None] * len(ids)
    else:
        ids = list(tanks or [])
        farms = [None] * len(ids)
        systems = [None] * len(ids)
    seen, out = set(), []
    for tank, farm, system in zip(ids, farms, systems):
        if tank is None or (isinstance(tank, float) and math.isnan(tank)):
            continue
        tank = str(tank)
        if tank in seen:
            continue
        seen.add(tank)
        farm_name = None if farm is None or (isinstance(farm, float) and math.isnan(farm)) else str(farm).strip()
        farm_name = farm_name or farm_from_id(tank) or UNASSIGNED_FARM
        shell: Optional[bool] = None
        if isinstance(system, str) and system.strip().upper() in ("DST", "SST"):
            shell = system.strip().upper() == "DST"
        out.append((tank, farm_name, shell))
    return out


def _sort_key(tank: str) -> Tuple[int, int, str]:
    number = tank_number(tank)
    return (0, number, tank) if number is not None else (1, 0, tank)


def farm_layout(tanks: object) -> FarmLayout:
    """Tile positions for a tank table (or a plain list of identifiers).

    Every distinct tank gets exactly one tile. Farms listed in AREAS keep
    their place; any other farm found in the data lands in an extra block
    at the right. Farms with no tank in the data take no room.
    """
    records = _tank_records(tanks)
    if not records:
        return FarmLayout()

    by_farm: Dict[str, List[Tuple[str, Optional[bool]]]] = {}
    for tank, farm, shell in records:
        by_farm.setdefault(farm, []).append((tank, shell))

    known = {farm for bands in AREAS.values() for band in bands for farm in band}
    plan: List[Tuple[str, List[List[str]]]] = []
    for area, bands in AREAS.items():
        kept = [[farm for farm in band if farm in by_farm] for band in bands]
        kept = [band for band in kept if band]
        if kept:
            plan.append((area, kept))
    others = sorted(farm for farm in by_farm if farm not in known)
    if others:
        plan.append((OTHER_AREA, [
            others[i:i + OTHER_FARMS_PER_BAND] for i in range(0, len(others), OTHER_FARMS_PER_BAND)
        ]))

    tiles: Dict[str, Tile] = {}
    farms: List[FarmBlock] = []
    areas: List[AreaBlock] = []
    x_area, height = 0.0, 0.0

    for area, bands in plan:
        y = TITLE_HEIGHT
        width = 0.0
        for band in bands:
            x, rows_max = x_area, 0.0
            for farm in band:
                members = sorted(by_farm[farm], key=lambda item: _sort_key(item[0]))
                main = [m for m in members if not is_small_tank(m[0])]
                small = [m for m in members if is_small_tank(m[0])]
                ncol = FARM_COLUMNS.get(farm, DEFAULT_COLUMNS)
                rows_main = math.ceil(len(main) / ncol) if main else 0
                per_small_row = max(1, int(ncol / SMALL_PITCH))
                rows_small = math.ceil(len(small) / per_small_row) if small else 0
                top = y + LABEL_HEIGHT

                def shell_of(member_shell: Optional[bool]) -> bool:
                    return member_shell if member_shell is not None else farm in DOUBLE_SHELL_FARMS

                for i, (tank, shell) in enumerate(main):
                    tiles[tank] = Tile(
                        tank=tank, farm=farm, area=area, number=tank_number(tank),
                        x=x + 0.5 + i % ncol, y=top + 0.5 + i // ncol,
                        radius=TANK_RADIUS, small=False, double_shell=shell_of(shell),
                    )
                for i, (tank, shell) in enumerate(small):
                    col, row = i % per_small_row, i // per_small_row
                    tiles[tank] = Tile(
                        tank=tank, farm=farm, area=area, number=tank_number(tank),
                        x=x + SMALL_PITCH / 2 + SMALL_PITCH * col,
                        y=top + rows_main + 0.5 + row,
                        radius=TANK_RADIUS * SMALL_SCALE, small=True, double_shell=shell_of(shell),
                    )
                rows_total = rows_main + rows_small
                shells = [shell_of(shell) for _tank, shell in members]
                farms.append(FarmBlock(
                    name=farm, area=area, x0=x, y0=top, x1=x + ncol, y1=top + rows_total,
                    label_x=x + ncol / 2.0, label_y=y + LABEL_HEIGHT / 2.0,
                    tanks=tuple(tank for tank, _shell in members),
                    double_shell=bool(shells) and all(shells),
                ))
                rows_max = max(rows_max, rows_total)
                x += ncol + FARM_GAP
            width = max(width, x - FARM_GAP - x_area)
            y += LABEL_HEIGHT + rows_max + BAND_GAP
        bottom = y - BAND_GAP
        areas.append(AreaBlock(
            title=area, x0=x_area, y0=0.0, x1=x_area + width, y1=bottom,
            rule_y=TITLE_HEIGHT - 0.45,
        ))
        x_area += width + AREA_GAP
        height = max(height, bottom)

    return FarmLayout(tiles=tiles, farms=farms, areas=areas,
                      width=x_area - AREA_GAP, height=height)


def overlapping_tiles(layout: FarmLayout) -> List[Tuple[str, str]]:
    """Pairs of tiles whose outlines intersect (empty for a sound layout)."""
    items = list(layout.tiles.values())
    out = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if math.hypot(a.x - b.x, a.y - b.y) < a.radius + b.radius - 1e-9:
                out.append((a.tank, b.tank))
    return out


def tanks_table(tank_ids: Iterable[str], farms: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """A minimal tank table from identifiers (farm parsed from the id)."""
    ids = [str(t) for t in tank_ids]
    return pd.DataFrame({
        "WasteSiteId": ids,
        "TankFarm": list(farms) if farms is not None else [farm_from_id(t) for t in ids],
    })
