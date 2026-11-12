"""
farm_layout: the schematic tile layout of the tank farms. The properties
that make it a usable map are checked on the real tank list: every tank
has exactly one tile, no two tiles overlap, and each farm stays inside
its area.
"""
import math

import pandas as pd
import pytest

import data_model as dm
import farm_layout as fl
import tank_science as tsci

from conftest import REAL_CSV_PATH, requires_real_data


@pytest.fixture(scope="module")
def real_tanks():
    dataset = dm.HanfordDataset()
    dataset.load(REAL_CSV_PATH, use_cache=True)
    return tsci.tank_attributes_table(dataset)


def _table(ids, farms=None, systems=None):
    data = {"WasteSiteId": ids}
    if farms is not None:
        data["TankFarm"] = farms
    if systems is not None:
        data["TankSystem"] = systems
    return pd.DataFrame(data)


class TestIdentifierParsing:
    def test_tank_number_and_farm(self):
        assert fl.tank_number("241-AW-106") == 106
        assert fl.farm_from_id("241-AW-106") == "AW"
        assert fl.tank_number("no number") is None
        assert fl.farm_from_id("oddly named") is None

    def test_the_200_series_are_the_small_tanks(self):
        assert fl.is_small_tank("241-C-201")
        assert fl.is_small_tank("241-T-204")
        assert not fl.is_small_tank("241-C-112")
        assert not fl.is_small_tank("241-AW-106")


@requires_real_data
class TestRealTankList:
    def test_every_tank_gets_exactly_one_position(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        tanks = [str(t) for t in real_tanks["WasteSiteId"]]
        assert len(tanks) == 177
        assert sorted(layout.tiles) == sorted(tanks)
        assert sum(len(block.tanks) for block in layout.farms) == 177

    def test_no_two_tiles_overlap(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        assert fl.overlapping_tiles(layout) == []
        positions = {(round(t.x, 6), round(t.y, 6)) for t in layout.tiles.values()}
        assert len(positions) == len(layout.tiles)

    def test_farms_stay_inside_their_area(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        areas = {area.title: area for area in layout.areas}
        assert list(areas) == ["200 West Area", "200 East Area"]
        for block in layout.farms:
            area = areas[block.area]
            assert area.x0 - 1e-9 <= block.x0 and block.x1 <= area.x1 + 1e-9, block.name
            assert area.y0 <= block.y0 and block.y1 <= area.y1 + 1e-9, block.name
        for tile in layout.tiles.values():
            area = areas[tile.area]
            assert area.x0 <= tile.x - tile.radius and tile.x + tile.radius <= area.x1, tile.tank
            assert area.rule_y < tile.y - tile.radius and tile.y + tile.radius <= area.y1, tile.tank

    def test_the_two_areas_do_not_touch(self, real_tanks):
        west, east = fl.farm_layout(real_tanks).areas
        assert west.x1 < east.x0
        assert east.x1 == pytest.approx(fl.farm_layout(real_tanks).width)

    def test_farms_are_in_the_documented_areas(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        by_area = {}
        for block in layout.farms:
            by_area.setdefault(block.area, set()).add(block.name)
        assert by_area["200 West Area"] == {"T", "TX", "TY", "U", "S", "SX", "SY"}
        assert by_area["200 East Area"] == {"B", "BX", "BY", "C", "A", "AX", "AY", "AZ", "AN", "AP", "AW"}

    def test_tanks_run_in_rows_by_number(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        tx = [layout.tiles[f"241-TX-{n}"] for n in (101, 102, 103, 104)]
        assert tx[0].y == tx[1].y == tx[2].y          # a row of three
        assert tx[0].x < tx[1].x < tx[2].x
        assert tx[3].y > tx[0].y and tx[3].x == tx[0].x  # the fourth starts the next row

    def test_small_tanks_sit_in_their_own_row_beneath(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        for farm in ("B", "C", "T", "U"):
            members = [layout.tiles[t] for t in layout.farm(farm).tanks]
            small = [t for t in members if t.small]
            main = [t for t in members if not t.small]
            assert len(small) == 4, farm
            assert len({t.y for t in small}) == 1, farm
            assert min(t.y for t in small) > max(t.y for t in main), farm
            assert all(t.radius < main[0].radius for t in small)

    def test_double_shell_follows_the_data(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        system = dict(zip(real_tanks["WasteSiteId"], real_tanks["TankSystem"]))
        for tank, tile in layout.tiles.items():
            assert tile.double_shell == (system[tank] == "DST"), tank
        assert {b.name for b in layout.farms if b.double_shell} == set(fl.DOUBLE_SHELL_FARMS)

    def test_farm_labels_do_not_collide(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        labels = [(b.label_x, b.label_y) for b in layout.farms]
        for i, (xa, ya) in enumerate(labels):
            for xb, yb in labels[i + 1:]:
                assert not (ya == yb and abs(xa - xb) < 2.0)

    def test_arrow_key_neighbours(self, real_tanks):
        layout = fl.farm_layout(real_tanks)
        assert layout.neighbour("241-TX-101", 1, 0) == "241-TX-102"
        assert layout.neighbour("241-TX-102", -1, 0) == "241-TX-101"
        assert layout.neighbour("241-TX-101", 0, 1) == "241-TX-104"
        # every tank can be reached and left by keyboard
        for tank in layout.tiles:
            assert any(layout.neighbour(tank, dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))


class TestDerivedFromTheData:
    def test_a_farm_missing_from_the_constants_still_gets_tiles(self):
        table = _table(["241-A-101", "241-ZZ-101", "241-ZZ-102", "241-QQ-201"],
                       farms=["A", "ZZ", "ZZ", "QQ"])
        layout = fl.farm_layout(table)
        assert sorted(layout.tiles) == sorted(table["WasteSiteId"])
        assert layout.tiles["241-ZZ-101"].area == fl.OTHER_AREA
        assert layout.tiles["241-QQ-201"].area == fl.OTHER_AREA
        assert [a.title for a in layout.areas] == ["200 East Area", fl.OTHER_AREA]
        assert fl.overlapping_tiles(layout) == []

    def test_farm_is_parsed_from_the_identifier_when_the_column_is_absent(self):
        layout = fl.farm_layout(["241-AN-104", "241-A-101"])
        assert layout.tiles["241-AN-104"].farm == "AN"
        assert layout.tiles["241-AN-104"].double_shell      # falls back to the farm constants
        assert not layout.tiles["241-A-101"].double_shell

    def test_an_unparseable_identifier_is_kept(self):
        layout = fl.farm_layout(_table(["Tank X", "241-A-101"]))
        assert "Tank X" in layout.tiles
        assert layout.tiles["Tank X"].farm == fl.UNASSIGNED_FARM

    def test_duplicates_and_nulls_are_ignored(self):
        layout = fl.farm_layout(_table(["241-A-101", "241-A-101", None]))
        assert list(layout.tiles) == ["241-A-101"]

    def test_empty_input(self):
        layout = fl.farm_layout(_table([]))
        assert layout.tiles == {} and layout.width == 0.0 and layout.height == 0.0
        assert fl.farm_layout([]).tiles == {}

    def test_farms_with_no_tank_take_no_room(self):
        layout = fl.farm_layout(_table(["241-AN-104"], farms=["AN"]))
        assert [b.name for b in layout.farms] == ["AN"]
        assert layout.width == pytest.approx(3.0)

    def test_extent_covers_every_tile(self):
        table = _table([f"241-TX-{100 + i}" for i in range(1, 19)] + ["241-SY-101"])
        layout = fl.farm_layout(table)
        for tile in layout.tiles.values():
            assert 0 <= tile.x - tile.radius and tile.x + tile.radius <= layout.width
            assert 0 <= tile.y - tile.radius and tile.y + tile.radius <= layout.height
        assert math.isfinite(layout.width) and math.isfinite(layout.height)
