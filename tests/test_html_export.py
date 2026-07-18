"""
html_export.py writes real files to disk (tmp_path), so these tests check
the actual written content rather than mocking plotly out: that the file
exists, is non-trivial HTML, and -- the one property that actually matters
for the "works fully offline" claim -- that plotly.js is embedded inline
rather than referencing a CDN.
"""
import pandas as pd
import pytest

import html_export as he


def _assert_offline_html(path):
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "<script" in text
    # The bundled plotly.js source itself contains the literal string
    # "cdn.plot.ly" (as a default config value, e.g. plotlyServerURL) even
    # when fully embedded -- that substring alone doesn't prove a CDN
    # reference. The actual online/offline signal is whether any <script>
    # tag *loads from* the CDN via a src= attribute.
    assert 'src="https://cdn.plot.ly' not in text
    assert 'src="https://cdnjs' not in text
    assert len(text) > 500_000  # embedded plotly.js is multiple MB; a CDN-referencing file would be a few KB


class TestExportCorrelationHeatmapHtml:
    def test_empty_matrix_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No correlation matrix"):
            he.export_correlation_heatmap_html(tmp_path / "out.html", pd.DataFrame())

    def test_writes_offline_html(self, tmp_path):
        corr_df = pd.DataFrame({"Element": ["Cs", "Sr"], "Cs": [1.0, 0.9], "Sr": [0.9, 1.0]})
        out = he.export_correlation_heatmap_html(tmp_path / "corr.html", corr_df, title="Test heatmap")
        _assert_offline_html(out)


class TestExportPcaScatterHtml:
    def test_empty_tank_summary_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No PCA scores"):
            he.export_pca_scatter_html(tmp_path / "out.html", pd.DataFrame())

    def test_missing_pc_columns_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No PCA scores"):
            he.export_pca_scatter_html(tmp_path / "out.html", pd.DataFrame({"WasteSiteId": ["T1"]}))

    def test_writes_offline_html_with_color_by(self, tmp_path):
        tank_summary = pd.DataFrame({
            "WasteSiteId": ["T1", "T2", "T3"], "PC1": [1.0, -1.0, 0.5], "PC2": [0.2, -0.3, 0.1],
            "Cluster": [1, 2, 1], "TankFarm": ["A", "AN", "A"],
        })
        pca_variance = pd.DataFrame({"PC": ["PC1", "PC2"], "ExplainedVarianceRatio": [0.8, 0.15]})
        out = he.export_pca_scatter_html(tmp_path / "pca.html", tank_summary, color_by="TankFarm", pca_variance=pca_variance)
        _assert_offline_html(out)

    def test_writes_offline_html_without_color_by(self, tmp_path):
        tank_summary = pd.DataFrame({"WasteSiteId": ["T1", "T2"], "PC1": [1.0, -1.0], "PC2": [0.2, -0.3]})
        out = he.export_pca_scatter_html(tmp_path / "pca.html", tank_summary, color_by=None)
        _assert_offline_html(out)

    def test_color_by_missing_column_falls_back_to_single_trace(self, tmp_path):
        tank_summary = pd.DataFrame({"WasteSiteId": ["T1", "T2"], "PC1": [1.0, -1.0], "PC2": [0.2, -0.3]})
        out = he.export_pca_scatter_html(tmp_path / "pca.html", tank_summary, color_by="NotAColumn")
        _assert_offline_html(out)

    def test_no_variance_table(self, tmp_path):
        tank_summary = pd.DataFrame({"WasteSiteId": ["T1", "T2"], "PC1": [1.0, -1.0], "PC2": [0.2, -0.3]})
        out = he.export_pca_scatter_html(tmp_path / "pca.html", tank_summary)
        _assert_offline_html(out)


class TestExportNetworkHtml:
    def test_empty_nodes_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No network data"):
            he.export_network_html(tmp_path / "out.html", pd.DataFrame(), pd.DataFrame())

    def test_writes_offline_html_with_edges(self, tmp_path):
        nodes = pd.DataFrame({
            "Element": ["Cs", "Sr", "Fe"], "Total_inventory_kg": [500.0, 300.0, 60.0],
            "LogTotalInventory": [2.7, 2.5, 1.8], "x": [0.1, 0.4, -0.2], "y": [0.2, -0.3, 0.5], "N_edges": [2, 1, 0],
        })
        edges = pd.DataFrame([
            {"Element_A": "Cs", "Element_B": "Sr", "Sign": "positive"},
            {"Element_A": "Cs", "Element_B": "Fe", "Sign": "negative"},
        ])
        out = he.export_network_html(tmp_path / "net.html", nodes, edges)
        _assert_offline_html(out)

    def test_writes_offline_html_with_only_one_sign_of_edge(self, tmp_path):
        # Only positive edges present -- the negative-sign loop iteration
        # must skip cleanly (empty sub-frame), not crash.
        nodes = pd.DataFrame({
            "Element": ["Cs", "Sr"], "Total_inventory_kg": [500.0, 300.0],
            "LogTotalInventory": [2.7, 2.5], "x": [0.1, 0.4], "y": [0.2, -0.3], "N_edges": [1, 1],
        })
        edges = pd.DataFrame([{"Element_A": "Cs", "Element_B": "Sr", "Sign": "positive"}])
        out = he.export_network_html(tmp_path / "net.html", nodes, edges)
        _assert_offline_html(out)

    def test_writes_offline_html_with_no_edges(self, tmp_path):
        nodes = pd.DataFrame({
            "Element": ["Cs"], "Total_inventory_kg": [500.0], "LogTotalInventory": [2.7],
            "x": [0.0], "y": [0.0], "N_edges": [0],
        })
        out = he.export_network_html(tmp_path / "net.html", nodes, pd.DataFrame())
        _assert_offline_html(out)

    def test_none_edges_treated_as_no_edges(self, tmp_path):
        nodes = pd.DataFrame({
            "Element": ["Cs"], "Total_inventory_kg": [500.0], "LogTotalInventory": [2.7],
            "x": [0.0], "y": [0.0], "N_edges": [0],
        })
        out = he.export_network_html(tmp_path / "net.html", nodes, None)
        _assert_offline_html(out)


class TestWriteHelper:
    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c.html"
        nodes = pd.DataFrame({
            "Element": ["Cs"], "Total_inventory_kg": [500.0], "LogTotalInventory": [2.7],
            "x": [0.0], "y": [0.0], "N_edges": [0],
        })
        out = he.export_network_html(nested, nodes, None)
        assert out == nested
        assert nested.exists()
