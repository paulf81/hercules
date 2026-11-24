"""Tests for hercules.analysis.plotting."""

import pytest
from hercules.analysis import plotting

from tests.test_inputs.analysis_output import build_hercules_output

H_DICT = {
    "plant": {"interconnect_limit": 1500.0},
    "solar": {"component_type": "SolarPySAMPVWatts"},
}


def test_plot_timeseries_plotly(tmp_path):
    # Rows: solar power, plant export, LMP. One shaded run of in_peak.
    ho = build_hercules_output(
        tmp_path,
        {
            "solar.power": [0.0, 500.0, 1000.0, 500.0],
            "controller.battery.in_peak": [0.0, 1.0, 1.0, 0.0],
            "external_signals.LMP_RT": [10.0, 20.0, 30.0, 40.0],
        },
        H_DICT,
    )
    fig = plotting.plot_timeseries_plotly(
        ho,
        ["solar.power"],
        shade_columns={"controller.battery.in_peak": "gold"},
        save_file=str(tmp_path / "plot.html"),
    )
    assert (tmp_path / "plot.html").is_file()
    assert fig.layout.height == 3 * 250
    bands = [s for s in fig.layout.shapes if s.yref == "paper"]
    assert len(bands) == 1

    with pytest.raises(ValueError, match="not found"):
        plotting.plot_timeseries_plotly(ho, ["missing.power"])


def test_plot_shading_ignores_NaN_and_subsampling(tmp_path):
    # NaN flags are not shaded, and a one-step flag on an odd index survives a stride of 2.
    nan = float("nan")
    ho = build_hercules_output(
        tmp_path,
        {"solar.power": [0.0] * 6, "controller.flag": [nan, 1.0, 0.0, nan, 0.0, 0.0]},
        H_DICT,
    )
    fig = plotting.plot_timeseries_plotly(
        ho, ["solar.power"], plot_dt=120.0, shade_columns={"controller.flag": "gold"}
    )
    bands = [s for s in fig.layout.shapes if s.yref == "paper"]
    assert len(bands) == 1
