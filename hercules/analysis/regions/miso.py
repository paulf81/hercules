"""MISO rules applied after the simulation: RA hours, peak blocks, shared-POI accreditation.

MISO data lives in ``miso.yaml``; region-agnostic capacity math lives in
``capacity_market``.
"""

import datetime

import numpy as np
import pandas as pd

from hercules.analysis import capacity_market
from hercules.analysis.regions.region_data import load_region
from hercules.analysis.utilities import _component_names, _get_column, _require_default_index
from hercules.hercules_output import HerculesOutput


def _zone_info(region: dict, zone: int) -> dict:
    """Return the subregion and peak zone of a MISO local resource zone."""
    return capacity_market._lookup(region["zones"], zone, "MISO zone")


def add_is_peak_column(ho: HerculesOutput, zone: int, region: dict | None = None) -> HerculesOutput:
    """Add ``analysis.is_peak``: 1 inside the zone's daily peak-load blocks, else 0.

    Each daily peak-load time starts a block of ``peak_block_duration_hours``. Peak times are
    the hour of each day's highest MISO medium-term load forecast (MTLF) for the zone. This is
    MISO market data, available whether or not the controller used it; a controller that
    did (e.g. Hycon's peak-locked battery controller) also logs its own ``in_peak``.

    Args:
        ho (HerculesOutput): Hercules output.
        zone (int): MISO local resource zone.
        region (dict | None): Region data. Defaults to ``load_region("miso")``.

    Returns:
        HerculesOutput: ``ho`` with ``analysis.is_peak`` added.

    Raises:
        ValueError: If the zone is unknown or peak-load data does not cover every
            simulated market day.
    """
    _require_default_index(ho)
    region = region or load_region("miso")
    peak_zone = _zone_info(region, zone)["peak_zone"]
    peaks = pd.read_feather(region["data_files"]["daily_peak_hours"])
    starts = pd.DatetimeIndex(peaks.loc[peaks["zone"] == peak_zone, "time_utc"]).sort_values()
    if starts.empty:
        raise ValueError(f"No peak-load times for peak zone '{peak_zone}'.")

    tz = datetime.timezone(datetime.timedelta(hours=region["market_time_offset_hours"]))
    time_index = pd.DatetimeIndex(ho.df["time_utc"]).as_unit("ns")
    sim_days = time_index.tz_convert(tz).normalize().unique()
    missing = sim_days[~sim_days.isin(starts.tz_convert(tz).normalize())]
    if not missing.empty:
        raise ValueError(
            f"No peak-load data for peak zone '{peak_zone}' on {len(missing)} market day(s), "
            f"first {missing[0].date()}."
        )

    starts = starts.as_unit("ns").asi8
    times = time_index.asi8
    block_ns = int(region["peak_block_duration_hours"] * 3600 * 10**9)
    idx = np.searchsorted(starts, times, side="right") - 1
    in_block = (idx >= 0) & (times < starts[np.maximum(idx, 0)] + block_ns)
    ho.df["analysis.is_peak"] = in_block.astype(int)
    return ho


def add_RA_hour_columns(
    ho: HerculesOutput, zone: int, region: dict | None = None
) -> HerculesOutput:
    """Add MISO RA-hour columns for the zone's subregion, held over each clock hour.

    Adds ``analysis.season`` (from the date in MISO market time) and ``analysis.RA_hour``
    and ``analysis.AAOC_hour`` (0/1, NaN outside ``RA_hours_coverage_utc``).

    Args:
        ho (HerculesOutput): Hercules output.
        zone (int): MISO local resource zone.
        region (dict | None): Region data. Defaults to ``load_region("miso")``.

    Returns:
        HerculesOutput: ``ho`` with the RA-hour columns added.
    """
    _require_default_index(ho)
    region = region or load_region("miso")
    subregion = _zone_info(region, zone)["subregion"]
    hours = pd.DatetimeIndex(ho.df["time_utc"]).floor("h").as_unit("ns")

    offset = datetime.timedelta(hours=region["market_time_offset_hours"])
    local = hours.tz_convert(datetime.timezone(offset))
    month_season = {m: season for season, months in region["seasons"].items() for m in months}
    ho.df["analysis.season"] = local.month.map(month_season).to_numpy()

    flagged = pd.read_csv(region["data_files"]["RA_hours"])
    flagged = flagged[flagged["subregion"] == subregion]
    for hour_type in ("RA", "AAOC"):
        times = pd.to_datetime(flagged.loc[flagged["hour_type"] == hour_type, "time_utc"], utc=True)
        values = hours.isin(pd.DatetimeIndex(times).as_unit("ns")).astype(float)
        start, end = (pd.Timestamp(t) for t in region["RA_hours_coverage_utc"][hour_type])
        values[(hours < start) | (hours >= end)] = np.nan
        ho.df[f"analysis.{hour_type}_hour"] = values
    return ho


def compute_capacity_accredited_seasonal_MW(
    ho: HerculesOutput,
    zone: int,
    component: str,
    region: dict | None = None,
    storage_available_columns: dict[str, str] | None = None,
) -> dict[str, float]:
    """Return MISO accredited capacity by season (MW), with hourly availability at a shared POI.

    Starts from the class-level value (``capacity_market``). MISO limits availability by hour,
    with storage served first. Storage keeps its class-level value, scaled by
    ``min(1, POI / sum of storage power_discharge_max)`` when several storage components share
    the POI. A generator sharing the POI with storage is scaled, per season, over the RA hours
    in the simulation by

        ratio = sum_h min(P_h, max(POI - A_h, 0)) / sum_h min(P_h, POI)

    where ``P_h`` is the hourly mean ``power_hypothetical`` of the generators (shared in
    proportion between generators), ``A_h`` the storage's available discharge and ``POI`` the
    interconnect limit converted to the bus with ``efficiency_bus_to_poi``. Seasons without
    RA hours in the simulation keep the class-level value. Without storage the result equals
    ``capacity_market.compute_capacity_accredited_seasonal_MW``.

    Args:
        ho (HerculesOutput): Hercules output; the zone's RA-hour columns are (re)added.
        zone (int): MISO local resource zone.
        component (str): Component name.
        region (dict | None): Region data. Defaults to ``load_region("miso")``.
        storage_available_columns (dict[str, str] | None): Available discharge column (kW)
            per storage component, e.g. ``{"battery": "controller.battery.offer_RT"}``.
            Defaults to None (rated ``power_discharge_max``).

    Returns:
        dict[str, float]: Accredited capacity (MW) by season.

    Raises:
        ValueError: If a generator shares the POI with storage and the RA-hour data does not
            cover the whole simulation.
    """
    region = region or load_region("miso")
    seasonal = capacity_market.compute_capacity_accredited_seasonal_MW(ho, region, component)
    h_dict = ho.metadata["h_dict"]
    classes = {
        c: capacity_market._RESOURCE_CLASS.get(h_dict[c]["component_type"])
        for c in _component_names(ho)
    }
    storage = [c for c, resource_class in classes.items() if resource_class == "battery"]
    if not storage:
        return seasonal
    plant = h_dict["plant"]
    POI = float(plant["interconnect_limit"]) / float(plant.get("efficiency_bus_to_poi", 1.0))
    if component in storage:
        storage_rated = sum(float(h_dict[s]["power_discharge_max"]) for s in storage)
        scale = min(1.0, POI / storage_rated)
        return {s: value * scale for s, value in seasonal.items()}

    add_RA_hour_columns(ho, zone, region)
    if ho.df["analysis.RA_hour"].isna().any():
        raise ValueError(
            "RA-hour data does not cover the whole simulation (see RA_hours_coverage_utc)."
        )
    in_RA_hour = ho.df["analysis.RA_hour"].to_numpy() == 1
    generators = [c for c, resource_class in classes.items() if resource_class in ("solar", "wind")]
    frame = pd.DataFrame(
        {g: _get_column(ho, f"{g}.power_hypothetical")[in_RA_hour] for g in generators}
    )
    frame["available"] = 0.0
    for s in storage:
        column = (storage_available_columns or {}).get(s)
        if column is None:
            frame["available"] += float(h_dict[s]["power_discharge_max"])
        else:
            frame["available"] += np.maximum(_get_column(ho, column)[in_RA_hour], 0.0)
    frame["hour"] = pd.DatetimeIndex(ho.df.loc[in_RA_hour, "time_utc"]).floor("h")
    frame["season"] = ho.df.loc[in_RA_hour, "analysis.season"].to_numpy()
    hourly = frame.groupby(["hour", "season"]).mean()

    total = hourly[generators].sum(axis=1)
    share = (hourly[component] / total.where(total > 0)).fillna(0.0)
    capped = share * np.minimum(total, np.maximum(POI - hourly["available"], 0.0))
    uncapped = share * np.minimum(total, POI)
    seasons = hourly.index.get_level_values("season")
    numerator = capped.groupby(seasons).sum()
    denominator = uncapped.groupby(seasons).sum()
    ratio = {s: numerator[s] / denominator[s] for s in denominator.index if denominator[s] > 0}
    return {s: value * ratio.get(s, 1.0) for s, value in seasonal.items()}
