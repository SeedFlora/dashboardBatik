"""Raw Vega-Lite dict builders for the BI dashboard: palette tokens, Indonesian locale; no Streamlit, no Altair."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from temajelajah.bi_data import ASPECT_NAMES, MONTHS, SENTIMENTS, fmt_dec, fmt_int, fmt_month, fmt_pct, fmt_share, venue_short

NILA, SOGA, KARAT, NETRAL = "#0B6E99", "#C2702F", "#9A3B2A", "#8FA3A6"  # Samudra Jelajah: batik hues on a cool base
TRACK, INK, MUTED, GRID, WHITE, PAPER = "#D5E6E3", "#11313A", "#4F6F73", "#DCE9E7", "#FFFFFF", "#EAF4F3"
SEQ_COOL, SEQ_WARM = ("#E3F1EF", "#0F5A66"), ("#FDEDDE", KARAT)  # one-hue ramps (hue spread 25° and 33°)
WHITE_TEXT_AT = {SEQ_COOL: 0.66, SEQ_WARM: 0.70}  # ramp position (hcl) from which white text out-contrasts INK
WEAK_BAND = {SEQ_COOL: (0.56, 0.76), SEQ_WARM: (0.60, 0.80)}  # ramp positions where neither INK nor white text reaches 4.5:1
NO_TIP = {"tooltip": None}  # decorative marks: no default tooltip listing internal field names
TOP_GAP = {"padding": {"top": 14}}  # Streamlit always sets a top-level padding object, so config.padding never reaches the chart
PALETTE = frozenset({NILA, SOGA, KARAT, NETRAL, TRACK, INK, MUTED, GRID, WHITE, PAPER, *SEQ_COOL, *SEQ_WARM})
FONT = '"Source Sans", "Source Sans 3", "Source Sans Pro", sans-serif'
PANDEMIC = ("2020-03-01", "2022-01-01")
LOCALE = {"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["Rp", ""]},
          "time": {"dateTime": "%A, %e %B %Y %X", "date": "%d/%m/%Y", "time": "%H.%M.%S", "periods": ["AM", "PM"],
                   "days": ["Minggu", "Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"],
                   "shortDays": ["Min", "Sen", "Sel", "Rab", "Kam", "Jum", "Sab"],
                   "months": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
                              "November", "Desember"], "shortMonths": list(MONTHS)}}


def base_config(print_mode: bool = False) -> dict:
    size = 15 if print_mode else 12
    axis = {"labelFont": FONT, "titleFont": FONT, "labelFontSize": size, "titleFontSize": size, "labelColor": MUTED, "titleColor": MUTED,
            "titleFontWeight": "normal", "grid": not print_mode, "gridColor": GRID, "gridWidth": 1, "domainColor": GRID, "tickColor": GRID,
            "tickSize": 4, "labelPadding": 6, "titlePadding": 10}
    return {"background": WHITE, "font": FONT, "view": {"stroke": None},
            "axis": axis, "axisBand": {"grid": False, "domain": False, "ticks": False},
            "legend": {"orient": "top", "labelFont": FONT, "labelFontSize": size, "labelColor": INK, "title": None,
                       "symbolType": "square", "symbolSize": 120, "columnPadding": 18},
            "header": {"labelFont": FONT, "labelFontSize": size + 1, "labelColor": INK, "labelFontWeight": 600, "labelAnchor": "start",
                       "title": None},
            "bar": {"cornerRadiusEnd": 4, "discreteBandSize": 18}, "line": {"strokeWidth": 2, "strokeCap": "round", "strokeJoin": "round"},
            "point": {"size": 80, "filled": True, "stroke": WHITE, "strokeWidth": 2},
            "text": {"font": FONT, "fontSize": size, "color": INK}, "rule": {"color": INK}, "locale": LOCALE}


def _size(cfg: dict) -> int:
    return int((cfg.get("text") or {}).get("fontSize", 12))


def _step(cfg: dict, base: int) -> int:
    return round(base * _size(cfg) / 12)


def _plain(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return str(value)


def _records(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> list[dict]:
    names = list(columns or frame.columns)
    return [{name: _plain(value) for name, value in zip(names, row)} for row in frame[names].itertuples(index=False, name=None)]


def _tip(field: str, title: str) -> dict:
    return {"field": field, "type": "nominal", "title": title}


def _halo(layer: dict, width: float = 4) -> list[dict]:
    """A white-stroked copy under a text layer keeps labels legible where they cross rules or gridlines."""
    mark = {key: value for key, value in layer["mark"].items() if key != "color"}  # fill and stroke replace color
    under = {**layer, "mark": {**mark, "fill": WHITE, "stroke": WHITE, "strokeWidth": width, "strokeJoin": "round", **NO_TIP}}
    under["encoding"] = {key: value for key, value in layer["encoding"].items() if key not in ("color", "tooltip")}
    return [under, layer]


def _text_color(field: str, ramp: tuple[str, str], domain_max: float = 1.0, *, muted_when_zero: bool = False,
                dark: str = WHITE, light: str = INK, zero: str = MUTED) -> dict:
    """Label colour by the cell's ramp position: `dark` on dark fills, `light` on light ones, `zero` on empty cells."""
    conditions = [{"test": f"datum.{field} / {domain_max} >= {WHITE_TEXT_AT[ramp]}", "value": dark}]
    conditions += [{"test": "datum.n == 0", "value": zero}] if muted_when_zero else []
    return {"condition": conditions if len(conditions) > 1 else conditions[0], "value": light}


def _cell_labels(layer: dict, field: str, ramp: tuple[str, str], domain_max: float = 1.0, *, muted_when_zero: bool = False) -> list[dict]:
    """Heatmap labels. In the ramp's middle band neither white nor INK reaches 4.5:1 (WEAK_BAND, from the hcl ramp), so labels
    there sit on a thin outline in the opposite colour: the ramp's dark end under white text, white under INK text."""
    text = {**layer, "encoding": {**layer["encoding"], "color": _text_color(field, ramp, domain_max, muted_when_zero=muted_when_zero)}}
    low, high = WEAK_BAND[ramp]
    position = f"datum.{field} / {domain_max}"
    under = {**layer, "transform": [*layer.get("transform", []), {"filter": f"{position} >= {low} && {position} < {high}"}],
             "mark": {**layer["mark"], "strokeWidth": 2.5, "strokeJoin": "round", "strokeOpacity": 0.85, **NO_TIP}}
    outline = _text_color(field, ramp, domain_max, dark=ramp[1], light=WHITE)
    under["encoding"] = {**{key: value for key, value in layer["encoding"].items() if key not in ("color", "tooltip")},
                         "fill": outline, "stroke": outline}
    return [under, text]


def _rev(values: list[dict]) -> dict:
    """Content digest of the rows. Streamlit swaps chart rows only when their Arrow shape changes, so equal-sized new rows (another
    filter, another selected cell) would leave the old picture on screen; a spec that changes with its rows rebuilds the view."""
    digest = hashlib.sha1(json.dumps(values, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return {"rev": digest[:16]}


def _spec(cfg: dict, height: int, values: list[dict], layers: list[dict], **extra: Any) -> dict:
    return {"config": cfg, "height": height, "data": {"values": values}, "layer": layers, "usermeta": _rev(values), **extra}


def _label_room(cfg: dict) -> dict:
    """Pixels right of the plot for end labels: the scale headroom is a share of the data range, so large print text (15 px)
    on a narrow card would otherwise be cut at the chart edge."""
    return {"right": 2 * _size(cfg)}


def trend_spec(monthly: pd.DataFrame, grain: str, cfg: dict, *, through: pd.Timestamp | None = None) -> dict:
    """R1: monthly columns with a 12-month rolling mean, or yearly columns; `through` is the last month the snapshot can cover."""
    if grain == "Tahunan":
        return _yearly_spec(monthly, cfg, through)
    rows = [{"bulan": _plain(row.bulan), "n": int(row.n), "rata12": _plain(row.rata12), "parsial": bool(row.parsial),
             "label_bulan": fmt_month(row.bulan) + (" (belum lengkap)" if row.parsial else ""), "label_rata": fmt_dec(row.rata12, 1),
             "label_rating": fmt_dec(row.rating, 2)} for row in monthly.itertuples(index=False)]
    first, last = (pd.Timestamp(rows[0]["bulan"]), pd.Timestamp(rows[-1]["bulan"])) if rows else (None, None)
    mid_years = [{"year": year, "month": 7, "date": 1, "utc": True} for year in range(first.year, last.year + 1)
                 if first <= pd.Timestamp(year, 7, 1) <= last] if rows else []
    x = {"field": "bulan", "type": "temporal", "timeUnit": "utcyearmonth", "title": None,
         "axis": {"format": "%Y", "values": mid_years, "ticks": False, "grid": False, "labelColor": INK}}
    tooltip = [_tip("label_bulan", "Bulan"), {"field": "n", "type": "quantitative", "title": "Ulasan"},
               _tip("label_rata", "Rata-rata 12 bulan"), _tip("label_rating", "Rating rata-rata")]
    layers: list[dict] = []
    if rows:
        start, stop = max(pd.Timestamp(PANDEMIC[0]), first), min(pd.Timestamp(PANDEMIC[1]), last + pd.DateOffset(months=1))
        if start < stop:
            band = {"values": [{"mulai": start.strftime("%Y-%m-%d"), "selesai": stop.strftime("%Y-%m-%d"), "teks": "Pandemi (konteks)"}]}
            layers += [{"data": band, "mark": {"type": "rect", "color": TRACK, "opacity": 0.6, **NO_TIP},
                        "encoding": {"x": {"field": "mulai", "type": "temporal"}, "x2": {"field": "selesai"}}},
                       {"data": band, "mark": {"type": "text", "align": "left", "baseline": "top", "dx": 5, "dy": 4, "color": MUTED,
                                               "fontSize": _size(cfg) - 1, **NO_TIP},
                        "encoding": {"x": {"field": "mulai", "type": "temporal"}, "y": {"value": 0}, "text": {"field": "teks"}}}]
    layers += [{"mark": {"type": "bar", "color": NILA, "cornerRadiusEnd": 2, "binSpacing": 1},
                "encoding": {"x": x, "y": {"field": "n", "type": "quantitative", "title": "Ulasan", "axis": {"tickMinStep": 1, "tickCount": 5}},
                             "opacity": {"condition": {"test": "datum.parsial", "value": 0.35}, "value": 1}, "tooltip": tooltip}},
               {"mark": {"type": "line", "color": WHITE, "strokeWidth": 4, "interpolate": "monotone", **NO_TIP},
                "encoding": {"x": x, "y": {"field": "rata12", "type": "quantitative"}}},
               {"mark": {"type": "line", "color": INK, "strokeWidth": 2, "interpolate": "monotone"},
                "encoding": {"x": x, "y": {"field": "rata12", "type": "quantitative"}, "tooltip": tooltip}}]
    return _spec(cfg, _step(cfg, 300), rows, layers)


def _yearly_spec(monthly: pd.DataFrame, cfg: dict, through: pd.Timestamp | None) -> dict:
    rows = []
    if len(monthly):
        years = monthly.groupby("tahun", sort=True)["n"].sum()
        for year, n in years.items():
            short = through is not None and int(year) == through.year and through.month < 12
            span = f"Jan–{MONTHS[through.month - 1]}" if short else ""
            rows.append({"tahun": str(year), "n": int(n), "parsial": bool(short), "label": fmt_int(n) + (f" · {span}" if span else ""),
                         "rentang": span or "Jan–Des"})
    tooltip = [_tip("tahun", "Tahun"), {"field": "n", "type": "quantitative", "title": "Ulasan"}, _tip("rentang", "Bulan tercakup")]
    x = {"field": "tahun", "type": "ordinal", "title": None, "axis": {"labelAngle": 0}}
    layers = [{"mark": {"type": "bar", "color": NILA, "size": 28},
               "encoding": {"x": x, "y": {"field": "n", "type": "quantitative", "title": "Ulasan", "axis": {"tickMinStep": 1}},
                            "opacity": {"condition": {"test": "datum.parsial", "value": 0.35}, "value": 1}, "tooltip": tooltip}},
              {"mark": {"type": "text", "baseline": "bottom", "dy": -5}, "encoding": {"x": x, "y": {"field": "n", "type": "quantitative"},
                                                                                      "text": {"field": "label"}, "tooltip": tooltip}}]
    return _spec(cfg, _step(cfg, 300), rows, layers)


def venue_share_spec(rows: pd.DataFrame, overall: float, cfg: dict) -> dict:
    """R3: complaint share per venue with Wilson 95% intervals and the overall share as a reference rule."""
    values = _records(rows, ["venue", "cohort", "n", "k", "share", "lo", "hi", "label", "ci"])
    for value in values:
        value["porsi"] = fmt_pct(value["share"])
    order = [value["venue"] for value in values]
    top = max([value["hi"] or 0 for value in values] + [overall if overall == overall else 0, 0.05])
    x = {"field": "share", "type": "quantitative", "title": "Porsi ulasan berkeluhan",
         "scale": {"domain": [0, top * 1.5], "nice": False}, "axis": {"format": ".0%", "tickCount": 5}}
    y = {"field": "venue", "type": "nominal", "sort": order, "title": None, "scale": {"paddingOuter": 0.6},
         "axis": {"labelLimit": 220, "labelColor": INK}}
    tooltip = [_tip("venue", "Venue"), {"field": "n", "type": "quantitative", "title": "Ulasan berlabel"},
               {"field": "k", "type": "quantitative", "title": "Berkeluhan"}, _tip("porsi", "Porsi"), _tip("ci", "IK95%")]
    layers: list[dict] = []
    mean = {"values": [{"overall": float(overall), "teks": f"rata-rata {fmt_pct(overall)}"}]} if overall == overall and len(values) > 1 else None
    if mean:
        layers.append({"data": mean, "mark": {"type": "rule", "color": MUTED, "strokeWidth": 1, **NO_TIP},
                       "encoding": {"x": {**x, "field": "overall"}}})
    layers += [{"mark": {"type": "bar", "color": NILA, "size": 18},
                "encoding": {"x": x, "y": y, "opacity": {"condition": {"test": "datum.cohort == 'legacy81'", "value": 0.55}, "value": 1},
                             "tooltip": tooltip}},
               {"mark": {"type": "rule", "color": INK, "strokeWidth": 1.5},
                "encoding": {"x": {**x, "field": "lo"}, "x2": {"field": "hi"}, "y": y, "tooltip": tooltip}},
               *_halo({"mark": {"type": "text", "align": "left", "dx": 6, "color": INK},
                       "encoding": {"x": {**x, "field": "hi"}, "y": y, "text": {"field": "label"}, "tooltip": tooltip}})]
    if mean:
        layers += _halo({"data": mean, "mark": {"type": "text", "align": "left", "baseline": "bottom", "dx": 4, "dy": -3, "color": MUTED,
                                                "fontSize": _size(cfg) - 1, **NO_TIP},
                         "encoding": {"x": {**x, "field": "overall"}, "y": {"value": "height"}, "text": {"field": "teks"}}})
    return _spec(cfg, len(values) * _step(cfg, 36) + _step(cfg, 64), values, layers)


def rating_text_spec(cells: pd.DataFrame, cfg: dict) -> dict:
    """R4: star rating rows against text sentiment columns; outlined cells are complaints hidden behind 4-5 stars."""
    values = _records(cells, ["rating", "sentimen", "n", "row_total", "row_share", "hidden", "label"])
    for value in values:
        value["kolom"] = f"{SENTIMENTS.get(value['sentimen'], '·')} {value['sentimen']}"
        value["baris"] = f"★{value['rating']}"
        value["porsi"] = fmt_share(value["row_share"]) if value["row_total"] else "—"
    columns = [f"{glyph} {name}" for name, glyph in SENTIMENTS.items()]
    x = {"field": "kolom", "type": "nominal", "sort": columns, "title": None,
         "axis": {"orient": "top", "labelAngle": 0, "labelColor": INK, "labelPadding": 8}}
    y = {"field": "rating", "type": "ordinal", "sort": "descending", "title": None,
         "axis": {"labelExpr": "'★' + datum.label", "labelColor": INK, "labelPadding": 8}}
    tooltip = [_tip("baris", "Rating"), _tip("sentimen", "Sentimen teks"), {"field": "n", "type": "quantitative", "title": "Ulasan"},
               _tip("porsi", "Porsi dalam baris")]
    layers = [{"mark": {"type": "rect", "stroke": WHITE, "strokeWidth": 3, "cornerRadius": 4},
               "encoding": {"x": x, "y": y, "tooltip": tooltip,
                            "color": {"field": "row_share", "type": "quantitative", "legend": None,
                                      "scale": {"domain": [0, 1], "range": list(SEQ_COOL), "interpolate": "hcl"}}}},
              {"transform": [{"filter": "datum.hidden && datum.n > 0"}],
               "mark": {"type": "rect", "filled": False, "stroke": INK, "strokeWidth": 2, "cornerRadius": 4},
               "encoding": {"x": x, "y": y, "tooltip": tooltip}},
              *_cell_labels({"mark": {"type": "text"}, "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}},
                            "row_share", SEQ_COOL, muted_when_zero=True)]
    return _spec(cfg, 5 * _step(cfg, 46) + _step(cfg, 30), values, layers, **TOP_GAP)


def aspect_heatmap_spec(matrix: pd.DataFrame, metric: str, cfg: dict) -> dict:
    """N1: aspect x venue mention rate (cool ramp) or complaint rate (warm ramp); a click selects the evidence cell."""
    values = _records(matrix, ["aspek_kode", "aspek", "venue_kode", "venue", "venue_urut", "n", "mentions", "complaints", "nilai", "small",
                               "polarity_n", "missing_polarity", "denominator", "missing", "terpilih", "label"])
    for value in values:
        value["kolom"] = venue_short(value["venue_kode"])
        value["nilai_label"] = fmt_pct(value["nilai"])
        value["label"] = value["label"] or "—"
    venues = [value["kolom"] for value in sorted(values, key=lambda item: item["venue_urut"])]
    columns = list(dict.fromkeys(venues))
    rows_order = [ASPECT_NAMES[key] for key in ASPECT_NAMES]
    top = max([value["nilai"] or 0 for value in values] + [0.0])
    domain_max = 1.0 if metric == "Penyebutan" else max(0.1, math.ceil(top * 20) / 20)
    ramp = SEQ_COOL if metric == "Penyebutan" else SEQ_WARM
    wrapped = any(len(column) > 10 and " " in column for column in columns)  # long names take two lines above the grid
    x = {"field": "kolom", "type": "nominal", "sort": columns, "title": None,
         "axis": {"orient": "top", "labelAngle": 0, "labelColor": INK, "labelPadding": 8 + (_size(cfg) + 2 if wrapped else 0),
                  "labelLineHeight": _size(cfg) + 2, "labelExpr": "length(datum.label) > 10 ? split(datum.label, ' ') : datum.label"}}
    y = {"field": "aspek", "type": "nominal", "sort": rows_order, "title": None, "axis": {"labelColor": INK, "labelLimit": 200, "labelPadding": 8}}
    tooltip = [_tip("venue", "Venue"), _tip("aspek", "Aspek"), {"field": "n", "type": "quantitative", "title": "Ulasan venue"},
               {"field": "mentions", "type": "quantitative", "title": "Penyebutan"},
               {"field": "complaints", "type": "quantitative", "title": "Penyebutan berkeluhan"},
               {"field": "polarity_n", "type": "quantitative", "title": "Penyebutan dengan polaritas tersedia (m)"},
               {"field": "missing_polarity", "type": "quantitative", "title": "Polaritas hilang"},
               {"field": "denominator", "type": "quantitative", "title": "Denominator sel"},
               _tip("nilai_label", "Porsi penyebutan" if metric == "Penyebutan" else "Porsi berkeluhan")]
    layers = [{"params": [{"name": "pilih_sel", "select": {"type": "point", "fields": ["aspek_kode", "venue_kode"], "toggle": False}}],
               "mark": {"type": "rect", "cornerRadius": 4, "cursor": "pointer"},
               "encoding": {"x": x, "y": y, "tooltip": tooltip,
                            "color": {"condition": {"test": "datum.missing", "value": PAPER}, "field": "nilai", "type": "quantitative", "legend": None,
                                      "scale": {"domain": [0, domain_max], "clamp": True, "interpolate": "hcl",
                                                "range": list(ramp)}},
                            "stroke": {"condition": {"test": "datum.missing", "value": MUTED}, "value": WHITE}, "strokeWidth": {"value": 3},
                            "strokeDash": {"condition": {"test": "datum.missing", "value": [3, 3]}, "value": [1, 0]}}},
              # One source for the frame: the server's selected cell, which also picks the quotes (a click reruns the page).
              {"mark": {"type": "rect", "filled": False, "cornerRadius": 4, "stroke": INK, "strokeWidth": 2.5},
               "encoding": {"x": x, "y": y, "tooltip": tooltip, "strokeOpacity": {"condition": {"test": "datum.terpilih", "value": 1}, "value": 0}}},
              *_cell_labels({"mark": {"type": "text", "lineBreak": "\n", "lineHeight": _size(cfg) + 3},
                             "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}}, "nilai", ramp, domain_max)]
    return _spec(cfg, len(rows_order) * _step(cfg, 76) + _step(cfg, 30) + (_size(cfg) + 2 if wrapped else 0), values, layers, **TOP_GAP)


def terms_spec(terms: pd.DataFrame, mode: str, cfg: dict) -> dict:
    """N3: weighted log-odds terms; diverging bars (complaints vs the rest) or one panel per venue."""
    if mode != "Keluhan vs lainnya":
        return _venue_terms_spec(terms, cfg)
    values = _records(terms, ["term", "z", "df_a", "df_b", "n_a", "n_b", "docs_a", "docs_b", "sisi", "label"])
    for value in values:
        value["z_label"] = fmt_dec(value["z"], 2)
    order = [value["term"] for value in values]
    zs = [value["z"] for value in values] or [0.0]
    low, high = min(min(zs), 0.0), max(max(zs), 0.0)
    span = max(high - low, 1.0)
    x = {"field": "z", "type": "quantitative", "title": "Skor log-odds eksploratif (← lainnya | berkeluhan →)",
         "scale": {"domain": [low - (span * 0.32 if low < 0 else 0), high + span * 0.2], "nice": False}, "axis": {"format": ".1f", "tickCount": 8}}
    y = {"field": "term", "type": "nominal", "sort": order, "title": None, "axis": {"labelColor": INK, "labelFontWeight": 600}}
    tooltip = [_tip("term", "Kata"), _tip("sisi", "Lebih khas pada"), _tip("z_label", "z"),
               {"field": "df_a", "type": "quantitative", "title": "Ulasan berkeluhan memuat kata"}, {"field": "df_b", "type": "quantitative", "title": "Ulasan lain memuat kata"},
               {"field": "n_a", "type": "quantitative", "title": "Massa frekuensi dokumen kelompok A (n_a)"},
               {"field": "n_b", "type": "quantitative", "title": "Massa frekuensi dokumen kelompok B (n_b)"}]
    layers = [{"params": [{"name": "pilih_kata", "select": {"type": "point", "fields": ["term"], "toggle": False}}],
               "mark": {"type": "bar", "size": 14, "cursor": "pointer"},
               "encoding": {"x": x, "y": y, "tooltip": tooltip,
                            "color": {"field": "sisi", "type": "nominal", "legend": None,
                                      "scale": {"domain": ["ulasan berkeluhan", "ulasan lain"], "range": [SOGA, NILA]}},
                            "opacity": {"value": 1}}},
              {"data": {"values": [{"nol": 0}]}, "mark": {"type": "rule", "color": MUTED, "strokeWidth": 1, **NO_TIP},
               "encoding": {"x": {"field": "nol", "type": "quantitative"}}},
              {"transform": [{"filter": "datum.z >= 0"}], "mark": {"type": "text", "align": "left", "dx": 5, "color": MUTED},
               "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}},
              {"transform": [{"filter": "datum.z < 0"}], "mark": {"type": "text", "align": "right", "dx": -5, "color": MUTED},
               "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}}]
    return _spec(cfg, max(len(values), 1) * _step(cfg, 20) + _step(cfg, 74), values, layers, padding=_label_room(cfg))


def _venue_terms_spec(terms: pd.DataFrame, cfg: dict) -> dict:
    values = _records(terms, ["venue_key", "venue", "n_docs", "term", "z", "df_a", "df_b", "n_a", "n_b", "docs_a", "docs_b", "label"])
    for value in values:
        value["panel"] = f"{value['venue']} · {fmt_int(value['n_docs'])} ulasan"
        value["z_label"] = fmt_dec(value["z"], 2)
    panels = list(dict.fromkeys(value["panel"] for value in values))
    top = max([value["z"] for value in values] + [1.0])
    x = {"field": "z", "type": "quantitative", "title": "Skor log-odds eksploratif", "scale": {"domain": [0, top * 1.45], "nice": False},
         "axis": {"format": ".1f", "tickCount": 4}}
    y = {"field": "term", "type": "nominal", "sort": {"field": "z", "order": "descending"}, "title": None,
         "axis": {"labelColor": INK, "labelFontWeight": 600}}
    tooltip = [_tip("venue", "Venue"), _tip("term", "Kata"), _tip("z_label", "z"), {"field": "df_a", "type": "quantitative", "title": "Ulasan venue ini"},
               {"field": "df_b", "type": "quantitative", "title": "Ulasan venue lain"}]
    inner = {"width": 150 if _size(cfg) <= 12 else 140, "height": {"step": _step(cfg, 20)},
             "layer": [{"params": [{"name": "pilih_kata", "select": {"type": "point", "fields": ["term"], "toggle": False}}],
                        "mark": {"type": "bar", "size": 13, "color": NILA, "cursor": "pointer"},
                        "encoding": {"x": x, "y": y, "tooltip": tooltip,
                                     "opacity": {"value": 1}}},
                       {"mark": {"type": "text", "align": "left", "dx": 5, "color": MUTED},
                        "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}}]}
    return {"config": cfg, **TOP_GAP, "data": {"values": values}, "usermeta": _rev(values), "columns": 2, "spacing": {"row": 22, "column": 30},
            "facet": {"field": "panel", "type": "nominal", "sort": panels, "title": None}, "spec": inner,
            "resolve": {"scale": {"y": "independent"}}}


def length_spec(bins: pd.DataFrame, cfg: dict) -> dict:
    """N4: complaint share by review length with Wilson intervals; the thin rule marks the overall share, named in the card
    caption (an in-plot label would cross the interval rules of filtered views)."""
    values = _records(bins, ["urut", "bin", "n", "k", "share", "lo", "hi", "label", "axis"])
    total = sum(value["n"] for value in values)
    overall = sum(value["k"] for value in values) / total if total else math.nan
    for value in values:
        value["ci"] = f"{fmt_pct(value['lo'])}–{fmt_pct(value['hi'])}" if value["lo"] is not None else "—"
        value["kata"] = f"{value['bin']} kata"
    order = [value["axis"] for value in values]
    top = max([value["hi"] or 0 for value in values] + [0.05])
    x = {"field": "axis", "type": "ordinal", "sort": order, "title": "Panjang ulasan (kata)",
         "axis": {"labelAngle": 0, "labelExpr": "split(datum.label, '\\n')", "labelColor": INK, "labelLineHeight": _size(cfg) + 3}}
    y = {"field": "share", "type": "quantitative", "title": "Porsi ulasan berkeluhan",
         "scale": {"domain": [0, top * 1.18], "nice": False}, "axis": {"format": ".0%", "tickCount": 5}}
    tooltip = [_tip("kata", "Panjang"), {"field": "n", "type": "quantitative", "title": "Ulasan"},
               {"field": "k", "type": "quantitative", "title": "Berkeluhan"}, _tip("label", "Porsi"), _tip("ci", "IK95%")]
    layers = [{"mark": {"type": "rule", "color": INK, "strokeWidth": 1.5},
               "encoding": {"x": x, "y": {**y, "field": "lo"}, "y2": {"field": "hi"}, "tooltip": tooltip}},
              {"mark": {"type": "point", "size": 110, "filled": True, "color": NILA, "stroke": WHITE, "strokeWidth": 2, "opacity": 1},
               "encoding": {"x": x, "y": y, "tooltip": tooltip}},
              *_halo({"mark": {"type": "text", "align": "left", "baseline": "bottom", "dx": 7, "dy": -3, "color": INK, "fontWeight": 600},
                      "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}})]
    if overall == overall:
        layers.insert(0, {"data": {"values": [{"overall": float(overall)}]}, "mark": {"type": "rule", "color": MUTED, "strokeWidth": 1, **NO_TIP},
                          "encoding": {"y": {**y, "field": "overall"}}})
    return _spec(cfg, _step(cfg, 400), values, layers, padding=_label_room(cfg))


def _row_bars(values: list[dict], cfg: dict, *, track: str, value: str, domain_max: float, left: str, right: str) -> dict:
    """Labelled bar rows: text above each thin bar, so long labels never collide with bar ends."""
    row, bar = _step(cfg, 46), _step(cfg, 12)
    y = {"field": "urut", "type": "ordinal", "axis": None, "scale": {"paddingInner": 0, "paddingOuter": 0}}
    x = {"type": "quantitative", "axis": None, "scale": {"domain": [0, domain_max], "nice": False, "zero": True}}
    lift = round(bar * 0.75)
    layers = [{"mark": {"type": "bar", "color": TRACK, "size": bar, "yOffset": lift, "cornerRadiusEnd": 3},
               "encoding": {"x": {**x, "field": track}, "y": y, "tooltip": {"field": "tip", "type": "nominal"}}},
              {"mark": {"type": "bar", "size": bar, "yOffset": lift, "cornerRadiusEnd": 3},
               "encoding": {"x": {**x, "field": value}, "y": y, "tooltip": {"field": "tip", "type": "nominal"},
                            "color": {"condition": {"test": "datum.perhatian", "value": SOGA}, "value": NILA}}},
              {"mark": {"type": "text", "align": "left", "baseline": "bottom", "dy": -2, "color": INK, "fontWeight": 600},
               "encoding": {"x": {"value": 0}, "y": y, "text": {"field": left}, "tooltip": {"field": "tip", "type": "nominal"}}},
              {"mark": {"type": "text", "align": "right", "baseline": "bottom", "dy": -2, "color": MUTED},
               "encoding": {"x": {"value": "width"}, "y": y, "text": {"field": right}, "tooltip": {"field": "tip", "type": "nominal"}}}]
    return _spec(cfg, max(len(values), 1) * row, values, layers, padding={"left": 2, "right": 2, "top": 8, "bottom": 2})


def funnel_spec(funnel: pd.DataFrame, cfg: dict) -> dict:
    """O1: data funnel from collection to the training split; the pale track is the previous stage."""
    values = _records(funnel, ["urut", "tahap", "n", "n_prev", "turun", "catatan", "label"])
    for value in values:
        split = value["tahap"].startswith("Split")
        value["kiri"] = f"{value['tahap']} · {fmt_int(value['n'])}"
        value["kanan"] = value["label"] if split else value["catatan"] or ""
        value["tip"] = f"{value['tahap']}: {fmt_int(value['n'])}" + (f" ({value['catatan']})" if value["catatan"] else "")
        value["perhatian"] = False
    top = max([value["n_prev"] for value in values] + [1])
    return _row_bars(values, cfg, track="n_prev", value="n", domain_max=top, left="kiri", right="kanan")


def coverage_spec(rows: pd.DataFrame, cfg: dict) -> dict:
    """O2: bullet bars of collected / platform-reported reviews per scraping target."""
    values = _records(rows.reset_index(drop=True).reset_index(names="urut"), ["urut", "target", "platform", "collected", "reported", "coverage",
                                                                               "status", "reason", "label"])
    for value in values:
        value["penuh"] = 1.0
        value["nilai"] = value["coverage"] or 0.0
        value["perhatian"] = value["status"] == "parsial"
        value["tip"] = f"{value['target']} · {value['label']} · {value['status']} ({value['reason']})"
    return _row_bars(values, cfg, track="penuh", value="nilai", domain_max=1.0, left="target", right="label")


def quality_spec(matrix: pd.DataFrame, cfg: dict) -> dict:
    """O4: data-quality checks per venue plus a total column; darker = larger share of that venue's rows."""
    values = _records(matrix, ["cek_kode", "cek", "urut_cek", "kolom_kode", "kolom", "urut_kolom", "n", "total", "pct", "label"])
    for value in values:
        value["porsi"] = fmt_pct(value["pct"])
        value["jumlah"] = f"{fmt_int(value['n'])} dari {fmt_int(value['total'])}"
    columns = list(dict.fromkeys(value["kolom"] for value in sorted(values, key=lambda item: item["urut_kolom"])))
    checks = list(dict.fromkeys(value["cek"] for value in sorted(values, key=lambda item: item["urut_cek"])))
    x = {"field": "kolom", "type": "nominal", "sort": columns, "title": None,
         "axis": {"orient": "top", "labelAngle": 0, "labelColor": INK, "labelPadding": 8, "labelLimit": 160}}
    y = {"field": "cek", "type": "nominal", "sort": checks, "title": None, "axis": {"labelColor": INK, "labelLimit": 280, "labelPadding": 8}}
    tooltip = [_tip("cek", "Pemeriksaan"), _tip("kolom", "Kolom"), _tip("jumlah", "Baris"), _tip("porsi", "Porsi")]
    layers = [{"mark": {"type": "rect", "stroke": WHITE, "strokeWidth": 3, "cornerRadius": 4},
               "encoding": {"x": x, "y": y, "tooltip": tooltip,
                            "color": {"condition": {"test": "datum.n == 0", "value": PAPER}, "field": "pct", "type": "quantitative", "legend": None,
                                      "scale": {"domain": [0, 1], "range": list(SEQ_WARM), "interpolate": "hcl"}}}},
              *(layer for test, weight in (("!=", 400), ("==", 700))
                for layer in _cell_labels({"transform": [{"filter": f"datum.kolom_kode {test} 'total'"}], "mark": {"type": "text", "fontWeight": weight},
                                           "encoding": {"x": x, "y": y, "text": {"field": "label"}, "tooltip": tooltip}},
                                          "pct", SEQ_WARM, muted_when_zero=True))]
    return _spec(cfg, len(checks) * _step(cfg, 32) + _step(cfg, 30), values, layers, **TOP_GAP)


def scroll_spec(curves: pd.DataFrame, note: str, cfg: dict) -> dict:
    """O5: cumulative unique reviews per scroll round of each Google Maps run; the note states the duplicate burden."""
    values = _records(curves, ["run", "putaran", "unik", "duplikat", "akhir"])
    runs = list(dict.fromkeys(value["run"] for value in values))
    palette = [SOGA, NILA, KARAT, INK][:max(len(runs), 1)]
    x = {"field": "putaran", "type": "quantitative", "title": "Putaran gulir", "axis": {"tickMinStep": 1, "tickCount": 12}}
    y = {"field": "unik", "type": "quantitative", "title": "Ulasan unik terkumpul", "axis": {"tickMinStep": 1, "tickCount": 5}}
    color = {"field": "run", "type": "nominal", "scale": {"domain": runs, "range": palette}, "title": None}
    tooltip = [_tip("run", "Run"), {"field": "putaran", "type": "quantitative", "title": "Putaran"},
               {"field": "unik", "type": "quantitative", "title": "Unik"}, {"field": "duplikat", "type": "quantitative", "title": "Duplikat kumulatif"}]
    layers = [{"mark": {"type": "line"}, "encoding": {"x": x, "y": y, "color": color, "tooltip": tooltip}},
              {"transform": [{"filter": "datum.akhir"}], "mark": {"type": "point", "size": 90},
               "encoding": {"x": x, "y": y, "color": color, "tooltip": tooltip}},
              {"transform": [{"filter": "datum.akhir"}], "mark": {"type": "text", "align": "left", "dx": 8, "color": INK},
               "encoding": {"x": x, "y": y, "text": {"field": "unik", "type": "quantitative", "format": ",d"}, "tooltip": tooltip}},
              {"data": {"values": [{"teks": note}]},
               "mark": {"type": "text", "align": "left", "baseline": "top", "dx": 4, "dy": 4, "color": MUTED, **NO_TIP},
               "encoding": {"x": {"value": 0}, "y": {"value": 0}, "text": {"field": "teks"}}}]
    return _spec(cfg, _step(cfg, 250), values, layers, **TOP_GAP)
