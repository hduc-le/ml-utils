import math as _math

import numpy as _np
import pandas as _pd
import plotly.graph_objects as go
from plotly import optional_imports

scipy_stats = optional_imports.get_module("scipy.stats")
# pybloqs
from pybloqs.html import append_to as _append_to
from pybloqs.block.layout import CompositeBlockMixin as _CompositeBlockMixin
from pybloqs.block.layout import BaseBlock as _BaseBlock
from pybloqs.block.layout import Cfg as _Cfg
from pybloqs import Block
import pybloqs.block.table_formatters as tf

# dash
from dash import html, dcc
import dash_dangerously_set_inner_html as dds
from pandas.api.types import is_numeric_dtype

# from plotly.offline import init_notebook_mode as _init_notebook_mode

# # Fix bug not show chart
# _init_notebook_mode(connected=True)

_table_width = "320px"
_FREQ_COL_NAME = "frequency (%)"
_COUNT_COL_NAME = "count"
_DEFAULT_NBINS_HISTOGRAM = 50
_MAIN_COLOR = "#1F77B4"


class CustomBaseStack(object):
    objs = []
    layout = ""
    default_grid_cell_styles = {"margin-right": "50px", "margin-bottom": "30px"}
    default_grid_row_styles = {"display": "flex", "align-items": "center"}
    default_grid_column_styles = {
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "center",
    }

    def __init__(
        self, objs, grid_cell_style=None, grid_row_style=None, grid_column_style=None
    ):
        if type(grid_cell_style) == dict:
            self.default_grid_cell_styles.update(grid_cell_style)
        if type(grid_row_style) == dict:
            self.default_grid_row_styles.update(grid_row_style)
        if type(grid_column_style) == dict:
            self.default_grid_column_styles.update(grid_column_style)
        grid_cell_style_str = ";".join(
            [f"{key}:{value}" for key, value in self.default_grid_cell_styles.items()]
        )
        grid_row_style_str = ";".join(
            [f"{key}:{value}" for key, value in self.default_grid_row_styles.items()]
        )
        grid_column_style_str = ";".join(
            [f"{key}:{value}" for key, value in self.default_grid_column_styles.items()]
        )
        self.objs = objs
        self.layout = self._generate_html_string(
            self.objs,
            grid_cell_style_str=grid_cell_style_str,
            grid_row_style_str=grid_row_style_str,
            grid_column_style_str=grid_column_style_str,
        )

    def show(self):
        from IPython.display import display_html

        display_html(self.layout, raw=True)

    def get_html(self):
        return self.layout

    def _generate_html_string(
        self,
        objs=None,
        grid_cell_style_str=None,
        grid_row_style_str=None,
        grid_column_style_str=None,
    ):
        pass

    def _get_plotly_html_string(self, fig):
        from plotly.io import renderers
        from plotly.io._utils import validate_coerce_fig_to_dict

        fig_dict = validate_coerce_fig_to_dict(fig, validate=True)
        fig_html = renderers._build_mime_bundle(fig_dict)["text/html"]
        return fig_html

    def _get_obj_html_string(self, obj):
        from plotly.graph_objects import Figure
        from pandas import DataFrame
        from pandas.io.formats.style import Styler

        if type(obj) == str:
            obj_html = obj
        elif isinstance(obj, CustomBaseStack):
            obj_html = obj.get_html()
        elif isinstance(obj, Figure):
            obj_html = self._get_plotly_html_string(obj)
        elif isinstance(obj, DataFrame) or isinstance(obj, Styler):
            obj_html = obj.to_html()
        else:
            obj_html = ""
            print(
                f"""Type {type(obj)} is not supported.
                    Only object of the following classes is supported: str, plotly.graph_objects.Figure,
                    pandas.DataFrame, pandas.io.formats.style.Styler, CustomHStack, CustomVStack"""
            )
        return obj_html


class CustomHStack(CustomBaseStack):
    def __init__(self, objs, grid_cell_style=None, grid_row_style=None):
        super(CustomHStack, self).__init__(objs, grid_cell_style, grid_row_style, None)
        return

    def _generate_html_string(
        self,
        objs=None,
        grid_cell_style_str=None,
        grid_row_style_str=None,
        grid_column_style_str=None,
    ):
        objs = objs or []
        grid_cell_style_str = grid_cell_style_str or ""
        grid_row_style_str = grid_row_style_str or ""

        serialized_html = ""
        for obj in objs:
            obj_html = self._get_obj_html_string(obj)
            serialized_html += f'<div class="custom_cell" style="{grid_cell_style_str}">{obj_html}</div>'
        layout = f"""
        <div class="custom_container">
            <div class="custom_row" style="{grid_row_style_str}">
            {serialized_html}
            </div>
        </div>"""
        return layout


class CustomVStack(CustomBaseStack):
    def __init__(self, objs, grid_cell_style=None, grid_column_style=None):
        super(CustomVStack, self).__init__(
            objs, grid_cell_style, None, grid_column_style
        )
        return

    def _generate_html_string(
        self,
        objs=None,
        grid_cell_style_str=None,
        grid_row_style_str=None,
        grid_column_style_str=None,
    ):
        objs = objs or []
        grid_cell_style_str = grid_cell_style_str or ""
        grid_column_style_str = grid_column_style_str or ""

        serialized_html = ""
        for obj in objs:
            obj_html = self._get_obj_html_string(obj)
            serialized_html += f'<div class="custom_cell" style="{grid_cell_style_str}">{obj_html}</div>'

        layout = f"""
        <div class="custom_container">
            <div class="custom_column" style="{grid_column_style_str}">
            {serialized_html}
            </div>
        </div>"""
        return layout


class CustomGrid(_CompositeBlockMixin, _BaseBlock):
    def __init__(
        self,
        contents,
        cols=1,
        cascade_cfg=True,
        grid_row_style=None,
        grid_cell_style=None,
        grid_style=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._contents = self._blockify_contents(
            contents, kwargs, self._settings.title_level
        )
        self._cols = cols
        self._cascade_cfg = cascade_cfg
        self._grid_style = grid_style
        self._grid_row_style = grid_row_style
        self._grid_cell_style = grid_cell_style

    def _write_contents(self, container, actual_cfg, *args, **kwargs):
        content_count = len(self._contents)

        # Skip layout if there is no content
        if content_count > 0:
            row_count = int(_math.ceil(content_count / float(self._cols)))

            for row_i in range(row_count):
                row_el = _append_to(container, "div")
                row_el["class"] = ["pybloqs-grid-row"]
                if self._grid_row_style:
                    row_el["style"] = self._grid_row_style

                written_row_item_count = row_i * self._cols
                for col_i in range(self._cols):
                    item_count = written_row_item_count + col_i
                    if item_count >= content_count:
                        break

                    cell_el = _append_to(row_el, "div")
                    cell_el = _append_to(row_el, "div", style=self._grid_cell_style)
                    cell_el["class"] = ["pybloqs-grid-cell"]
                    self._contents[item_count]._write_block(
                        cell_el,
                        actual_cfg if self._cascade_cfg else _Cfg(),
                        *args,
                        **kwargs,
                    )

            _append_to(container, "div", style="clear:both")


# Add default grid_row_style and grid_cell_style
class CustomHStack(CustomGrid):
    def __init__(
        self,
        contents,
        cascade_cfg=True,
        grid_row_style="display:flex;align-items:center;",
        grid_cell_style="margin-right:50px;",
        **kwargs,
    ):
        super().__init__(
            contents,
            cascade_cfg=cascade_cfg,
            grid_row_style=grid_row_style,
            grid_cell_style=grid_cell_style,
            **kwargs,
        )
        self._cols = len(self._contents)


# histogram calculate bins
def increment_numeric(x, delta):
    if not delta:
        return x
    scale = 1 / abs(delta)
    if scale > 1:
        newx = (scale * x + scale * delta) / scale
    else:
        newx = x + delta
    len_newx = len(str(newx))
    if len_newx > 16:  # likely a rounding error
        len_x = len(str(x))
        len_delta = len(str(delta))
        if len_newx >= len_x + len_delta:
            newx = float("{:.12f}".format(newx))
    return newx


def make_kde(hist_data, name="kernel density", n_points=100):
    start = min(hist_data) * 1.0
    end = max(hist_data) * 1.0
    curve_x = [start + x * (end - start) / n_points for x in range(n_points)]
    curve_y = scipy_stats.gaussian_kde(hist_data)(curve_x)
    curve = go.Scatter(x=curve_x, y=curve_y, mode="lines", name=name)
    return curve


def pearson(num1, num2):
    mean1 = num1.mean()
    mean2 = num2.mean()
    if _pd.isna(mean1) or _pd.isna(mean2):
        return 0
    return scipy_stats.pearsonr(num1.fillna(mean1), num2.fillna(mean2)).statistic


def describe_percentiles(df, col, percentiles=None, fmt=".2f"):
    """
    Generate descriptive statistics of a numeric column."""
    if not percentiles:
        percentiles = [i * 0.1 for i in range(1, 10)] + [0.01, 0.05, 0.95, 0.99]

    percentile_df = (
        _pd.to_numeric(df[col])
        .describe(percentiles=percentiles)
        .to_frame()
        .rename(columns={col: "value"})
    )

    # Add Null count, Skewness & Kurtosis
    percentile_df.loc["nan"] = df[col].isnull().sum() / df.shape[0]
    percentile_df.loc["skew"] = df[col].skew()
    percentile_df.loc["kurt"] = df[col].kurtosis()
    percentile_df = percentile_df.loc[
        ["count", "mean", "std", "min"]
        + [f"{int(x * 100)}%" for x in sorted(percentiles)]
        + ["max"],
        :,
    ]
    return percentile_df


def series_get_percentiles(series, percentiles):
    result_series = (
        _pd.to_numeric(series)
        .describe(percentiles)
        .loc[[f"{x:.0%}" for x in percentiles]]
    )
    return result_series


def describe_percentiles_shortened(df, col, percentiles=None, fmt=".2f"):
    """
    Generate descriptive statistics of a numeric column."""
    if not percentiles:
        percentiles = [i * 0.1 for i in range(1, 10)] + [0.01, 0.05, 0.95, 0.99]
    quartiles = [0.25, 0.75]

    percentile_df = (
        _pd.to_numeric(df[col])
        .describe(percentiles=percentiles + quartiles)
        .to_frame()
        .rename(columns={col: "value"})
    )

    # Add Null count, Skewness & Kurtosis
    percentile_df.loc["count"] = df.shape[0]
    percentile_df.loc["null"] = df[col].isnull().sum() / df.shape[0]
    percentile_df.loc["skew"] = df[col].skew()
    percentile_df.loc["kurt"] = df[col].kurtosis()
    # percentile_df.loc["distinct"] = df[col].nunique()
    mode_series = df[col].mode()
    if len(mode_series) > 0:
        percentile_df.loc["mode"] = df[col].mode()[0]
    else:
        percentile_df.loc["mode"] = None
    percentile_df.loc["mad"] = (df[col] - df[col].mean()).abs().mean()
    percentile_df.loc["range"] = (
        percentile_df.loc["max", "value"] - percentile_df.loc["min", "value"]
    )
    percentile_df.loc["IQR"] = (
        percentile_df.loc["75%", "value"] - percentile_df.loc["25%", "value"]
    )
    value_1 = percentile_df.loc["1%", "value"]
    value_99 = percentile_df.loc["99%", "value"]
    percentile_df.loc["tr_mean"] = (
        df[
            (_pd.to_numeric(df[col]) >= value_1) & (_pd.to_numeric(df[col]) <= value_99)
        ][col]
        .astype("double")
        .mean()
    )
    percentile_df = percentile_df.loc[
        [
            "count",
            "null",
            "mean",
            "tr_mean",
            "std",
            "skew",
            "kurt",
            "min",
            "mode",
            "mad",
            "range",
            "IQR",
        ]
        + [f"{int(x * 100)}%" for x in sorted(percentiles)]
        + ["max"],
        :,
    ]
    return percentile_df


def render_html(obj):
    if isinstance(obj, _pd.core.frame.DataFrame):
        return obj.to_html()
    elif isinstance(obj, _pd.io.formats.style.Styler):
        return obj.set_table_attributes("style='display:inline-block'").render()
    else:
        return obj


def display_hstack(arr: list, margin=75):
    render_arr = [render_html(x) for x in arr]
    return CustomHStack(
        render_arr,
        grid_row_style="display:flex;align-items:center;",
        grid_cell_style=f"margin-right:{margin}px;",
    )


histfunc_map = {
    "count": _np.ma.count,
    "sum": _np.sum,
    "min": _np.min,
    "max": _np.max,
    "avg": _np.mean,
}


def aggregate_value_by_bin(series, binsize, histfunc="count"):
    if binsize == 0:
        return [0, 1], [0], [0.5], 0
    min_value = series.min()
    max_value = series.max()
    bin_edges = []
    bin_values = []
    bin_centers = []
    if isinstance(histfunc, str):
        if histfunc in histfunc_map:
            histfunc = histfunc_map[histfunc]
        else:
            raise Exception(
                f'Invalid value for histfunc. Supported histfunc values are {",".join(histfunc_map.keys())}'
            )
    # zero is a tick no matter if zero is in the range visilbe o
    start_value = _math.floor(min_value / binsize) * binsize
    bin_edges.append(start_value)
    while bin_edges[-1] < max_value:
        x0 = bin_edges[-1]
        x1 = increment_numeric(x0, binsize)
        center = increment_numeric(x0, binsize / 2)
        value = histfunc(series[(series >= x0) & (series < x1)])
        bin_values.append(value)
        bin_edges.append(x1)
        bin_centers.append(center)
    return bin_edges, bin_values, bin_centers, binsize


def round_bin_size(raw_binsize):
    if raw_binsize == 0:
        return 0
    multiplier = 1
    b = raw_binsize
    if b <= 9:
        while b < 1:
            b *= 10
            multiplier /= 10
    elif b > 9:
        while b > 9:
            b /= 10
            multiplier *= 10
    breakpoints = [1, 2, 5, 10]
    rounded_idx = 0
    rounded_up = breakpoints[rounded_idx]
    while rounded_up < b:
        rounded_idx += 1
        rounded_up = breakpoints[rounded_idx]
    rounded_up = rounded_up * multiplier
    return rounded_up


def calculate_bin(series, nbins=None, binsize=None):
    value_range = series.max() - series.min()
    nbins = nbins or _DEFAULT_NBINS_HISTOGRAM

    if binsize and binsize < 1:
        nbins = 0
    elif binsize:
        nbins = value_range / binsize

    if nbins > series.nunique():
        nbins = series.nunique()

    if nbins < 1 or _pd.isna(nbins):
        raw_binsize = 0
    else:
        raw_binsize = value_range / nbins

    binsize = round_bin_size(raw_binsize)
    return aggregate_value_by_bin(series, binsize)


# precompute box plot
def get_box_trace(series, color=_MAIN_COLOR, orientation="v"):
    axis_name = series.name
    q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
    lowerfence = series.min()
    mean = series.mean()
    upperfence = series.max()
    if orientation == "v":
        box_plot = go.Box(
            y=[axis_name],
            q1=[q1],
            median=[median],
            q3=[q3],
            lowerfence=[lowerfence],
            upperfence=[upperfence],
            mean=[mean],
            marker=dict(color=color),
            boxpoints=False,
        )
    elif orientation == "h":
        box_plot = go.Box(
            x=[axis_name],
            q1=[q1],
            median=[median],
            q3=[q3],
            lowerfence=[lowerfence],
            upperfence=[upperfence],
            mean=[mean],
            marker=dict(color=color),
            boxpoints=False,
        )
    return box_plot
