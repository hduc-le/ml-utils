import math as _math

import numpy as _np
import pandas as _pd
import plotly.graph_objects as go
from plotly import optional_imports
from plotly.subplots import make_subplots

scipy_stats = optional_imports.get_module("scipy.stats")
# pybloqs
from pybloqs.html import append_to as _append_to
from pybloqs.block.layout import CompositeBlockMixin as _CompositeBlockMixin
from pybloqs.block.layout import BaseBlock as _BaseBlock
from pybloqs.block.layout import Cfg as _Cfg

# dash
from . import describe_utils

# from plotly.offline import init_notebook_mode as _init_notebook_mode

# # Fix bug not show chart
# _init_notebook_mode(connected=True)

_table_width = "320px"
_FREQ_COL_NAME = "frequency (%)"
_COUNT_COL_NAME = "count"
_DEFAULT_NBINS_HISTOGRAM = 50
_MAIN_COLOR = "#1F77B4"
_NUM_DEFAULT_NBINS = 50
_KDE_DEFAULT_NPOINTS = 300


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


# num num
def calculate_bin_number(data_sorted, bin_edges):
    if len(bin_edges) <= 0:
        return _np.full(len(data_sorted), -1)
    i = 0
    e = 1
    bin_number = _np.zeros(len(data_sorted))
    for data in data_sorted:
        if e >= len(bin_edges):
            bin_number[i] = _np.NaN
        elif data <= bin_edges[e]:
            bin_number[i] = e - 1
        else:
            bin_number[i] = e
            e += 1
        i += 1
    return bin_number


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


def tranform_numeric_column(df, col, dict_format):
    df_temp = _pd.DataFrame(index=df.index, columns=[col])
    for idx in df.index:
        fmt = dict_format.get(idx, ".2f")
        df_temp.loc[idx, col] = f"{{:{fmt}}}".format(float(df.loc[idx, col]))
    return df_temp[col]


def describe_1d_numeric_table(df, col, percentiles=None, fmt=",.2f", is_2d=False):
    percentile_df = describe_utils.describe_percentiles_shortened(df, col, percentiles)
    dict_format = {
        "count": ",.0f",
        "mean": ",.2f",
        "tr_mean": ",.2f",
        "std": ",.2f",
        "null": ".2%",
        "skew": ".2f",
        "kurt": ".2f",
    }
    for stat in percentile_df.index:
        if stat not in ["count", "null", "mean", "tr_mean", "std", "skew", "kurt"]:
            dict_format[stat] = fmt
        # percentile_df.loc[stat, "value"] = f'{{:{fmt_stat}}}'.format(percentile_df.loc[stat, "value"])
    percentile_df["value"] = tranform_numeric_column(
        percentile_df, "value", dict_format
    )

    # concat the 2
    if is_2d:
        # quantile stats
        percentiles = (
            ["min", "1%", "5%"]
            + [f"{i * 10}%" for i in range(1, 10)]
            + ["95%", "99%", "max"]
        )
        # descriptive stats
        descriptives = ["count", "null", "mean", "tr_mean", "std", "skew", "kurt"]
        df_stat = (
            percentile_df.loc[descriptives + percentiles]
            .rename_axis("stat", axis="index")
            .reset_index()
        )
    else:
        # quantile stats
        percentiles = (
            ["1%", "5%"] + [f"{i * 10}%" for i in range(1, 10)] + ["95%", "99%"]
        )
        quantiles_df = (
            percentile_df.loc[percentiles]
            .rename_axis("percentile", axis="index")
            .reset_index()
        )
        # descriptive stats
        descriptives = [
            "count",
            "null",
            "min",
            "max",
            "mode",
            "mean",
            "tr_mean",
            "std",
            "mad",
            "range",
            "IQR",
            "skew",
            "kurt",
        ]
        descriptive_df = (
            percentile_df.loc[descriptives]
            .rename_axis("stat", axis="index")
            .reset_index()
        )
        # concat
        df_stat = _pd.concat([descriptive_df, quantiles_df], axis=1)
    df_stat = df_stat.set_index("stat").rename_axis(None)
    return df_stat


def _describe_1d_numeric_get_plot(
    df,
    col,
    show_density=True,
    show_box=True,
    nbins=_NUM_DEFAULT_NBINS,
    plot_kwargs={},
    binsize=None,
    is_trim=False,
    kde_npoints=_KDE_DEFAULT_NPOINTS,
):
    # histogram plot
    bin_edges, bin_values, bin_centers, binsize = describe_utils.calculate_bin(
        df[col], nbins=nbins, binsize=binsize
    )
    text = [
        f"{col}={left_edge}-{right_edge}"
        for left_edge, right_edge in zip(bin_edges[:-1], bin_edges[1:])
    ]
    # start plotting
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=bin_values,
            name="count",
            hovertext=text,
            hovertemplate="%{hovertext}<br>count=%{y:.0f}",
        )
    )
    fig.update_xaxes(
        tick0=bin_edges[0], dtick=binsize * _math.ceil(len(bin_edges) / 10)
    )
    fig.update_layout(bargap=0.01, width=620, height=480)
    # kde plot
    if show_density:
        if df[col].dropna().shape[0] > 1:
            kde_plot = describe_utils.make_kde(df[col].dropna(), n_points=kde_npoints)
            fig.add_trace(kde_plot, secondary_y=True)
    # box plot
    if show_box:
        box_plot = describe_utils.get_box_trace(df[col], color=_MAIN_COLOR)
        box_plot.update(xaxis="x3", yaxis="y3", name="", showlegend=False)
        fig.add_trace(box_plot)
    # set default title
    # update layout
    fig.layout.xaxis = {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": col}}
    fig.layout.xaxis3 = {
        "anchor": "y3",
        "domain": [0.0, 1.0],
        "matches": "x",
        "showticklabels": False,
    }
    fig.layout.yaxis = {
        "anchor": "x",
        "domain": [0.0, 0.8316],
        "title": {"text": "count"},
    }
    fig.layout.yaxis3 = {
        "anchor": "x3",
        "domain": [0.8416, 1.0],
        "matches": "y3",
        "showline": False,
        "showticklabels": False,
        "ticks": "",
    }
    if is_trim:
        title = f"Trimmed distribution of {col}"
    else:
        title = f"Distribution of {col}"
    fig_layout = {
        "title": title,
        "xaxis_title": col,
        "yaxis_title": "count",
        "width": 570,
        "height": 480,
        "legend": dict(
            title="", orientation="h", yanchor="top", y=1.12, xanchor="right", x=0.95
        ),
    }
    fig_layout.update(plot_kwargs)
    fig.update_layout(**fig_layout)

    if show_density:
        fig.data[1].hovertemplate = f"{col}=%{{x:.2f}}<br>kde=%{{y:.4f}}"
        fig.layout.yaxis2.update(title_text="density")
    return fig
