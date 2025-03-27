#
import warnings

#
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import seaborn as _sns
from IPython.core.display import HTML as _HTML

#
from IPython.core.display import display as _display

warnings.simplefilter("ignore")


def get_mode_table(df):
    """
    Get mode value information of each column in a DataFrame.
    Only return a random mode value in case many modes are detected.
    """
    n_rows = df.shape[0]
    mode_values = []
    for col in df.columns:
        values = df[col].convert_dtypes().value_counts(dropna=False)
        mode_values.append([col, values.idxmax(), values.max()])

    mode_df = (
        _pd.DataFrame(mode_values, columns=["column", "mode", "mode_count"])
        .assign(mode_perc=lambda x: x["mode_count"] / n_rows)
        .set_index("column")
    )
    return mode_df


def data_overview(df, verbose=False):
    def highlight(row):
        """Heuristics to add visual cues."""
        ret = ["" for _ in row.index]

        if row["dtype"].name == "object":
            ret[row.index.get_loc("dtype")] = "color: red; font-weight: bold"

        if (row.unique_perc <= 0.05) & (row.unique_count <= 100):
            ret[row.index.get_loc("unique_count")] = "color: blue; font-weight: bold"
        if row.unique_perc <= 0.1:
            ret[row.index.get_loc("unique_perc")] = "color: blue"

        if row.null_perc >= 0.2:
            ret[row.index.get_loc("null_count")] = "color: red; font-weight: bold"
            ret[row.index.get_loc("null_perc")] = "color: red; font-weight: bold"
        elif row.null_perc >= 0.1:
            ret[row.index.get_loc("null_count")] = "color: red"
            ret[row.index.get_loc("null_perc")] = "color: red"

        if row.unique_count <= 1:
            ret[row.index.get_loc("unique_count")] = "background-color: yellow"

        if row.zero_perc >= 0.2:
            ret[row.index.get_loc("zero_count")] = "color: red; font-weight: bold"
            ret[row.index.get_loc("zero_perc")] = "color: red; font-weight: bold"
        elif row.zero_perc >= 0.1:
            ret[row.index.get_loc("zero_count")] = "color: red"
            ret[row.index.get_loc("zero_perc")] = "color: red"

        if row.blank_perc >= 0.2:
            ret[row.index.get_loc("blank_count")] = "color: red; font-weight: bold"
            ret[row.index.get_loc("blank_perc")] = "color: red; font-weight: bold"
        elif row.blank_perc >= 0.1:
            ret[row.index.get_loc("blank_count")] = "color: red"
            ret[row.index.get_loc("blank_perc")] = "color: red"

        if row.dup_perc <= 0.2:
            ret[row.index.get_loc("dup_count")] = "color: blue; font-weight: bold"
            ret[row.index.get_loc("dup_perc")] = "color: blue; font-weight: bold"

        if row.hit_rate <= 0.5:
            ret[row.index.get_loc("hit_rate")] = "color: red; font-weight: bold"
        elif row.hit_rate <= 0.6:
            ret[row.index.get_loc("hit_rate")] = "color: red"
        return ret

    # total = df.shape[0]
    n_rows = df.shape[0]
    data_type = df.dtypes
    object_cols = data_type[data_type == "object"]

    # Convert object columns (list, set, ndarray, ...) to string for stats computing
    if len(object_cols) > 0:
        df = df.copy()
        df[object_cols.index] = df[object_cols.index].astype("string")

    unique_count = df.nunique()

    null_count = df.isnull().sum()
    zero_count = df.isin([0, "0"]).sum()
    blank_count = df.isin(["", "{}", "[]"]).sum()

    unique_perc = unique_count / n_rows
    null_perc = null_count / n_rows
    zero_perc = zero_count / n_rows
    blank_perc = blank_count / n_rows

    dup_count = []
    dup_perc = []
    for column_name in df.columns:
        tmp_count = df.duplicated(subset=column_name, keep="first").sum()
        dup_count.append(tmp_count)
        dup_perc.append(tmp_count / n_rows)

    hit_rate = 1 - null_perc

    missing_df = _pd.DataFrame(
        {
            "total": n_rows,
            "dtype": data_type,
            "unique_count": unique_count,
            "unique_perc": unique_perc,
            "dup_count": dup_count,
            "dup_perc": dup_perc,
            "blank_count": blank_count,
            "zero_count": zero_count,
            "null_count": null_count,
            "blank_perc": blank_perc,
            "zero_perc": zero_perc,
            "null_perc": null_perc,
            "hit_rate": hit_rate,
        }
    )
    # .sort_values(['null_count', 'zero_count', 'blank_count'], ascending=[False, False, False])

    missing_df.insert(0, "#", _np.arange(1, 1 + len(missing_df)))  # add columns index

    if verbose:
        mode_df = get_mode_table(df)
        missing_df = _pd.concat([missing_df, mode_df], axis=1)

    # Add Pandas style
    border_props = [
        ("border-left-color", "black"),
        ("border-left-style", "dotted"),
        ("border-left-width", "thin"),
    ]

    missing_df_style = (
        missing_df.style
        # .set_caption(f'Data Overview')
        .format(
            {
                "blank_count": lambda x: x if x > 0 else "-",
                "zero_count": lambda x: x if x > 0 else "-",
                "null_count": lambda x: x if x > 0 else "-",
                "dup_count": lambda x: x if x > 0 else "-",
                "unique_perc": lambda x: "{:.2%}".format(x) if x >= 0.1 else "-",
                "blank_perc": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-",
                "zero_perc": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-",
                "null_perc": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-",
                "dup_perc": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-",
                "hit_rate": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-",
            }
        )
        .apply(highlight, axis=1)
        .set_table_styles(
            [
                dict(
                    selector="caption",
                    props=[
                        ("color", "black"),
                        ("font-size", "16px"),
                        ("font-weight", "bold"),
                    ],
                ),
                dict(selector="th.col_heading.level0.col3", props=border_props),
                dict(selector="th.col_heading.level0.col5", props=border_props),
                dict(selector="th.col_heading.level0.col7", props=border_props),
                dict(selector="th.col_heading.level0.col10", props=border_props),
            ]
        )
        .set_properties(**{"text-align": "center"})
        .set_properties(
            **dict(border_props),
            subset=["unique_count", "dup_count", "blank_count", "blank_perc"],
        )
    )

    if verbose:
        missing_df_style = (
            missing_df_style.format(
                {"mode_perc": lambda x: "{:.2%}".format(x) if x >= 0.01 else "-"},
                subset=["mode_perc"],
            )
            .set_properties(**dict(border_props), subset=["mode"])
            .set_properties(
                **{
                    "max-width": "200px",
                    "overflow": "hidden",
                    "text-overflow": "ellipsis",
                },
                subset=["mode"],
            )
            .set_table_styles(
                [dict(selector="th.col_heading.level0.col14", props=border_props)],
                overwrite=False,
            )
        )

    return missing_df_style


#
def display_df_with_title(df, title):
    """
    df : pandas dataframe
    title : title
    """
    output = df.style.set_table_attributes("style='font-size:100%'")._repr_html_()
    output = "<b>{}</b>".format(title) + output
    _display(_HTML(output))


#
def describe_percentiles(df, col, percentiles=None, fmt=".2f"):
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
        ["count", "nan", "mean", "std", "skew", "kurt", "min"]
        + [f"{int(x * 100)}%" for x in sorted(percentiles)]
        + ["max"],
        :,
    ]
    return percentile_df


def style_describe_percentiles(percentile_df):
    percentile_df_style = (
        percentile_df.style.format("{:.1f}")
        .format("{:.2%}", subset=_pd.IndexSlice[["nan"], :])
        .format(
            "{:.1f}",
            subset=_pd.IndexSlice[
                [
                    x
                    for x in percentile_df.index
                    if x not in ["skew", "kurt", "nan", "std"]
                ],
                :,
            ],
        )
        .set_table_styles(
            [
                dict(
                    selector="caption",
                    props=[
                        ("font-size", "15px"),
                        ("font-weight", "bold"),
                        (
                            "max-width",
                            "150px",
                        ),  # avoid long index name makes the table uneccesary wide
                        ("overflow", "visible"),
                    ],
                )
            ]
        )
    )
    return percentile_df_style


def descibe_2d_category_data_extend(
    df_input, cate_col1, cate_col2, dropna=True, threshold_cate_1=9, threshold_cate_2=9
):
    """
    Show describe table 2 categories data
    Args:
        df_input: dataframe contain 2 categories column
        cate_col1: category column 1
        cate_col2: category column 2
        dropna: if true, input data will drop na on columns cate_col1 and cate_col2
        threshold_cate_1: Maximum category show cate_col1 (all other will combine to Others category)
        threshold_cate_2: Maximum category show cate_col2 (all other will combine to Others category)
    Returns:
        Table describe data (count values, percentiles)
    """
    df_input = df_input.copy()
    if dropna:
        df_input = df_input.dropna(subset=[cate_col1, cate_col2])

    #     df_input[cate_col1] = _data_format_helper._format_limit_data_with_top_number_value(df_input, column_name=cate_col1,
    #                                                                                        ntop=threshold_cate_1)
    #     df_input[cate_col2] = _data_format_helper._format_limit_data_with_top_number_value(df_input, column_name=cate_col2,
    #                                                                                        ntop=threshold_cate_2)
    df_count = (
        df_input.groupby([cate_col1, cate_col2])[cate_col2].size().unstack(fill_value=0)
    )
    list_columns = list(df_count.columns)
    df_count.loc["Total"] = df_count.sum()
    df_percentile = _pd.DataFrame()
    for value in list_columns:
        df_percentile["percentile_hor_" + str(value)] = df_count[value] / df_count.sum(
            axis=1
        )
        df_percentile["percentile_hor_" + str(value)] = (
            df_percentile["percentile_hor_" + str(value)] * 100
        ).round(2)
    for value in list_columns:
        df_percentile["percentile_ver_" + str(value)] = df_count[value] / df_count[
            value
        ][:-1].sum(axis=0)
        df_percentile["percentile_ver_" + str(value)] = (
            df_percentile["percentile_ver_" + str(value)] * 100
        ).round(2)

    df_count["Total"] = df_count.sum(axis=1)
    df_count["%Total"] = (
        df_count["Total"] / df_count["Total"][:-1].sum(axis=0) * 100
    ).round(2)
    df_output = df_count.join(df_percentile)
    df_output = df_output.reset_index()
    df_output[cate_col1] = df_output[cate_col1].astype(str)
    df_output = df_output.rename(columns={cate_col1: cate_col1 + " / " + cate_col2})
    return df_output


#
def describe_sns_2d_numeric_with_label(
    df, feature_name, label_name="label", figsize=(32, 5)
):
    """
    Plot 2d histogram chart and line chart numeric with label column
    """
    fig, ax = _plt.subplots(1, 2, figsize=figsize)
    if isinstance(df[label_name], (int, float, complex)):
        label_values = sorted(df[label_name].unique())
    else:
        label_values = df[label_name].unique()
    _sns.kdeplot(
        x=df[feature_name],
        hue=df[label_name],
        hue_order=label_values,
        common_norm=False,
        ax=ax[0],
    )
    if len(df[feature_name].unique()) > 1000:
        _sns.histplot(
            x=df[feature_name],
            hue=df[label_name],
            hue_order=label_values,
            multiple="stack",
            ax=ax[1],
            element="poly",
        )
    else:
        _sns.histplot(
            x=df[feature_name],
            hue=df[label_name],
            hue_order=label_values,
            discrete=True,
            multiple="stack",
            ax=ax[1],
            element="poly",
        )
    return fig, ax


#
def describe_numeric_with_label(df, column_name, label="label"):
    values = df[label].dropna().unique()
    df_describe = _pd.DataFrame({})
    for value in values:
        tmp = (
            df.loc[df[label] == value, column_name]
            .describe(percentiles=[i * 0.1 for i in range(10)] + [0.01] + [0.99])
            .reset_index()
        )
        tmp.columns = ["Describe", "Label_" + str(value)]
        if len(df_describe.columns) > 0:
            df_describe = _pd.merge(df_describe, tmp, on="Describe", how="inner")
        else:
            df_describe = tmp

    return df_describe


#
def describe_cate_with_label_stype(pdf):
    def highlight_max(s, props=""):
        return _np.where(s == _np.nanmax(s[:-1].values), props, "")

    def highlight_min(s, props=""):
        return _np.where(s == _np.nanmin(s[:-1].values), props, "")

    return (
        pdf.style.background_gradient(cmap="PuBu", subset=["Total"])
        .apply(
            highlight_max,
            axis=0,
            props="color: red; font-weight: bold",
            subset=["percentile_ver_0", "percentile_ver_1", "Total"],
        )
        .apply(
            highlight_min,
            axis=0,
            props="color: blue",
            subset=["percentile_ver_0", "percentile_ver_1", "Total"],
        )
    )


#
def plot_correlation(plot_df, tshold=0.9, title=""):
    corr = plot_df.corr()
    mask = _np.triu(_np.ones_like(corr, dtype=bool))
    corr = corr.copy()
    fig, ax = _plt.subplots(figsize=(24, 24))
    _sns.heatmap(
        corr[corr > tshold],
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        cbar=False,
    )
    ax.tick_params(left=False, bottom=False)
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize=12
    )
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    _plt.title(title, fontsize=16)
    fig.show()


#
DEFAULT_COLOR_LIST = _plt.rcParams["axes.prop_cycle"].by_key()["color"] * 10


def tables_side_by_side(
    dfs: dict,
    name: str = "",
    colors=DEFAULT_COLOR_LIST,
    precision=2,
    with_grad=False,
    show=True,
):
    spaces = "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0"
    p = _np.max([0, int(precision)])

    output = ""
    for i, (caption, df) in enumerate(dfs.items()):
        styles = [
            dict(
                selector="caption",
                props=[
                    ("text-align", "center"),
                    ("font-size", "120%"),
                    ("color", colors[i]),
                ],
            )
        ]
        new_table = (
            df.round(p)
            .style.set_table_attributes("style='display:inline; font-size:100%;'")
            .set_caption(caption)
            .set_table_styles(styles)
        )
        if with_grad:
            if with_grad == 2:
                new_table = new_table.background_gradient(axis=1)
            else:
                new_table = new_table.background_gradient()

        output += new_table._repr_html_()
        output += spaces

    output = f"<h3>{name}</h3>" + output
    if show:
        _display(_HTML(output))
    else:
        return output


1
