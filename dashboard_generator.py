from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
import streamlit as st
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    initialise_page()
    df = file_upload()
    if df is not None:
        generate_dashboards(df)


def initialise_page():
    st.title("Automatic dashboard generator")


def file_upload() -> pd.DataFrame | None:
    data = st.file_uploader("upload file", type=["xlsx", "csv"])
    if data is not None:
        try:
            return pd.read_csv(data)
        except UnicodeDecodeError:
            return pd.read_excel(data)


def combinatorics(input_list):
    pool = tuple(input_list)
    n = len(pool)
    if 2 > n:
        return
    indices = list(range(2))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(2)):
            if indices[i] != i + n - 2:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, 2):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def plot_scatterplots(data, columns):
    # Ensure only unique combinations get shown
    columns_comb = list(combinatorics(columns))

    # First we need to calculate the amount of plots to create
    # If we have an even number of plots, we create 2 columns
    # We also calculate the ideal height and width for the plots
    if len(columns_comb) % 2 == 0:
        cols = 2
        rows = len(columns_comb) // 2
        plot_height = len(columns_comb) * 150
    else:
        cols = 1
        rows = len(columns_comb)
        plot_height = len(columns_comb) * 500

    # Calculate the correlations between the selected columns
    correlations = [
        data[[columns_comb[i][0], columns_comb[i][1]]]
        .corr()[columns_comb[i][0]]
        .iloc[1]
        for i in range(len(columns_comb))
    ]

    # Initialise the subplots for the columns
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=tuple(
            [
                f"The correlation between {columns_comb[i][0]} and {columns_comb[i][1]} is {round(correlations[i] * 100, 2)}%"
                for i in range(len(columns_comb))
            ]
        ),
    )

    # Add all the scatterplots to the large plot
    for index, scatter_cols in enumerate(columns_comb):
        index += 1
        col1, col2 = scatter_cols

        # Select the right row and column for this iteration of the loop
        if cols == 1:
            col_number = 1
            row_number = index
        elif index % 2 == 0:
            row_number = index // 2
            col_number = 2
        else:
            row_number = index // 2 + 1
            col_number = 1

        # Add the trace for the scatterplot
        fig.add_trace(
            go.Scatter(x=data[col1], y=data[col2], mode="markers"),
            row=row_number,
            col=col_number,
        )

        # Add text to the plots
        fig.update_xaxes(title_text=col1, row=row_number, col=col_number)
        fig.update_yaxes(title_text=col2, row=row_number, col=col_number)

    # Ensure a comfortable height and width for the plot
    fig.update_layout(height=plot_height, width=1200, showlegend=False)
    st.warning("Correlation does not imply causation")
    st.plotly_chart(fig)


def plot_line_chart(data: pd.DataFrame, columns: List[str]):
    fig = go.Figure()
    for col in columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data[col], mode="lines+markers", name=col)
        )
    st.plotly_chart(fig)


def plot_bar_chart(ser: pd.Series):
    value_counts = ser.value_counts(ascending=False)
    unique_categories = value_counts.index
    unique_categories_count = value_counts.values

    fig = go.Figure()
    fig.add_bar(x=unique_categories, y=unique_categories_count)
    st.plotly_chart(fig)


def generate_dashboards(input_df: pd.DataFrame):
    columns = input_df.columns
    selected_cols = st.multiselect("select columns", columns)
    numeric_cols = []
    other_cols = []
    for col in selected_cols:
        # st.write(f"{input_df[col].dtype}")
        if is_numeric_dtype(input_df[col]):
            numeric_cols.append(col)
        else:
            other_cols.append(col)

    if len(numeric_cols) >= 1:
        st.header("Line charts")
        plot_line_chart(input_df, numeric_cols)

    if len(numeric_cols) >= 2:
        st.write(numeric_cols)
        st.header("Scatterplots")
        plot_scatterplots(input_df, numeric_cols)

    for col in other_cols:
        st.header("Bar charts")
        plot_bar_chart(input_df[col])


if __name__ == "__main__":
    main()
