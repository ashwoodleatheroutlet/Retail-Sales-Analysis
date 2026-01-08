import streamlit as st
import altair as alt
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
import seaborn as sns
import datetime
import re

pd.options.mode.chained_assignment = None

def dataframe_with_selections(df, inp_key):
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", False)

    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config={"Select": st.column_config.CheckboxColumn(required=False)},
        disabled=df.columns,
        key=inp_key,
        use_container_width=True,
    )

    selected_rows = edited_df[edited_df["Select"]].drop(columns=["Select"])
    return {"selected_rows": selected_rows}

def graph_condense(df):

    # Ensure Date is datetime (dtype=str safe)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Date"])

    # Units numeric (dtype=str safe)
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0)

    # Filter 2023–2025
    df = df[df["Date"].between("2023-01-01", "2025-12-31")]

    # Monthly aggregation
    graph_df = (
        df.groupby(pd.Grouper(key="Date", freq="MS"))["Units"]
          .sum()
          .reset_index()
    )

    # Fill missing months with 0
    all_months = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    graph_df = (
        graph_df.set_index("Date")
                .reindex(all_months, fill_value=0)
                .rename_axis("Date")
                .reset_index()
    )

    # IMPORTANT: don't set domain with pandas Timestamps
    chart = alt.Chart(graph_df, title="Units Sold (Monthly) 2023–2025").mark_line(point=True).encode(
        x=alt.X("Date:T", title="Month"),
        y=alt.Y("Units:Q", title="Units"),
        tooltip=["Date:T", "Units:Q"]
    )

    return chart


st.title('Sales Analysis')

df = pd.read_csv('Data.csv', dtype=str)
image_file = pd.read_csv('Return Inventory File.csv', dtype=str)
raw_file = pd.read_csv('Raw file for graph.csv', dtype=str)

raw_file['Date'] = raw_file['Date created']
raw_file['Units'] = raw_file['Quantity'].astype(int)

image_file.dropna(subset=['ImageURL'], inplace=True)
image_file.drop_duplicates(subset=['Style'], inplace=True)
category_arr = df['Category'].unique().tolist()
# category_arr = np.sort(category_arr).tolist()
# category_arr = ['All categories'] + category_arr
category_options = st.selectbox('Select a category', options=category_arr)

df = df[df['Category'] == category_options]

df['2023'] = df['2023'].astype(int)
df['2024'] = df['2024'].astype(int)
df['2025'] = df['2025'].astype(int)
df['Stock'] = df['Stock'].astype(int)
df_style = df.groupby(['Style'])[['2023', '2024', '2025', 'Stock']].sum().reset_index().sort_values(by='Stock', ascending=False)
dict_rrp = pd.Series(df['RRP'].values,index=df['Style']).to_dict()
dict_saleprice = pd.Series(df['Sale price'].values,index=df['Style']).to_dict()
df_style['RRP'] = df_style['Style']
df_style['Sale price'] = df_style['Style']
df_style['RRP'] = df_style['RRP'].map(dict_rrp)
df_style['Sale price'] = df_style['Sale price'].map(dict_saleprice)
selection1 = dataframe_with_selections(df_style, 1)
if not selection1["selected_rows"].empty:
    selected_style = selection1["selected_rows"].iloc[0]["Style"]
    df = df[df["Style"] == selected_style]
    raw_file = raw_file[raw_file["Style"] == selected_style]
    image_col, dataframe_col = st.columns([0.4, 0.6])

    image_url = image_file[image_file['Style'] == selected_style]['ImageURL'].iloc[0]
    image_col.image(image_url, width=400)
    df_colour = df.groupby(['Style', 'Colour'])[['2023', '2024', '2025', 'Stock']].sum().reset_index().sort_values(by='Stock', ascending=False).reset_index(drop=True)
    dataframe_col.dataframe(df_colour, use_container_width=True)

    chart = graph_condense(raw_file)
    st.altair_chart(chart, use_container_width=True)