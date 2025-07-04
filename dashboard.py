import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import psycopg2
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import os
from st_aggrid import AgGrid
import ast
from st_aggrid import GridOptionsBuilder
import math
from pandas.api.types import is_numeric_dtype
import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

host = "27.71.237.112"
database = "pbreport"
user = "pbuser"
password = "p0w3rb!"

#Auto refresh mỗi 24h
#1 phút = 60 giây = 60.000 ms
#1 giờ = 60 phút = 3.600.000 ms
#24 giờ = 24 × 3.600.000 = 86.400.000 ms

st_autorefresh(interval=10800000, key="datarefresh")

st.markdown(
    '''
    <style>
    .block-container {
        padding-left: 4.0rem !important;
        padding-right: 4.0rem !important;
        max-width: 100vw !important;
    }
    .main .block-container {
        max-width: 100vw !important;
        padding-left: 4.0rem !important;
        padding-right: 4.0rem !important;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Tạo một hàm để quản lý kết nối database
def get_db_connection():
    return psycopg2.connect(
        host=host, 
        database=database, 
        user=user, 
        password=password
    )

# Sử dụng context manager để đảm bảo đóng kết nối
def execute_query(query):
    with get_db_connection() as connection:
        return pd.read_sql(query, connection)

# 1. Tách hàm load từng bảng và cache riêng biệt
@st.cache_data
def load_ticket():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = """
    SELECT id, number, stage_id, approved_date, last_stage_update, category_id, team_id, create_date, priority, mall_id, sla_reached_late
    FROM helpdesk_ticket
    """
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data
def load_category():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, name FROM helpdesk_ticket_category"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data
def load_team():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, name FROM helpdesk_ticket_team"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data
def load_tag():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = """
    SELECT helpdesk_ticket_id, helpdesk_ticket_tag_id 
    FROM helpdesk_ticket_helpdesk_ticket_tag_rel
    """
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data
def load_res_partner():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, display_name, create_date, function, street, email, phone, mobile, active, helpdesk_team_id, mall_code, is_company FROM res_partner"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data
def load_res_partner_display_name():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, display_name FROM res_partner"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data
def load_helpdesk_ticket():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT * FROM helpdesk_ticket"
    df = pd.read_sql(query, connection)
    connection.close()
    return df


df = load_ticket()
df_category = load_category()
df_team = load_team()
df_tag = load_tag()
df_res_partner = load_res_partner()
df_res_partner_display_name = load_res_partner_display_name()

# 2. Chuyển đổi kiểu dữ liệu ngay sau khi load
for col in ['create_date', 'approved_date', 'last_stage_update']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Lọc ticket có create_date từ 23/10/2023 13:52:55 trở đi
start_date = pd.Timestamp('2023-10-23 13:52:55')
df = df[df['create_date'] >= start_date]

# 3. Merge dữ liệu hợp lý
df = df.merge(df_category, left_on='category_id', right_on='id', how='left', suffixes=('', '_cat'))
df = df.merge(df_team.rename(columns={'name': 'team_name'}), left_on='team_id', right_on='id', how='left', suffixes=('', '_team'))
df = df.merge(df_tag, left_on='id', right_on='helpdesk_ticket_id', how='left', suffixes=('', '_tag'))

df['category_name'] = df['name']
df['team_name'] = df['team_name']

# --- ĐẶT ĐOẠN CLEAN Ở ĐÂY ---
def extract_en_name(val):
    try:
        if isinstance(val, dict):
            return val.get('en_US', str(val))
        if isinstance(val, str) and val.startswith("{'en_US':"):
            d = ast.literal_eval(val)
            return d.get('en_US', val)
        return val
    except Exception:
        return val

df['category_name'] = df['category_name'].apply(extract_en_name)
# --- ĐẾN ĐÂY ---

# --- CHUẨN HÓA mall_id và merge với res_partner để lấy mall_display_name ---
df['mall_id'] = pd.to_numeric(df['mall_id'], errors='coerce').astype('Int64')
df_res_partner_display_name = load_res_partner_display_name()
df_res_partner_display_name['id'] = pd.to_numeric(df_res_partner_display_name['id'], errors='coerce').astype('Int64')
df = df.merge(
    df_res_partner_display_name[['id', 'display_name']],
    left_on='mall_id',
    right_on='id',
    how='left',
    suffixes=('', '_res_partner')
)
df['mall_display_name'] = df['display_name']

# 4. Loại bỏ team không cần thiết
team_ids_exclude = [6,7, 12, 13, 25]
df = df[~df['team_id'].isin(team_ids_exclude)]

# Loại bỏ category_id = 3,7,9
category_ids_exclude = [3, 7, 9]
df = df[~df['category_id'].isin(category_ids_exclude)]

# 5. Tối ưu tạo custom_end_date (vectorized)
mask_approved = df['approved_date'].notnull() & (df['approved_date'].astype(str).str.strip() != "") & (df['approved_date'].astype(str).str.strip() != "0")
mask_stage = df['stage_id'].isin([9, 10, 11, 12])
df['custom_end_date'] = "not yet end"
df.loc[mask_approved, 'custom_end_date'] = df.loc[mask_approved, 'approved_date']
df.loc[mask_stage & ~mask_approved, 'custom_end_date'] = df.loc[mask_stage & ~mask_approved, 'last_stage_update']

# Nếu custom_end_date là datetime, chuyển về string cho các giá trị khác "not yet end"
df['custom_end_date'] = df['custom_end_date'].apply(
    lambda x: "not yet end" if x == "not yet end" else (x.strftime("%Y-%m-%d %H:%M:%S") if pd.notnull(x) and not isinstance(x, str) else x)
)

# 6. Tạo tuần
week_starts = [datetime(2025, 3, 3) + timedelta(weeks=i) for i in range(18)]
week_ends = [start + timedelta(days=6, hours=23, minutes=59, seconds=59) for start in week_starts]
week_labels = [f"W{10+i} ({start.strftime('%d/%m')} - {end.strftime('%d/%m')})" for i, (start, end) in enumerate(zip(week_starts, week_ends))]

# 7. Tạo bảng kiểm tra theo Category
df['category_name'] = df['category_name'].astype(str)
category_names = df['category_name'].dropna().unique()
table_data = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names:
        mask = (
            (df['category_name'] == cat) &
            (df['create_date'] <= end) &
            (
                (df['custom_end_date'] == "not yet end") |
                (
                    (df['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df[mask].shape[0]
        row[cat] = count
    table_data.append(row)
df_table = pd.DataFrame(table_data)

# 8. Tạo bảng kiểm tra theo Team
df['team_name'] = df['team_name'].astype(str)
team_names = df['team_name'].dropna().unique()
team_names_exclude = df_team[df_team['id'].isin(team_ids_exclude)]['name'].tolist()
team_names = [t for t in team_names if t not in team_names_exclude]
table_data_team = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for team in team_names:
        mask = (
            (df['team_name'] == team) &
            (df['create_date'] <= end) &
            (
                (df['custom_end_date'] == "not yet end") |
                (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
            )
        )
        count = df[mask].shape[0]
        row[team] = count
    table_data_team.append(row)
df_table_team = pd.DataFrame(table_data_team)

# Tạo bảng kiểm tra theo Priority
table_data_priority = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    
    # Low priority
    mask_low = (
        (df['helpdesk_ticket_tag_id'] != 3) &
        (
            (df['priority'].isna()) |
            (df['priority'].astype(str).str.strip() == '0') |
            (df['priority'].fillna(0).astype(int) == 0) |
            (df['priority'].fillna(0).astype(int) == 1)
        ) &
        (df['create_date'] <= end) &
        (
            (df['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df[mask_low].shape[0]
    
    # Medium priority
    mask_medium = (
        (df['helpdesk_ticket_tag_id'] != 3) &
        (df['priority'].fillna(0).astype(int) == 2) &  # Xử lý NULL bằng cách fillna(0)
        (df['create_date'] <= end) &
        (
            (df['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df[mask_medium].shape[0]
    
    # High priority
    mask_high = (
        (df['helpdesk_ticket_tag_id'] != 3) &
        (df['priority'].fillna(0).astype(int) == 3) &  # Xử lý NULL bằng cách fillna(0)
        (df['create_date'] <= end) &
        (
            (df['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df[mask_high].shape[0]
    
    # Emergency
    mask_emergency = (
        (df['helpdesk_ticket_tag_id'] == 3) &
        (df['create_date'] <= end) &
        (
            (df['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df[mask_emergency].shape[0]
    
    table_data_priority.append(row)
df_table_priority = pd.DataFrame(table_data_priority)

# WATERFALL CHART - Tính bảng kiểm tra số lượng Created và Solved ticket theo tuần (dùng cho cả hai trang)
created_counts = []
solved_counts = []
waterfall_week_labels = []
for i in range(15):
    start = week_starts[i]
    end = week_ends[i]
    week_label = week_labels[i]
    waterfall_week_labels.append(week_label)
    created = df[(df['create_date'] >= start) & (df['create_date'] <= end)].shape[0]
    solved = df[(pd.to_datetime(df['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)
result_df = pd.DataFrame({
    'Tuần': waterfall_week_labels,
    'Created': created_counts,
    'Solved': solved_counts
})

# Tạo bảng kiểm tra theo Banner
banner_names = [
    "GO Mall",
    "Hyper",
    "Tops",
    "CBS",
    "Nguyen Kim",
    "KUBO",
    "mini go!"
]

# Chuẩn hóa mall_display_name
df['mall_display_name'] = df['mall_display_name'].fillna('').str.strip()

table_data_banner = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for banner in banner_names:
        # Điều kiện 1: custom_end_date == "not yet end"
        mask1 = (
            df['mall_display_name'].str.lower().str.startswith(banner.lower(), na=False) &
            (df['create_date'] <= end) &
            (df['custom_end_date'] == "not yet end")
        )
        # Điều kiện 2: custom_end_date != "not yet end" và custom_end_date > end
        mask2 = (
            df['mall_display_name'].str.lower().str.startswith(banner.lower(), na=False) &
            (df['create_date'] <= end) &
            (df['custom_end_date'] != "not yet end") &
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > end)
        )
        count = df[mask1].shape[0] + df[mask2].shape[0]
        row[banner] = count
    table_data_banner.append(row)
df_table_banner = pd.DataFrame(table_data_banner)

# Thêm cột processing_time
now = pd.Timestamp.now()
def calc_duration(row):
    try:
        if row['custom_end_date'] == 'not yet end':
            return (now - row['create_date']).days
        else:
            end_date = pd.to_datetime(row['custom_end_date'], errors='coerce')
            return (end_date - row['create_date']).days
    except Exception:
        return None

df['processing_time'] = df.apply(calc_duration, axis=1)

# Tính toán Start date và End date cho các cột tính toán
today = pd.Timestamp.now().normalize()

if today.dayofweek == 0:
    start_date = today - pd.Timedelta(days=3)
else:
    start_date = today - pd.Timedelta(days=1)

end_date = today

# Thêm cột tính UNDER THIS MONTH REPORT
def calculate_condition(row):
    create_date = pd.to_datetime(row['create_date'])
    custom_end_date = row['custom_end_date']
    
    # Điều kiện 1: create_date nằm trong khoảng start_date và end_date
    condition1 = (create_date >= start_date) and (create_date <= end_date)
    
    # Điều kiện 2: create_date < start_date VÀ (custom_end_date = "not yet end" HOẶC custom_end_date > end_date)
    if custom_end_date == "not yet end":
        condition2 = create_date < start_date
    else:
        try:
            end_date_value = pd.to_datetime(custom_end_date)
            condition2 = create_date < start_date and end_date_value > end_date
        except:
            condition2 = False
    
    return 1 if (condition1 or condition2) else 0

# Thêm cột tính Carry over ticket
def calculate_carry_over(row):
    create_date = pd.to_datetime(row['create_date'])
    custom_end_date = row['custom_end_date']
    
    # Điều kiện: custom_end_date = "not yet end" VÀ create_date < start_date
    return 1 if (custom_end_date == "not yet end" and create_date < start_date) else 0

df['Under this month report'] = df.apply(calculate_condition, axis=1)
df['Carry over ticket'] = df.apply(calculate_carry_over, axis=1)



    # Căn giữa title ở top center
st.markdown(
    """
    <h1 style='text-align: center; margin-top: 0; margin-bottom: 7rem;'>
        Facility Management System (FMS) Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)

# Stacked Column Chart theo Priority
fig_stack_priority = go.Figure()
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',      # xanh lá nhạt
    'Medium priority': '#fff9b1',   # vàng nhạt
    'High priority': '#ffd6a0',     # cam nhạt
    'Emergency': '#ff2222'          # đỏ tươi
}
for priority in priority_cols:
    y_values = df_table_priority[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority.add_trace(go.Bar(
        name=priority,
        x=df_table_priority["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
# --- Thêm box so sánh % giữa tuần hiện tại và tuần trước cho từng priority ---
idx_w = len(df_table_priority["Tuần"]) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority.add_trace(go.Scatter(
    x=df_table_priority["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority["Tuần"], totals_offset, totals)):
    fig_stack_priority.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        y=0.97,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    )
)
st.plotly_chart(fig_stack_priority)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

# Stacked Column Chart theo Category
fig_stack = go.Figure()
for cat in category_names:
    y_values = df_table[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack.add_trace(go.Bar(
        name=cat,
        x=df_table["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
# --- Thêm box so sánh % giữa tuần hiện tại và tuần trước cho từng category ---
idx_w = len(df_table["Tuần"]) - 1
idx_w1 = idx_w - 1
w_label = df_table["Tuần"].iloc[idx_w]
active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names:
    count_w = float(df_table[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w
if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table[category_names].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack.add_trace(go.Scatter(
    x=df_table["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table["Tuần"], totals_offset, totals)):
    fig_stack.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    )
)
st.plotly_chart(fig_stack)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Column Chart theo Team
fig_stack_team = go.Figure()
for team in df_table_team.columns:
    if team == "Tuần":
        continue
    y_values = df_table_team[team].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_team.add_trace(go.Bar(
        name=team,
        x=df_table_team["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
# --- Thêm box so sánh % giữa tuần hiện tại và tuần trước cho từng team ---
team_cols = [col for col in df_table_team.columns if col != "Tuần"]
idx_w = len(df_table_team["Tuần"]) - 1
idx_w1 = idx_w - 1
w_label = df_table_team["Tuần"].iloc[idx_w]
active_teams = []
percent_changes = {}
team_positions = {}
cumulative_height = 0
for team in team_cols:
    count_w = float(df_table_team[team].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_team[team].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_teams.append(team)
    percent_changes[team] = percent
    team_positions[team] = cumulative_height + count_w / 2
    cumulative_height += count_w
if active_teams:
    total_height = cumulative_height
    x_vals = list(df_table_team["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_teams = sorted(active_teams, key=lambda x: team_positions[x])
    for i, team in enumerate(sorted_teams):
        percent = percent_changes[team]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = team_positions[team]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_teams)/2))
        fig_stack_team.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_team.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_team[team_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_team.add_trace(go.Scatter(
    x=df_table_team["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_team["Tuần"], totals_offset, totals)):
    fig_stack_team.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_team.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL EVOLUTION OA TICKETS PER REGION",
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    )
)
st.plotly_chart(fig_stack_team)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Column Chart theo Banner
fig_stack_banner = go.Figure()
banner_cols = ["GO Mall", "Hyper", "Tops", "CBS", "Nguyen Kim", "KUBO", "mini go!"]
for banner in banner_cols:
    y_values = df_table_banner[banner].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_banner.add_trace(go.Bar(
        name=banner,
        x=df_table_banner["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
# --- Thêm box so sánh % giữa tuần hiện tại và tuần trước cho từng banner ---
idx_w = len(df_table_banner["Tuần"]) - 1
idx_w1 = idx_w - 1
w_label = df_table_banner["Tuần"].iloc[idx_w]
active_banners = []
percent_changes = {}
banner_positions = {}
cumulative_height = 0
for banner in banner_cols:
    count_w = float(df_table_banner[banner].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_banner[banner].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_banners.append(banner)
    percent_changes[banner] = percent
    banner_positions[banner] = cumulative_height + count_w / 2
    cumulative_height += count_w
if active_banners:
    total_height = cumulative_height
    x_vals = list(df_table_banner["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_banners = sorted(active_banners, key=lambda x: banner_positions[x])
    for i, banner in enumerate(sorted_banners):
        percent = percent_changes[banner]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = banner_positions[banner]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_banners)/2))
        fig_stack_banner.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_banner.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_banner[banner_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_banner.add_trace(go.Scatter(
    x=df_table_banner["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_banner["Tuần"], totals_offset, totals)):
    fig_stack_banner.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_banner.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL EVOLUTION OA TICKETS PER BANNER",
        y=0.97,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    )
)
st.plotly_chart(fig_stack_banner)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Waterfall Chart
week_start = datetime(2025, 3, 3)
week_labels = []
created_counts = []
solved_counts = []
for i in range(18):
    start = week_start + timedelta(weeks=i)
    end = start + timedelta(days=6)
    week_label = f"W{10+i} ({start.strftime('%d/%m')} - {end.strftime('%d/%m')})"
    week_labels.append(week_label)
    created = df[(df['create_date'] >= start) & (df['create_date'] <= end)].shape[0]
    solved = df[(pd.to_datetime(df['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

# Tạo dữ liệu Waterfall step-by-step: mỗi tuần 2 bước Created/Solved
x_waterfall = []
y_waterfall = []
measure_waterfall = []
text_waterfall = []
for i, week in enumerate(week_labels):
    x_waterfall.append(f"{week} Created")
    y_waterfall.append(created_counts[i])
    measure_waterfall.append("relative")
    text_waterfall.append(str(created_counts[i]))
    x_waterfall.append(f"{week} Solved")
    y_waterfall.append(-solved_counts[i])
    measure_waterfall.append("relative")
    text_waterfall.append(str(-solved_counts[i]))
    # Thêm cột trắng (dummy) sau mỗi tuần, trừ tuần cuối
    if i < len(week_labels) - 1:
        x_waterfall.append(f"{week} Spacer")
        y_waterfall.append(0)
        measure_waterfall.append("relative")
        text_waterfall.append("")

# 2. Tạo tickvals và ticktext cho trục x (giữa 2 cột Created/Solved, bỏ qua Spacer)
tickvals = []
ticktext = []
for i, week in enumerate(week_labels):
    # Vị trí giữa 2 cột Created/Solved (bỏ qua Spacer)
    tickvals.append(i*3 + 0.5)
    ticktext.append(week)

fig = go.Figure(go.Waterfall(
    x=x_waterfall,
    y=y_waterfall,
    measure=measure_waterfall,
    text=[f'<span style="background-color:#fff9b1; color:#444; padding:2px 6px; border-radius:4px; font-weight:bold;">{v}</span>' if v not in [None, "", "0"] else "" for v in text_waterfall],
    textposition="outside",
    connector={"line": {"color": "rgba(0,0,0,0.3)"}},
    increasing={"marker": {"color": "#e74c3c", "line": {"width": 0}}},
    decreasing={"marker": {"color": "#27ae60", "line": {"width": 0}}},
    showlegend=False  # Ẩn legend mặc định của Waterfall
))
# Thêm legend thủ công cho Created (đỏ) và Solved (xanh)
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=14, color='#e74c3c'),
    name='Created ticket',
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=14, color='#27ae60'),
    name='Solved ticket',
    showlegend=True
))

fig.update_layout(
    title=dict(
        text="NATIONWIDE ON ASSESSMENT TICKET OVER WEEKS",
        y=0.97,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1000,
    height=700,
    xaxis=dict(
        tickangle=-35,
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        tickvals=tickvals,
        ticktext=ticktext,
    ),
    yaxis=dict(
        showticklabels=False,
        zeroline=False,  
        showgrid=True, 
        gridcolor='#ccc',
        gridwidth=0.1
    ),
    plot_bgcolor='white'
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)



# Lấy danh sách category id và tên (chỉ lấy tên tiếng Anh ngắn gọn)
def extract_short_en_name(val):
    try:
        if isinstance(val, dict):
            return val.get('en_US', str(val))
        if isinstance(val, str) and val.startswith("{'en_US':"):
            import ast
            d = ast.literal_eval(val)
            return d.get('en_US', val)
        return val
    except Exception:
        return val

category_map = df_category.set_index('id')['name'].apply(extract_short_en_name).to_dict()
category_ids = [cid for cid in df['category_id'].unique() if cid in category_map]

teams = df['team_name'].unique()
teams_df = pd.DataFrame({'Team': teams}).sort_values('Team').reset_index(drop=True)

for cid in category_ids:
    cat_name = category_map[cid]
    # Rút gọn tên nếu cần (chỉ lấy phần đầu, loại bỏ phần trong ngoặc)
    short_name = cat_name.split('(')[0].strip() if isinstance(cat_name, str) else cat_name
    # Created
    mask_created = (df['Under this month report'] == 1) & (df['Carry over ticket'] == 0) & (df['category_id'] == cid)
    created = df[mask_created].groupby('team_name').size()
    teams_df[f'{short_name} Newly Created'] = teams_df['Team'].map(created).fillna(0).astype(int)
    # Emergency created
    mask_emergency = mask_created & (df['helpdesk_ticket_tag_id'] == 3)
    emergency_created = df[mask_emergency].groupby('team_name').size()
    teams_df[f'{short_name} Emergency created'] = teams_df['Team'].map(emergency_created).fillna(0).astype(int)
    # Solved
    mask_solved = (df['custom_end_date'] != 'not yet end') & (df['category_id'] == cid)
    df_solved = df[mask_solved].copy()
    df_solved['custom_end_date_dt'] = pd.to_datetime(df_solved['custom_end_date'], errors='coerce')
    mask_time = (df_solved['custom_end_date_dt'] >= start_date) & (df_solved['custom_end_date_dt'] < end_date)
    solved = df_solved[mask_time].groupby('team_name').size()
    teams_df[f'{short_name} Solved'] = teams_df['Team'].map(solved).fillna(0).astype(int)

# Thêm 3 cột tổng: Newly created, Emergency created, Solved
newly_cols = [col for col in teams_df.columns if col.endswith('Newly Created') and not col.endswith('Emergency created')]
emergency_cols = [col for col in teams_df.columns if col.endswith('Emergency created')]
solved_cols = [col for col in teams_df.columns if col.endswith('Solved')]

teams_df['Total Newly created'] = teams_df[newly_cols].sum(axis=1)
teams_df['Total Emergency created'] = teams_df[emergency_cols].sum(axis=1)
teams_df['Total Solved'] = teams_df[solved_cols].sum(axis=1)

# Đưa 3 cột tổng lên đầu bảng
total_cols = ['Total Newly created', 'Total Emergency created', 'Total Solved']
other_cols = [col for col in teams_df.columns if col not in total_cols and col != 'Team']
teams_df = teams_df[['Team'] + total_cols + other_cols]

# Thêm một hàng Total vào cuối bảng
total_row = {}
for col in teams_df.columns:
    if col == 'Team':
        total_row[col] = 'Total'
    else:
        try:
            total_row[col] = teams_df[col].sum()
        except Exception:
            total_row[col] = ''
teams_df = pd.concat([teams_df, pd.DataFrame([total_row])], ignore_index=True)

# Highlight các cột Emergency created nếu value > 0, riêng Total Emergency created: nền vàng, nhưng nếu >0 thì nền đỏ
def highlight_emergency(val):
    try:
        if pd.notnull(val) and float(val) > 0:
            return 'background-color: red; color: black;'
    except Exception:
        pass
    return ''
def highlight_total(val):
    try:
        if pd.notnull(val) and float(val) > 0:
            return 'background-color: red; color: black;'
    except Exception:
        pass
    return 'background-color: #fff9b1; color: black;'
def highlight_total_row(row):
    if row.name == len(teams_df) - 1:
        styles = []
        for col, val in zip(row.index, row):
            if col.endswith('Emergency created') and pd.notnull(val) and float(val) > 0:
                styles.append('background-color: red; color: black;')
            else:
                styles.append('background-color: #fff9b1; color: black;')
        return styles
    return ['' for _ in row]

emergency_cols = [col for col in teams_df.columns if col.endswith('Emergency created') and col != 'Total Emergency created']
total_cols = ['Total Newly created', 'Total Emergency created', 'Total Solved']

styled_df = teams_df.style
styled_df = styled_df.applymap(highlight_emergency, subset=emergency_cols)
styled_df = styled_df.applymap(highlight_total, subset=['Total Emergency created'])
styled_df = styled_df.applymap(lambda v: 'background-color: #fff9b1; color: black;', subset=['Total Newly created', 'Total Solved'])
styled_df = styled_df.apply(highlight_total_row, axis=1)

st.markdown(
    '<h3 style="text-align: center;">HELPDESK TICKET D-1 (LAST WEEKEND IN CASE TODAY IS MONDAY) REPORT <br>(INCLUDING SOLVED CARRY OVER TICKETS)</h3>',
    unsafe_allow_html=True
)

num_rows = teams_df.shape[0]
row_height = 35
total_height = (num_rows + 1) * row_height
st.dataframe(styled_df, use_container_width=True, height=total_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)


# Bảng mới bên dưới
teams_df2 = pd.DataFrame({'Team': teams})


# Thêm cột "ACMV Total ticket" (category_id = 1)
teams_df2['ACMV Total ticket'] = teams_df2['Team'].apply(
    lambda team: df[(df['team_name'] == team) & (df['category_id'] == 1)].shape[0]
)

# Thêm cột "D-1 ticket (or wkn)" lấy từ cột "ACMV Newly Created" của bảng teams_df
if 'ACMV Newly Created' in teams_df.columns:
    teams_df2['ACMV D-1 ticket (or wkn)'] = teams_df2['Team'].map(teams_df.set_index('Team')['ACMV Newly Created']).fillna(0).astype(int)
else:
    # Nếu tên cột là "ACMV Newly Created" chưa đúng, dò tìm cột đúng theo category_id=1
    acmv_col = None
    for col in teams_df.columns:
        if col.lower().startswith('acmv') and 'newly' in col.lower():
            acmv_col = col
            break
    if acmv_col:
        teams_df2['ACMV D-1 ticket (or wkn)'] = teams_df2['Team'].map(teams_df.set_index('Team')[acmv_col]).fillna(0).astype(int)
    else:
        teams_df2['ACMV D-1 ticket (or wkn)'] = 0

# Thêm cột "ACMV OA" (category_id = 1, custom_end_date = 'not yet end')
teams_df2['ACMV OA'] = teams_df2['Team'].apply(
    lambda team: df[(df['team_name'] == team) & (df['category_id'] == 1) & (df['custom_end_date'] == 'not yet end')].shape[0]
)

# Thêm cột "ACMV OA%" (tỷ lệ ACMV OA trên tổng ACMV Total ticket, làm tròn lên số nguyên)
def calc_oa_percent(row):
    total = row['ACMV Total ticket']
    oa = row['ACMV OA']
    if total == 0:
        return 'NA'
    else:
        percent = oa / total * 100
        return f"{math.ceil(percent):.0f}%"
teams_df2['ACMV OA%'] = teams_df2.apply(calc_oa_percent, axis=1)

def render_data_bar(val):
    if val == 'NA':
        return 'NA'
    try:
        percent = int(val.replace('%', ''))
        bar = f'''
            <div style="background: linear-gradient(90deg, #ffe082 {percent}%, #fff {percent}%); 
                        border-radius: 4px; 
                        width: 100%; 
                        height: 22px; 
                        display: flex; 
                        align-items: center; 
                        padding-left: 6px;
                        font-weight: bold;">
                {val}
            </div>
        '''
        return bar
    except:
        return val


# Thêm cột "ACMV Emerg OA" (category_id = 1, custom_end_date = 'not yet end', helpdesk_ticket_tag_id = 3)
teams_df2['ACMV Emerg OA'] = teams_df2['Team'].apply(
    lambda team: df[(df['team_name'] == team) & (df['category_id'] == 1) & (df['custom_end_date'] == 'not yet end') & (df['helpdesk_ticket_tag_id'] == 3)].shape[0]
)

# Thêm cột "ACMV SLA Late OA" (category_id = 1, custom_end_date = 'not yet end', sla_reached_late = True)
teams_df2['ACMV SLA Late OA'] = teams_df2['Team'].apply(
    lambda team: df[(df['team_name'] == team) & (df['category_id'] == 1) & (df['custom_end_date'] == 'not yet end') & (df['sla_reached_late'] == True)].shape[0]
)

# --- TỰ ĐỘNG TẠO 6 CỘT CHO TẤT CẢ CATEGORY (trừ 3,7,9) ---
category_ids_exclude = [3, 7, 9]
teams = df['team_name'].unique()
teams_df2 = pd.DataFrame({'Team': teams})
for cid, cat_name in category_map.items():
    if cid in category_ids_exclude:
        continue
    short_name = cat_name.split('(')[0].strip() if isinstance(cat_name, str) else str(cat_name)
    # 1. Total ticket
    col_total = f'{short_name} Total ticket'
    teams_df2[col_total] = teams_df2['Team'].apply(
        lambda team: df[(df['team_name'] == team) & (df['category_id'] == cid)].shape[0]
    )
    # 2. D-1 ticket (or wkn)
    col_d1 = f'{short_name} D-1 ticket (or wkn)'
    mask_created = (df['Under this month report'] == 1) & (df['Carry over ticket'] == 0) & (df['category_id'] == cid)
    created = df[mask_created].groupby('team_name').size()
    teams_df2[col_d1] = teams_df2['Team'].map(created).fillna(0).astype(int)
    # 3. OA
    col_oa = f'{short_name} OA'
    teams_df2[col_oa] = teams_df2['Team'].apply(
        lambda team: df[(df['team_name'] == team) & (df['category_id'] == cid) & (df['custom_end_date'] == 'not yet end')].shape[0]
    )
    # 4. OA%
    col_oa_percent = f'{short_name} OA%'
    def calc_oa_percent_cat(row):
        total = row[col_total]
        oa = row[col_oa]
        if total == 0:
            return 'NA'
        else:
            percent = oa / total * 100
            return f"{round(percent):.0f}%"
    teams_df2[col_oa_percent] = teams_df2.apply(calc_oa_percent_cat, axis=1)
    # 5. Emerg OA
    col_emerg = f'{short_name} Emerg OA'
    teams_df2[col_emerg] = teams_df2['Team'].apply(
        lambda team: df[(df['team_name'] == team) & (df['category_id'] == cid) & (df['custom_end_date'] == 'not yet end') & (df['helpdesk_ticket_tag_id'] == 3)].shape[0]
    )
    # 6. SLA Late OA
    col_sla = f'{short_name} SLA Late OA'
    teams_df2[col_sla] = teams_df2['Team'].apply(
        lambda team: df[(df['team_name'] == team) & (df['category_id'] == cid) & (df['custom_end_date'] == 'not yet end') & (df['sla_reached_late'] == True)].shape[0]
    )



# Thêm cột 'Total Ticket' là tổng tất cả các cột '[category] Total ticket' trên từng dòng
total_ticket_cols = [col for col in teams_df2.columns if col.endswith('Total ticket')]
teams_df2['Total Ticket'] = teams_df2[total_ticket_cols].sum(axis=1)
# Đưa cột 'Total Ticket' lên đầu (sau cột Team)
cols = list(teams_df2.columns)
cols.insert(1, cols.pop(cols.index('Total Ticket')))
teams_df2 = teams_df2[cols]


# Thêm cột 'Total D-1 ticket (or wkn)' là tổng tất cả các cột '[category] D-1 ticket (or wkn)' trên từng dòng
d1_ticket_cols = [col for col in teams_df2.columns if col.endswith('D-1 ticket (or wkn)')]
teams_df2['Total D-1 ticket (or wkn)'] = teams_df2[d1_ticket_cols].sum(axis=1)
# Đưa cột này lên vị trí thứ 2 (sau 'Total Ticket')
cols = list(teams_df2.columns)
cols.insert(2, cols.pop(cols.index('Total D-1 ticket (or wkn)')))
teams_df2 = teams_df2[cols]

# Thêm cột 'Total OA' là tổng tất cả các cột '[category] OA' trên từng dòng
oa_cols = [col for col in teams_df2.columns if col.endswith(' OA')]
teams_df2['Total OA'] = teams_df2[oa_cols].sum(axis=1)
# Đưa cột này lên vị trí thứ 3 (sau 'Total D-1 ticket (or wkn)')
cols = list(teams_df2.columns)
cols.insert(3, cols.pop(cols.index('Total OA')))
teams_df2 = teams_df2[cols]

# Thêm cột 'Total OA%' = Total OA / Total Ticket, làm tròn số nguyên gần nhất (round), nếu mẫu số = 0 thì trả về 'NA'
def calc_total_oa_percent(row):
    total = row['Total Ticket']
    oa = row['Total OA']
    if total == 0:
        return 'NA'
    else:
        percent = oa / total * 100
        return f"{round(percent):.0f}%"
teams_df2['Total OA%'] = teams_df2.apply(calc_total_oa_percent, axis=1)
# Đưa cột này lên vị trí thứ 4 (sau 'Total OA')
cols = list(teams_df2.columns)
cols.insert(4, cols.pop(cols.index('Total OA%')))
teams_df2 = teams_df2[cols]

# Thêm cột 'Total Emerg OA' là tổng tất cả các cột '[category] Emerg OA' trên từng dòng
emerg_oa_cols = [col for col in teams_df2.columns if col.endswith('Emerg OA')]
teams_df2['Total Emerg OA'] = teams_df2[emerg_oa_cols].sum(axis=1)
# Đưa cột này lên vị trí thứ 5 (sau 'Total OA%')
cols = list(teams_df2.columns)
cols.insert(5, cols.pop(cols.index('Total Emerg OA')))
teams_df2 = teams_df2[cols]

# Thêm cột 'Total SLA Late OA' là tổng tất cả các cột '[category] SLA Late OA' trên từng dòng
sla_late_oa_cols = [col for col in teams_df2.columns if col.endswith('SLA Late OA')]
teams_df2['Total SLA Late OA'] = teams_df2[sla_late_oa_cols].sum(axis=1)
# Đưa cột này lên vị trí thứ 6 (sau 'Total Emerg OA')
cols = list(teams_df2.columns)
cols.insert(6, cols.pop(cols.index('Total SLA Late OA')))
teams_df2 = teams_df2[cols]

# Tô màu vàng nhạt cho 6 cột tổng đầu bảng khi hiển thị
def highlight_total_cols(s):
    if s.name in ['Total Ticket', 'Total D-1 ticket (or wkn)', 'Total OA', 'Total OA%', 'Total Emerg OA', 'Total SLA Late OA']:
        return ['background-color: #fff9b1'] * len(s)
    return [''] * len(s)

# Đảm bảo không có hàng Grand Total cũ bị lặp lại
teams_df2 = teams_df2[teams_df2['Team'] != 'Grand Total']

# Thêm một hàng Grand Total cuối bảng
grand_total = {}
for col in teams_df2.columns:
    if col == 'Team':
        grand_total[col] = 'Grand Total'
    elif teams_df2[col].dtype.kind in 'biufc':  # chỉ cộng các cột số
        grand_total[col] = teams_df2[col].sum()
    else:
        grand_total[col] = ''
# Tính lại Total OA% cho Grand Total
try:
    total_oa = grand_total.get('Total OA', 0)
    total_ticket = grand_total.get('Total Ticket', 0)
    if total_ticket == 0:
        grand_total['Total OA%'] = 'NA'
    else:
        percent = total_oa / total_ticket * 100
        grand_total['Total OA%'] = f"{round(percent):.0f}%"
except Exception:
    grand_total['Total OA%'] = 'NA'
teams_df2 = pd.concat([teams_df2, pd.DataFrame([grand_total])], ignore_index=True)

# Tô màu vàng nhạt cho 6 cột tổng đầu bảng và toàn bộ hàng Grand Total
def highlight_grand_total(row):
    if row.name == len(teams_df2) - 1:
        return ['background-color: #fff9b1'] * len(row)
    return [''] * len(row)

# Sort lại bảng theo tên team trước khi thêm Grand Total
teams_df2 = teams_df2.sort_values('Team').reset_index(drop=True)
# Đảm bảo không có hàng Grand Total cũ bị lặp lại
teams_df2 = teams_df2[teams_df2['Team'] != 'Grand Total']

# Thêm cột phụ is_grand_total
teams_df2['is_grand_total'] = 0

# Thêm một hàng Grand Total cuối bảng
grand_total = {}
for col in teams_df2.columns:
    if col == 'Team':
        grand_total[col] = 'Grand Total'
    elif col == 'is_grand_total':
        grand_total[col] = 1
    elif teams_df2[col].dtype.kind in 'biufc':  # chỉ cộng các cột số
        grand_total[col] = teams_df2[col].sum()
    else:
        grand_total[col] = ''
# Tính lại Total OA% cho Grand Total
try:
    total_oa = grand_total.get('Total OA', 0)
    total_ticket = grand_total.get('Total Ticket', 0)
    if total_ticket == 0:
        grand_total['Total OA%'] = 'NA'
    else:
        percent = total_oa / total_ticket * 100
        grand_total['Total OA%'] = f"{round(percent):.0f}%"
except Exception:
    grand_total['Total OA%'] = 'NA'
teams_df2 = pd.concat([teams_df2, pd.DataFrame([grand_total])], ignore_index=True)

# Luôn sort lại theo is_grand_total để Grand Total ở cuối
teams_df2 = teams_df2.sort_values('is_grand_total').reset_index(drop=True)

# Khi hiển thị, ẩn cột is_grand_total và mở rộng bảng ra toàn bộ chiều rộng, loại bỏ thanh cuộn
num_rows = teams_df2.shape[0]
row_height = 38  # hoặc 35 tuỳ font, có thể chỉnh lại nếu cần

st.markdown(
    '<h3 style="text-align: center;">HELPDESK TICKETS ALL TIME</h3>',
    unsafe_allow_html=True
)

st.dataframe(
    teams_df2.drop(columns=['is_grand_total']).style.apply(highlight_total_cols, axis=0).apply(highlight_grand_total, axis=1),
    use_container_width=True,
    height=num_rows * row_height + 38  # +38 cho header
)
st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)


# --- Bảng trung bình processing_time theo team và category ---
st.markdown(
    '<h3 style="text-align: center;">AVERAGE TICKET PROCESSING SPEED OF ALL TIME<br>(DAYS UP UNTIL "APPROVED" STAGE OF TICKET)</h3>',
    
    unsafe_allow_html=True
)
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
# Thêm cột Across all category
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
# Làm tròn cho đẹp
pivot = pivot.round(0).astype(int)
# Conditional formatting: xanh nhạt (min), trắng (giữa), đỏ đậm (#ff4d4d)
import matplotlib
def color_scale(val, vmin, vmax):
    norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-77)*(norm-0.5)*2)
        b = int(255 - (255-77)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'
vmin = pivot.min().min()
vmax = pivot.max().max()
styled = pivot.style.applymap(lambda v: color_scale(v, vmin, vmax)).set_properties(**{'text-align': 'center', 'color': 'black'})
num_rows = len(pivot.index)
row_height = 35
total_height = (num_rows + 1) * row_height


# --- Thêm hàng 'Avg. across all' vào bảng pivot ---
# Lấy lại bảng teams_df2 (bảng tổng hợp ticket theo team/category)
teams_df2_no_total = teams_df2[teams_df2['Team'] != 'Grand Total'].drop_duplicates(subset=['Team']).set_index('Team')
team_order = list(pivot.index)
# Cột 'Across all category'
total_ticket_team = teams_df2_no_total.loc[team_order, 'Total Ticket'] if 'Total Ticket' in teams_df2_no_total.columns else pd.Series(1, index=team_order)
total_ticket_all = total_ticket_team.sum()
avg_across_all = (pivot['Across all category'] * total_ticket_team).sum() / total_ticket_all if total_ticket_all > 0 else 0
avg_row = {'Across all category': round(avg_across_all)}
# Các cột category
for cat in pivot.columns:
    if cat == 'Across all category':
        continue
    # Lấy phần tên trước dấu '(' nếu có
    cat_short = cat.split('(')[0].strip()
    cat_total_col = None
    for c in teams_df2_no_total.columns:
        if c.endswith('Total ticket'):
            c_short = c.replace('Total ticket', '').strip()
            if c_short == cat_short:
                cat_total_col = c
                break
    if not cat_total_col:
        # Nếu không tìm thấy, thử match gần đúng
        for c in teams_df2_no_total.columns:
            if c.endswith('Total ticket') and cat_short.lower() in c.lower():
                cat_total_col = c
                break
    if cat_total_col:
        cat_ticket_team = teams_df2_no_total.loc[team_order, cat_total_col]
        cat_ticket_all = cat_ticket_team.sum()
        if cat_ticket_all > 0:
            avg_val = (pivot[cat] * cat_ticket_team).sum() / cat_ticket_all
            avg_row[cat] = round(avg_val)
        else:
            avg_row[cat] = 0
    else:
        avg_row[cat] = 0
# Thêm hàng vào pivot
pivot.loc['Avg. across all'] = avg_row
num_rows = len(pivot.index)
total_height = (num_rows + 1) * row_height
styled = pivot.style.applymap(lambda v: color_scale(v, vmin, vmax)).set_properties(**{'text-align': 'center', 'color': 'black'})
st.dataframe(styled, use_container_width=True, height=total_height)
st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)


import plotly.graph_objects as go
# 1. Lấy các cột Newly Created (không lấy Emergency)
newly_cols = [col for col in teams_df.columns if col.endswith('Newly Created') and 'Emergency' not in col]
category_names = [col.replace(' Newly Created', '') for col in newly_cols]

# 2. Loại bỏ hàng Total
df_bar = teams_df[teams_df['Team'] != 'Total']

# 3. Tạo clustered (grouped) bar chart
fig = go.Figure()
team_labels = list(df_bar['Team'])
team_indices = list(range(len(team_labels)))

# Định nghĩa màu cho các category, tránh trùng màu đỏ
category_colors = [
    '#1f77b4',  # xanh dương
    '#ff7f0e',  # cam
    '#2ca02c',  # xanh lá
    '#9467bd',  # tím
    '#8c564b',  # nâu
    '#e377c2',  # hồng
    '#7f7f7f',  # xám
    '#bcbd22',  # vàng xanh
    '#17becf',  # xanh ngọc
]

# Vẽ bar Emergency màu đỏ đứng đầu tiên
if 'Total Emergency created' in teams_df.columns:
    emergency_vals = df_bar['Total Emergency created'] if 'Total Emergency created' in df_bar.columns else [0]*len(team_indices)
    text_emergency = [str(v) if v != 0 else '' for v in emergency_vals]
    # Tạo custom texttemplate với background vàng nhạt và chữ đỏ
    custom_text = [
        f'<span style="background-color:#fff9b1; color:#d62728; padding:2px 6px; border-radius:4px; font-weight:bold;">{v}</span>' if v != '' else ''
        for v in text_emergency
    ]
    fig.add_trace(go.Bar(
        x=team_indices,
        y=emergency_vals,
        name='Total Emergency created',
        text=custom_text,
        textposition='outside',
        offsetgroup='emergency',
        marker_color='#d62728',  # đỏ
        textfont=dict(size=12),
        # Sử dụng texttemplate để giữ HTML style
        texttemplate='%{text}'
    ))

# Vẽ các bar cho từng category (trừ Emergency)
category_colors = [c for c in category_colors if c != '#d62728']
for i, (cat, col) in enumerate(zip(category_names, newly_cols)):
    text_vals = [str(v) if v != 0 else '' for v in df_bar[col]]
    color = category_colors[i % len(category_colors)]
    fig.add_trace(go.Bar(
        x=team_indices,
        y=df_bar[col],
        name=cat,
        text=text_vals,
        textposition='outside',
        offsetgroup=cat,
        marker_color=color,
        textfont=dict(size=12, color='black')
    ))

# Thêm các đường kẻ thẳng để phân chia các team
team_count = len(team_indices)
shapes = []
for i in range(1, team_count):
    shapes.append(dict(
        type='line',
        xref='x', yref='paper',
        x0=i-0.5, x1=i-0.5,
        y0=0, y1=1,
        line=dict(color='gray', width=1, dash='dot')
    ))

fig.update_layout(
    barmode='group',
    bargap=0.25,
    bargroupgap=0.3,
    title={
        'text': "TOTAL CREATED TICKETS PAST 1 DAY BY REGIONS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1000,
    height=700,
    xaxis_tickangle=-70,
    xaxis=dict(
        tickmode='array',
        tickvals=team_indices,
        ticktext=team_labels,
        tickfont=dict(size=12, color='black'),
        showgrid=False
    ),
    yaxis=dict(
        tickfont=dict(size=12, color='black')
    ),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=9,
        xanchor='center',
        x=0.5
    ),
    shapes=shapes
)

st.plotly_chart(fig, use_container_width=True)

# --- BAR CHART OA CỦA TỪNG CATEGORY VÀ TEAM ---
# Lấy các cột OA (chỉ lấy đúng '[category] OA', không lấy SLA Late OA, Emerg OA...)
oa_cols = [col for col in teams_df2.columns if col.endswith(' OA') and not any(x in col for x in ['Emerg', 'SLA Late']) and col != 'Total OA']
oa_category_names = [col.replace(' OA', '') for col in oa_cols]
df_oa = teams_df2[teams_df2['Team'] != 'Grand Total']
teams_oa = list(df_oa['Team'])
teams_oa_indices = list(range(len(teams_oa)))

# Định nghĩa màu cho các category OA, tránh trùng màu đỏ
category_colors_oa = [
    '#1f77b4',  # xanh dương
    '#ff7f0e',  # cam
    '#2ca02c',  # xanh lá
    '#9467bd',  # tím
    '#8c564b',  # nâu
    '#e377c2',  # hồng
    '#7f7f7f',  # xám
    '#bcbd22',  # vàng xanh
    '#17becf',  # xanh ngọc
]

fig_oa = go.Figure()
# Thêm bar Total Emerg OA màu đỏ đứng đầu
if 'Total Emerg OA' in df_oa.columns:
    emerg_oa_vals = df_oa['Total Emerg OA']
    text_emerg_oa = [str(v) if v != 0 else '' for v in emerg_oa_vals]
    fig_oa.add_trace(go.Bar(
        x=teams_oa_indices,
        y=emerg_oa_vals,
        name='Total Emerg OA',
        text=text_emerg_oa,
        textposition='outside',
        offsetgroup='emerg_oa',
        marker_color='#d62728',
        textfont=dict(size=14, color='#d62728'),  # màu đỏ, size lớn hơn
        texttemplate='%{text}'
    ))
st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)


# Vẽ các bar OA cho từng category (trừ Total Emerg OA)
category_colors_oa = [c for c in category_colors_oa if c != '#d62728']
for i, (cat, col) in enumerate(zip(oa_category_names, oa_cols)):
    text_vals = [str(v) if v != 0 else '' for v in df_oa[col]]
    color = category_colors_oa[i % len(category_colors_oa)]
    fig_oa.add_trace(go.Bar(
        x=teams_oa_indices,
        y=df_oa[col],
        name=cat + ' OA',
        text=text_vals,
        textposition='outside',
        offsetgroup=cat,
        marker_color=color,
        textfont=dict(size=12, color='black')
    ))

# Thêm các đường kẻ thẳng để phân chia các team
team_count_oa = len(teams_oa_indices)
shapes_oa = []
for i in range(1, team_count_oa):
    shapes_oa.append(dict(
        type='line',
        xref='x', yref='paper',
        x0=i-0.5, x1=i-0.5,
        y0=0, y1=1,
        line=dict(color='gray', width=1, dash='dot')
    ))

fig_oa.update_layout(
    barmode='group',
    bargap=0.25,
    bargroupgap=0.3,
    title={
        'text': "TOTAL ACCUMULATIVE OA TICKETS UP TO DATE BY REGIONS",
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1200,
    height=800,
    xaxis_tickangle=-70,
    xaxis=dict(
        tickmode='array',
        tickvals=teams_oa_indices,
        ticktext=teams_oa,
        tickfont=dict(size=13, color='black'),
        showgrid=False
    ),
    yaxis=dict(
        tickfont=dict(size=13, color='black')
    ),
    legend=dict(
        orientation='h',
        yanchor='top',
        y=9,
        xanchor='center',
        x=0.5
    ),
    shapes=shapes_oa
)
st.plotly_chart(fig_oa, use_container_width=True)
st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)


# --- LINE CHART DAILY OA TICKETS BY CATEGORY ---
from datetime import timedelta

# Xác định ngày bắt đầu và ngày kết thúc
start_line_date = pd.Timestamp('2025-04-10')
today = pd.Timestamp.now().normalize()
if today.dayofweek == 0:  # Thứ 2
    end_line_date = today - pd.Timedelta(days=3)
else:
    end_line_date = today - pd.Timedelta(days=1)

# Tạo list ngày
date_range = pd.date_range(start=start_line_date, end=end_line_date, freq='D')

# Lấy danh sách category thực tế từ dữ liệu, loại bỏ rỗng
category_names = [c for c in sorted(df['category_name'].unique()) if c and c != 'nan']

def count_ton_tung_ngay(cat, day):
    cat = str(cat)
    mask1 = (df['category_name'] == cat) & (df['custom_end_date'] == 'not yet end')
    count1 = df[mask1].shape[0]
    mask2 = (
        (df['category_name'] == cat) &
        (df['custom_end_date'] != 'not yet end') &
        (pd.to_datetime(df['custom_end_date'], errors='coerce') > day) &
        (df['create_date'] <= day)
    )
    count2 = df[mask2].shape[0]
    return count1 + count2

# Tạo dataframe kết quả
line_data = {'date': date_range}
for cat in category_names:
    line_data[cat] = [count_ton_tung_ngay(cat, d) for d in date_range]
df_line = pd.DataFrame(line_data)

# Vẽ line chart
import plotly.graph_objects as go
fig_line = go.Figure()
color_map = [
    '#00cfff', '#2ca02c', '#222', '#bcbd22', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#17becf', '#d62728', '#7f7f7f', '#ffd700', '#00bfff'
]
for i, cat in enumerate(category_names):
    fig_line.add_trace(go.Scatter(
        x=df_line['date'],
        y=df_line[cat],
        mode='lines',
        name=cat,
        line=dict(width=3, color=color_map[i % len(color_map)])
    ))
fig_line.update_layout(
    title={
        'text': 'DAILY ON ASSESSMENT TICKETS BY CATEGORY',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=800,
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.25,  # Đặt legend xuống dưới chart
        xanchor='center',
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=13, color='black', family='Arial', weight='bold'),
        tickformat='%d/%m/%Y',
        showgrid=False
    ),
    yaxis=dict(
        tickfont=dict(size=14, color='black', family='Arial', weight='bold'),
        showgrid=True
    ),
    margin=dict(l=40, r=40, t=80, b=120)
)
st.plotly_chart(fig_line, use_container_width=True)
st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)


    # --- LINE CHART DAILY OA TICKETS BY PRIORITY ---

# Xác định ngày bắt đầu và kết thúc
start_line_date = pd.Timestamp('2025-04-10')
today = pd.Timestamp.now().normalize()
if today.dayofweek == 0:  # Thứ 2
    end_line_date = today - pd.Timedelta(days=3)
else:
    end_line_date = today - pd.Timedelta(days=1)
date_range = pd.date_range(start=start_line_date, end=end_line_date, freq='D')

priority_names = ['Low priority', 'Medium priority', 'High priority', 'Emergency']

def count_ton_tung_ngay_priority(priority, day):
    if priority == 'Low priority':
        mask1 = (
            (df['helpdesk_ticket_tag_id'] != 3) &
            (
                df['priority'].isna() |
                (df['priority'].astype(str).str.strip() == '0') |
                (df['priority'].fillna(0).astype(int) == 1)
            ) &
            (df['custom_end_date'] == 'not yet end')
        )
        count1 = df[mask1].shape[0]
        mask2 = (
            (df['helpdesk_ticket_tag_id'] != 3) &
            (
                df['priority'].isna() |
                (df['priority'].astype(str).str.strip() == '0') |
                (df['priority'].fillna(0).astype(int) == 1)
            ) &
            (df['custom_end_date'] != 'not yet end') &
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > day) &
            (df['create_date'] <= day)
        )
        count2 = df[mask2].shape[0]
        return count1 + count2

    elif priority == 'Medium priority':
        mask1 = (
            (df['helpdesk_ticket_tag_id'] != 3) &
            (df['priority'].fillna(0).astype(int) == 2) &
            (df['custom_end_date'] == 'not yet end')
        )
        count1 = df[mask1].shape[0]
        mask2 = (
            (df['helpdesk_ticket_tag_id'] != 3) &
            (df['priority'].fillna(0).astype(int) == 2) &
            (df['custom_end_date'] != 'not yet end') &
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > day) &
            (df['create_date'] <= day)
        )
        count2 = df[mask2].shape[0]
        return count1 + count2

    elif priority == 'High priority':
        mask1 = (
            (df['helpdesk_ticket_tag_id'] != 3) &
            (df['priority'].fillna(0).astype(int) == 3) &
            (df['custom_end_date'] == 'not yet end')
        )
        count1 = df[mask1].shape[0]
        mask2 = (
            (df['helpdesk_ticket_tag_id'] != 3) &
            (df['priority'].fillna(0).astype(int) == 3) &
            (df['custom_end_date'] != 'not yet end') &
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > day) &
            (df['create_date'] <= day)
        )
        count2 = df[mask2].shape[0]
        return count1 + count2

    elif priority == 'Emergency':
        mask1 = (
            (df['helpdesk_ticket_tag_id'] == 3) &
            (df['custom_end_date'] == 'not yet end')
        )
        count1 = df[mask1].shape[0]
        mask2 = (
            (df['helpdesk_ticket_tag_id'] == 3) &
            (df['custom_end_date'] != 'not yet end') &
            (pd.to_datetime(df['custom_end_date'], errors='coerce') > day) &
            (df['create_date'] <= day)
        )
        count2 = df[mask2].shape[0]
        return count1 + count2

# Tạo dataframe kết quả
line_data_priority = {'date': date_range}
for pri in priority_names:
    line_data_priority[pri] = [count_ton_tung_ngay_priority(pri, d) for d in date_range]
df_line_priority = pd.DataFrame(line_data_priority)

# Vẽ line chart
fig_line_priority = go.Figure()
priority_colors = {
    'Low priority': '#61ee9c',
    'Medium priority': '#f5f541',
    'High priority': '#f7a31b',
    'Emergency': '#e54125'
}
for pri in priority_names:
    fig_line_priority.add_trace(go.Scatter(
        x=df_line_priority['date'],
        y=df_line_priority[pri],
        mode='lines',
        name=pri,
        line=dict(width=3, color=priority_colors[pri])
    ))
fig_line_priority.update_layout(
    title={
        'text': 'DAILY ON ASSESSMENT TICKETS BY PRIORITY',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=800,
    legend=dict(
        orientation='h',
        yanchor='top',
        y=7,
        xanchor='center',
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        tickangle=0,
        tickfont=dict(size=13, color='black', family='Arial', weight='bold'),
        tickformat='%d/%m/%Y',
        showgrid=False
    ),
    yaxis=dict(
        tickfont=dict(size=14, color='black', family='Arial', weight='bold'),
        showgrid=True
    ),
    margin=dict(l=40, r=40, t=80, b=120)
)
st.plotly_chart(fig_line_priority, use_container_width=True)
st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# -------------------------------NORTH 1------------------------------------------------------

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>NORTH 1 - Nguyen Van Khuong</h2>", unsafe_allow_html=True)
df_north1 = df[df['team_id'] == 17]  # team_id = 17 cho North 1
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Tạo lại pivot nếu chưa có
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho North 1
value = pivot.loc['NORTH 1 - Nguyen Van Khuong', 'Across all category']

gauge_max = 100
gauge_min = 0
value = pivot.loc['NORTH 1 - Nguyen Van Khuong', 'Across all category']

# Xác định các mức fill
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
# Phần còn lại là màu xám nhạt
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})


fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    # Không đặt title ở đây để tránh bị cắt
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

    # Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W) cho North 1
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

# W-1
mask_w1 = (
    (df_north1['create_date'] <= end_w1) &
    (
        (df_north1['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_north1[mask_w1].shape[0]

# W
mask_w = (
    (df_north1['create_date'] <= end_w) &
    (
        (df_north1['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_north1[mask_w].shape[0]

    # Tính % thay đổi
if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

# Hiển thị box
if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

# Tách phần trăm và text thành 2 dòng
percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

# Hiển thị cạnh box so sánh
col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    # Chỉnh thông số tại đây, ví dụ: margin-left: 80px
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)


# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_north1[(df_north1['create_date'] >= start) & (df_north1['create_date'] <= end)].shape[0]
    solved = -df_north1[(pd.to_datetime(df_north1['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',

    title={
        'text': "NORTH 1 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },

    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1000,
    height=700,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)


# (Các phần biểu đồ khác, bảng site, ... giữ nguyên như cũ, đặt sau các phần trên)


    # Stacked Bar Chart theo Category cho North 1
category_names_north1 = df_north1['category_name'].dropna().unique()
table_data_north1 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_north1:
        mask = (
            (df_north1['category_name'] == cat) &
            (df_north1['create_date'] <= end) &
            (
                (df_north1['custom_end_date'] == "not yet end") |
                (
                    (df_north1['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_north1[mask].shape[0]
        row[cat] = count
    table_data_north1.append(row)
df_table_north1 = pd.DataFrame(table_data_north1)

fig_stack_north1 = go.Figure()
for cat in category_names_north1:
    y_values = df_table_north1[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_north1.add_trace(go.Bar(
        name=cat,
        x=df_table_north1["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_north1[category_names_north1].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_north1.add_trace(go.Scatter(
    x=df_table_north1["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_north1["Tuần"], totals_offset, totals)):
    fig_stack_north1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# Tính % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1  # W24
idx_w1 = idx_w - 1          # W23
end_w = week_ends[idx_w]    # 15.6.2025
end_w1 = week_ends[idx_w1]  # 8.6.2025
w_label = df_table_north1["Tuần"].iloc[idx_w]  # Label của W24

# Lọc ra các category có data trong W24
active_categories = []
percent_changes = {}
category_positions = {}  # Lưu vị trí y của mỗi category trong cột stacked

# Tính tổng chiều cao tích lũy và lưu vị trí của mỗi category
cumulative_height = 0
for cat in category_names_north1:
    count_w24 = float(df_table_north1[cat].iloc[idx_w])  # Chuyển đổi sang float để so sánh chính xác
    if count_w24 <= 0:  # Bỏ qua các category không có data
        continue

    # Tính % thay đổi
    count_w23 = float(df_table_north1[cat].iloc[idx_w1])
    if count_w23 == 0:
        percent = 100 if count_w24 > 0 else 0
    else:
        percent = ((count_w24 - count_w23) / count_w23) * 100

    # Chỉ thêm vào danh sách nếu có data thực sự
    active_categories.append(cat)
    percent_changes[cat] = percent
    # Lưu vị trí giữa của category trong cột stacked
    category_positions[cat] = cumulative_height + count_w24 / 2
    cumulative_height += count_w24

if active_categories:
    # Tính toán vị trí cho các box
    total_height = cumulative_height
    x_vals = list(df_table_north1["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2  # Tăng khoảng cách theo chiều ngang

    # Sắp xếp các category theo thứ tự từ dưới lên trên trong cột stacked
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"  # cam nhạt
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"  # xanh nhạt
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"  # cam nhạt

        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))

        # Đường ngang từ cột ra ngoài
        fig_stack_north1.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        # Box chứa %
        fig_stack_north1.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_north1.update_layout(
    barmode='stack',
    title=dict(
        text="NORTH 1 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_north1)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)



    # Stacked Bar Chart theo Priority cho North 1
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_north1 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_north1['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_north1['priority'].isna()) |
            (df_north1['priority'].astype(str).str.strip() == '0') |
            (df_north1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north1['create_date'] <= end) &
        (
            (df_north1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_north1[mask_low].shape[0]
    mask_medium = (
        (df_north1['helpdesk_ticket_tag_id'] != 3) &
        (df_north1['priority'].fillna(0).astype(int) == 2) &
        (df_north1['create_date'] <= end) &
        (
            (df_north1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_north1[mask_medium].shape[0]
    mask_high = (
        (df_north1['helpdesk_ticket_tag_id'] != 3) &
        (df_north1['priority'].fillna(0).astype(int) == 3) &
        (df_north1['create_date'] <= end) &
        (
            (df_north1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_north1[mask_high].shape[0]
    mask_emergency = (
        (df_north1['helpdesk_ticket_tag_id'] == 3) &
        (df_north1['create_date'] <= end) &
        (
            (df_north1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_north1[mask_emergency].shape[0]
    table_data_priority_north1.append(row)
df_table_priority_north1 = pd.DataFrame(table_data_priority_north1)

# Tính % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_north1["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_north1[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_north1[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_north1 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_north1[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_north1.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_north1["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_north1["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_north1.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_north1.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_north1[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_north1.add_trace(go.Scatter(
    x=df_table_priority_north1["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_north1["Tuần"], totals_offset, totals)):
    fig_stack_priority_north1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_north1.update_layout(
    barmode='stack',
    title={
        'text': "NORTH 1 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_north1)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)


# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)

created_counts = []
solved_counts = []

for start, end in zip(week_starts, week_ends):
    # Created ticket: High priority & Non-Emergency
    created_high_non_emergency = df_north1[
        (df_north1['create_date'] >= start) &
        (df_north1['create_date'] <= end) &
        (df_north1['priority'].fillna(0).astype(int) == 3) &
        (df_north1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]

    # Created ticket: High priority & Emergency
    created_high_emergency = df_north1[
        (df_north1['create_date'] >= start) &
        (df_north1['create_date'] <= end) &
        (df_north1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]

    created = created_high_non_emergency + created_high_emergency

    # Solved ticket: High priority & Non-Emergency
    solved_high_non_emergency = df_north1[
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') <= end) &
        (df_north1['priority'].fillna(0).astype(int) == 3) &
        (df_north1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]

    # Solved ticket: High priority & Emergency
    solved_high_emergency = df_north1[
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') <= end) &
        (df_north1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]

    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',

    title={
        'text': "NORTH 1 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1000,
    height=700,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)



# Clustered Chart: Created/Solved ticket Low & Medium Priority 

created_counts = []
solved_counts = []

for start, end in zip(week_starts, week_ends):
    # Created ticket: Low priority & Medium
    created_low = df_north1[
        (df_north1['create_date'] >= start) &
        (df_north1['create_date'] <= end) &
        (
            df_north1['priority'].isna() |
            (df_north1['priority'].astype(str).str.strip() == '0') |
            (df_north1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]

    # Created ticket: Medium priority
    created_medium = df_north1[
        (df_north1['create_date'] >= start) &
        (df_north1['create_date'] <= end) &
        (df_north1['priority'].fillna(0).astype(int) == 2)
            &
        (df_north1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]

    created = created_low + created_medium

    # Solved ticket: Low priority & Non-Emergency
    solved_low = df_north1[
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') <= end) &
        (
            df_north1['priority'].isna() |
            (df_north1['priority'].astype(str).str.strip() == '0') |
            (df_north1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]

    # Solved ticket: Medium priority
    solved_medium = df_north1[
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') <= end) &
        (df_north1['priority'].fillna(0).astype(int) == 2)
            &
        (df_north1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]

    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',

    title={
        'text': "NORTH 1 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },

    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho North 1 (cột 2: số ticket chưa end)

st.markdown("<h3 style='text-align: center;'>NORTH 1 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    "CBS Crocs Aeon Hai Phong (2763F6)", "CBS Crocs Nguyen Duc Canh Hai Phong (2763P7)","CBS Crocs Vincom Ha Long (276391)","CBS Dyson Aeon Hai Phong (2763CJ)","CBS Fila Aeon Hai Phong (2763F7)","GO Mall Hai Duong (HDG)",
    "GO Mall Hai Phong (HPG)","GO Mall Ha Long (HLG)","GO Mall Nam Dinh (NDH)","GO Mall Thai Binh (TBH)",
    "Hyper Hai Duong (HDG)","Hyper Hai Phong (HPG)","Hyper Ha Long (HLG)","Hyper Ha Nam (HNM)","Hyper Nam Dinh (NDH)",
    "Hyper Thai Binh (TBH)","KUBO NANO Hai Duong (6445)","KUBO NANO Hai Phong (6422)","KUBO NANO Ha Long (6423)",
    "KUBO NANO Ha Nam (6451)","KUBO NANO Nam Dinh (6409)","KUBO NANO Thai Binh (6438)","Nguyen Kim Hai Phong (HP01)",
    "Nguyen Kim Nam Dinh (ND01)","KUBO NANO Hung Yen (6427)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})




today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

# Khởi tạo các list lưu kết quả
site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

# Lấy danh sách 11 category cần thống kê
category_list = df_north1['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_north1 = df_special_sites['Sites'].tolist()

for site in sites_north1:
    # Tổng số ticket chưa end
    count_not_end = df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    # CỘT 3 - 7 ngày
    count_old_not_end_7 = df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['create_date'] <= seven_days_ago) &
        (df_north1['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['create_date'] <= seven_days_ago) &
        (df_north1['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    # CỘT 4 - 70 ngày
    count_old_not_end_70 = df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['create_date'] <= seventy_days_ago) &
        (df_north1['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['create_date'] <= seventy_days_ago) &
        (df_north1['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_north1['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    # Emergency chưa end
    site_ticket_emergency.append(df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['custom_end_date'] == "not yet end") &
        (df_north1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    # High priority chưa end, không phải emergency
    site_ticket_high_priority.append(df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['custom_end_date'] == "not yet end") &
        (df_north1['helpdesk_ticket_tag_id'] != 3) &
        (df_north1['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    # Medium priority chưa end, không phải emergency
    site_ticket_medium_priority.append(df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['custom_end_date'] == "not yet end") &
        (df_north1['helpdesk_ticket_tag_id'] != 3) &
        (df_north1['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    # Low priority chưa end, không phải emergency
    site_ticket_low_priority.append(df_north1[
        (df_north1['mall_display_name'] == site) &
        (df_north1['custom_end_date'] == "not yet end") &
        (df_north1['helpdesk_ticket_tag_id'] != 3) &
        (
            df_north1['priority'].isna() |
            (df_north1['priority'].astype(str).str.strip() == '0') |
            (df_north1['priority'].fillna(0).astype(int) == 0) |
            (df_north1['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    # Category columns
    for cat in category_list:
        site_ticket_by_category[cat].append(df_north1[
            (df_north1['mall_display_name'] == site) &
            (df_north1['custom_end_date'] == "not yet end") &
            (df_north1['category_name'] == cat)
        ].shape[0])

# Tạo DataFrame
data = {
    'Sites': sites_north1,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_north1 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_north1[col].sum() if df_sites_north1[col].dtype != 'O' else 'TOTAL' for col in df_sites_north1.columns}
df_sites_north1 = pd.concat([df_sites_north1, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_north1.columns if col != 'Sites']
df_no_total = df_sites_north1.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_north1) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_north1.style.apply(lambda s: apply_color_scale(df_sites_north1), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_north1) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_north1.shape[0]
row_height = 35  # hoặc 32, tuỳ font
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>NORTH 2 - Vu Ngoc Hieu</h2>", unsafe_allow_html=True)
df_north2 = df[df['team_id'] == 2]  # team_id = 2 cho North 2
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Tạo lại pivot nếu chưa có
pivot2 = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all2 = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot2.insert(0, 'Across all category', across_all2)
pivot2 = pivot2.round(0).astype(int)

# Lấy giá trị cho North 2
value2 = pivot2.loc['NORTH 2 - Vu Ngoc Hieu', 'Across all category']

gauge_max2 = 100
gauge_min2 = 0

# Xác định các mức fill
level1_2 = 33
level2_2 = 66

steps2 = []
if value2 > 0:
    steps2.append({'range': [0, min(value2, level1_2)], 'color': '#b7f7b7'})
if value2 > level1_2:
    steps2.append({'range': [level1_2, min(value2, level2_2)], 'color': '#ffe082'})
if value2 > level2_2:
    steps2.append({'range': [level2_2, min(value2, gauge_max2)], 'color': '#ffb3b3'})
if value2 < gauge_max2:
    steps2.append({'range': [value2, gauge_max2], 'color': '#eeeeee'})

fig_gauge2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value2,
    gauge={
        'axis': {'range': [gauge_min2, gauge_max2]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps2,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge2.update_layout(
    annotations=[
        dict(
            x=0.5, y=0,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W) cho North 2
idx_w2 = len(week_ends) - 1
idx_w1_2 = idx_w2 - 1
end_w2 = week_ends[idx_w2]
end_w1_2 = week_ends[idx_w1_2]

# W-1
mask_w1_2 = (
    (df_north2['create_date'] <= end_w1_2) &
    (
        (df_north2['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end_w1_2)
    )
)
count_w1_2 = df_north2[mask_w1_2].shape[0]

# W
mask_w2 = (
    (df_north2['create_date'] <= end_w2) &
    (
        (df_north2['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end_w2)
    )
)
count_w2 = df_north2[mask_w2].shape[0]

# Tính % thay đổi
if count_w1_2 == 0:
    percent2 = 100 if count_w2 > 0 else 0
else:
    percent2 = ((count_w2 - count_w1_2) / count_w1_2) * 100

if percent2 > 0:
    percent_text2 = f"W vs W-1: +{percent2:.1f}%"
    bgcolor2 = "#f2c795"
elif percent2 < 0:
    percent_text2 = f"W vs W-1: -{abs(percent2):.1f}%"
    bgcolor2 = "#abf3ab"
else:
    percent_text2 = "W vs W-1: 0.0%"
    bgcolor2 = "#f2c795"

percent_value2 = f"{percent2:+.1f}%" if percent2 != 0 else "0.0%"

col1_2, col2_2 = st.columns([1, 0.9])
with col1_2:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor2}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value2}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2_2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge2)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts2 = []
solved_counts2 = []
for start, end in zip(week_starts, week_ends):
    created = df_north2[(df_north2['create_date'] >= start) & (df_north2['create_date'] <= end)].shape[0]
    solved = -df_north2[(pd.to_datetime(df_north2['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts2.append(created)
    solved_counts2.append(solved)

fig2 = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts2,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts2],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts2,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts2],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig2.update_layout(
    barmode='group',
    title={
        'text': "NORTH 2 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig2, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# ...phần đầu giữ nguyên...

# --- Stacked Bar Chart theo Category cho North 2 ---
category_names_north2 = df_north2['category_name'].dropna().unique()
table_data_north2 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_north2:
        mask = (
            (df_north2['category_name'] == cat) &
            (df_north2['create_date'] <= end) &
            (
                (df_north2['custom_end_date'] == "not yet end") |
                (
                    (df_north2['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_north2[mask].shape[0]
        row[cat] = count
    table_data_north2.append(row)
df_table_north2 = pd.DataFrame(table_data_north2)

fig_stack_north2 = go.Figure()
for cat in category_names_north2:
    y_values = df_table_north2[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_north2.add_trace(go.Bar(
        name=cat,
        x=df_table_north2["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals2 = df_table_north2[category_names_north2].sum(axis=1)
totals_offset2 = totals2 + totals2 * 0.04
fig_stack_north2.add_trace(go.Scatter(
    x=df_table_north2["Tuần"],
    y=totals_offset2,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_north2["Tuần"], totals_offset2, totals2)):
    fig_stack_north2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w2 = len(week_ends) - 1
idx_w1_2 = idx_w2 - 1
end_w2 = week_ends[idx_w2]
end_w1_2 = week_ends[idx_w1_2]
w_label2 = df_table_north2["Tuần"].iloc[idx_w2]

active_categories2 = []
percent_changes2 = {}
category_positions2 = {}
cumulative_height2 = 0
for cat in category_names_north2:
    count_w2 = float(df_table_north2[cat].iloc[idx_w2])
    if count_w2 <= 0:
        continue
    count_w1_2 = float(df_table_north2[cat].iloc[idx_w1_2])
    if count_w1_2 == 0:
        percent = 100 if count_w2 > 0 else 0
    else:
        percent = ((count_w2 - count_w1_2) / count_w1_2) * 100
    active_categories2.append(cat)
    percent_changes2[cat] = percent
    category_positions2[cat] = cumulative_height2 + count_w2 / 2
    cumulative_height2 += count_w2

if active_categories2:
    total_height2 = cumulative_height2
    x_vals2 = list(df_table_north2["Tuần"])
    x_idx2 = x_vals2.index(w_label2)
    x_offset2 = x_idx2 + 2
    sorted_categories2 = sorted(active_categories2, key=lambda x: category_positions2[x])
    for i, cat in enumerate(sorted_categories2):
        percent = percent_changes2[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions2[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height2 * spacing_factor * (i - len(sorted_categories2)/2))
        fig_stack_north2.add_annotation(
            x=w_label2, y=y_col,
            ax=x_offset2, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_north2.add_annotation(
            x=x_offset2, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_north2.update_layout(
    barmode='stack',
    title=dict(
        text="NORTH 2 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_north2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Stacked Bar Chart theo Priority cho North 2 ---
priority_cols2 = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors2 = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_north2 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_north2['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_north2['priority'].isna()) |
            (df_north2['priority'].astype(str).str.strip() == '0') |
            (df_north2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north2['create_date'] <= end) &
        (
            (df_north2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_north2[mask_low].shape[0]
    mask_medium = (
        (df_north2['helpdesk_ticket_tag_id'] != 3) &
        (df_north2['priority'].fillna(0).astype(int) == 2) &
        (df_north2['create_date'] <= end) &
        (
            (df_north2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_north2[mask_medium].shape[0]
    mask_high = (
        (df_north2['helpdesk_ticket_tag_id'] != 3) &
        (df_north2['priority'].fillna(0).astype(int) == 3) &
        (df_north2['create_date'] <= end) &
        (
            (df_north2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_north2[mask_high].shape[0]
    mask_emergency = (
        (df_north2['helpdesk_ticket_tag_id'] == 3) &
        (df_north2['create_date'] <= end) &
        (
            (df_north2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_north2[mask_emergency].shape[0]
    table_data_priority_north2.append(row)
df_table_priority_north2 = pd.DataFrame(table_data_priority_north2)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w2 = len(week_ends) - 1
idx_w1_2 = idx_w2 - 1
w_label2 = df_table_priority_north2["Tuần"].iloc[idx_w2]
active_priorities2 = []
percent_changes2 = {}
priority_positions2 = {}
cumulative_height2 = 0
for pri in priority_cols2:
    count_w2 = float(df_table_priority_north2[pri].iloc[idx_w2])
    if count_w2 <= 0:
        continue
    count_w1_2 = float(df_table_priority_north2[pri].iloc[idx_w1_2])
    if count_w1_2 == 0:
        percent = 100 if count_w2 > 0 else 0
    else:
        percent = ((count_w2 - count_w1_2) / count_w1_2) * 100
    active_priorities2.append(pri)
    percent_changes2[pri] = percent
    priority_positions2[pri] = cumulative_height2 + count_w2 / 2
    cumulative_height2 += count_w2

fig_stack_priority_north2 = go.Figure()
for priority in priority_cols2:
    y_values = df_table_priority_north2[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_north2.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_north2["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors2[priority]
    ))
if active_priorities2:
    total_height2 = cumulative_height2
    x_vals2 = list(df_table_priority_north2["Tuần"])
    x_idx2 = x_vals2.index(w_label2)
    x_offset2 = x_idx2 + 2
    sorted_priorities2 = sorted(active_priorities2, key=lambda x: priority_positions2[x])
    for i, pri in enumerate(sorted_priorities2):
        percent = percent_changes2[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions2[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height2 * spacing_factor * (i - len(sorted_priorities2)/2))
        fig_stack_priority_north2.add_annotation(
            x=w_label2, y=y_col,
            ax=x_offset2, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_north2.add_annotation(
            x=x_offset2, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals2 = df_table_priority_north2[priority_cols2].sum(axis=1)
totals_offset2 = totals2 + totals2 * 0.04
fig_stack_priority_north2.add_trace(go.Scatter(
    x=df_table_priority_north2["Tuần"],
    y=totals_offset2,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_north2["Tuần"], totals_offset2, totals2)):
    fig_stack_priority_north2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_north2.update_layout(
    barmode='stack',
    title={
        'text': "NORTH 2 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_north2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency) ---
created_counts_high2 = []
solved_counts_high2 = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_north2[
        (df_north2['create_date'] >= start) &
        (df_north2['create_date'] <= end) &
        (df_north2['priority'].fillna(0).astype(int) == 3) &
        (df_north2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_north2[
        (df_north2['create_date'] >= start) &
        (df_north2['create_date'] <= end) &
        (df_north2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_north2[
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') <= end) &
        (df_north2['priority'].fillna(0).astype(int) == 3) &
        (df_north2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_north2[
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') <= end) &
        (df_north2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts_high2.append(created)
    solved_counts_high2.append(solved)

fig_high2 = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts_high2,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts_high2],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts_high2,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts_high2],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig_high2.update_layout(
    barmode='group',
    title={
        'text': "NORTH 2 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig_high2, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Clustered Chart: Created/Solved ticket Low & Medium Priority ---
created_counts_lowmed2 = []
solved_counts_lowmed2 = []
for start, end in zip(week_starts, week_ends):
    created_low = df_north2[
        (df_north2['create_date'] >= start) &
        (df_north2['create_date'] <= end) &
        (
            df_north2['priority'].isna() |
            (df_north2['priority'].astype(str).str.strip() == '0') |
            (df_north2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_north2[
        (df_north2['create_date'] >= start) &
        (df_north2['create_date'] <= end) &
        (df_north2['priority'].fillna(0).astype(int) == 2) &
        (df_north2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_north2[
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') <= end) &
        (
            df_north2['priority'].isna() |
            (df_north2['priority'].astype(str).str.strip() == '0') |
            (df_north2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_north2[
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') <= end) &
        (df_north2['priority'].fillna(0).astype(int) == 2) &
        (df_north2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts_lowmed2.append(created)
    solved_counts_lowmed2.append(solved)

fig_lowmed2 = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts_lowmed2,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts_lowmed2],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts_lowmed2,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts_lowmed2],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig_lowmed2.update_layout(
    barmode='group',
    title={
        'text': "NORTH 2 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig_lowmed2, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)




# Bảng Sites cho North 2 (cột 2: số ticket chưa end)
st.markdown("<h3 style='text-align: center;'>NORTH 2 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names2 = [

    "GO Mall Hung Yen (HYN)", "GO Mall Lao Cai (LCI)", "GO Mall Thai Nguyen (TNN)",
    "GO Mall Viet Tri (VTI)", "GO Mall Vinh Phuc (VPC)", "Hyper Lao Cai (LCI)",
    "Hyper Thai Nguyen (TNN)", "Hyper Viet Tri (VTI)", "Hyper Vinh Phuc (VPC)",
    "KUBO NANO Lao Cai (6439)", "KUBO NANO Thai Nguyen (6429)", "KUBO NANO Viet Tri (6437)",
    "KUBO NANO Vinh Phuc (6426)", "Nguyen Kim Thai Nguyen (TA02)"

]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company2 = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company2 = True

df_special_sites2 = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names2)
    & mask_company2
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites2 = df_special_sites2.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today2 = pd.Timestamp.now().normalize()
seven_days_ago2 = today2 - pd.Timedelta(days=7)
seventy_days_ago2 = today2 - pd.Timedelta(days=70)

# Khởi tạo các list lưu kết quả
site_ticket_not_end2 = []
site_ticket_7days2 = []
site_ticket_70days2 = []
site_ticket_emergency2 = []
site_ticket_high_priority2 = []
site_ticket_medium_priority2 = []
site_ticket_low_priority2 = []

# Lấy danh sách 11 category cần thống kê
category_list2 = df_north2['category_name'].dropna().unique()[:11]
site_ticket_by_category2 = {cat: [] for cat in category_list2}

sites_north2 = df_special_sites2['Sites'].tolist()

for site in sites_north2:
    # Tổng số ticket chưa end
    count_not_end2 = df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end2.append(count_not_end2)

    # CỘT 3 - 7 ngày
    count_old_not_end_7_2 = df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['create_date'] <= seven_days_ago2) &
        (df_north2['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7_2 = df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['create_date'] <= seven_days_ago2) &
        (df_north2['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > seven_days_ago2)
    ].shape[0]
    site_ticket_7days2.append(count_not_end2 - (count_old_not_end_7_2 + count_old_end_late_7_2))

    # CỘT 4 - 70 ngày
    count_old_not_end_70_2 = df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['create_date'] <= seventy_days_ago2) &
        (df_north2['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70_2 = df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['create_date'] <= seventy_days_ago2) &
        (df_north2['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_north2['custom_end_date'], errors='coerce') > seventy_days_ago2)
    ].shape[0]
    site_ticket_70days2.append(count_not_end2 - (count_old_not_end_70_2 + count_old_end_late_70_2))

    # Emergency chưa end
    site_ticket_emergency2.append(df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['custom_end_date'] == "not yet end") &
        (df_north2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    # High priority chưa end, không phải emergency
    site_ticket_high_priority2.append(df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['custom_end_date'] == "not yet end") &
        (df_north2['helpdesk_ticket_tag_id'] != 3) &
        (df_north2['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    # Medium priority chưa end, không phải emergency
    site_ticket_medium_priority2.append(df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['custom_end_date'] == "not yet end") &
        (df_north2['helpdesk_ticket_tag_id'] != 3) &
        (df_north2['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    # Low priority chưa end, không phải emergency
    site_ticket_low_priority2.append(df_north2[
        (df_north2['mall_display_name'] == site) &
        (df_north2['custom_end_date'] == "not yet end") &
        (df_north2['helpdesk_ticket_tag_id'] != 3) &
        (
            df_north2['priority'].isna() |
            (df_north2['priority'].astype(str).str.strip() == '0') |
            (df_north2['priority'].fillna(0).astype(int) == 0) |
            (df_north2['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    # Category columns
    for cat in category_list2:
        site_ticket_by_category2[cat].append(df_north2[
            (df_north2['mall_display_name'] == site) &
            (df_north2['custom_end_date'] == "not yet end") &
            (df_north2['category_name'] == cat)
        ].shape[0])

# Tạo DataFrame
data2 = {
    'Sites': sites_north2,
    'Total OA tickets': site_ticket_not_end2,
    'Vs last 7 days': site_ticket_7days2,
    'Vs last 70 days': site_ticket_70days2,
    'Emergency OA': site_ticket_emergency2,
    'High priority OA': site_ticket_high_priority2,
    'Medium priority OA': site_ticket_medium_priority2,
    'Low priority OA': site_ticket_low_priority2,
}
for cat in category_list2:
    data2[cat] = site_ticket_by_category2[cat]

df_sites_north2 = pd.DataFrame(data2)

# Thêm hàng Total (sum các cột số)
total_row2 = {col: df_sites_north2[col].sum() if df_sites_north2[col].dtype != 'O' else 'TOTAL' for col in df_sites_north2.columns}
df_sites_north2 = pd.concat([df_sites_north2, pd.DataFrame([total_row2])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols2 = [col for col in df_sites_north2.columns if col != 'Sites']
df_no_total2 = df_sites_north2.iloc[:-1][num_cols2]
vmin2 = df_no_total2.min().min()
vmax2 = df_no_total2.max().max()
vmid2 = df_no_total2.stack().quantile(0.5)  # 50th percentile

def color_scale2(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax2 == vmin2:
        norm = 0.5
    elif val <= vmid2:
        norm = (val - vmin2) / (vmid2 - vmin2) / 2 if vmid2 > vmin2 else 0
    else:
        norm = 0.5 + (val - vmid2) / (vmax2 - vmid2) / 2 if vmax2 > vmid2 else 1
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func2(val, row_idx):
    if row_idx == len(df_sites_north2) - 1:
        return ""
    return color_scale2(val)

def apply_color_scale2(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols2:
            styled.at[row_idx, col] = color_scale2(df.at[row_idx, col])
    return styled

styled2 = df_sites_north2.style.apply(lambda s: apply_color_scale2(df_sites_north2), axis=None)

def highlight_total2(s):
    is_total = s.name == len(df_sites_north2) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled2 = styled2.apply(highlight_total2, axis=1)

num_rows2 = df_sites_north2.shape[0]
row_height2 = 35
header_height2 = 38
st.dataframe(styled2, use_container_width=True, height=num_rows2 * row_height2 + header_height2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)



st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>NORTH 3 - Do Van Nam</h2>", unsafe_allow_html=True)
df_north3 = df[df['team_id'] == 1]  # team_id = 1 cho North 3
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Tạo lại pivot nếu chưa có
pivot3 = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all3 = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot3.insert(0, 'Across all category', across_all3)
pivot3 = pivot3.round(0).astype(int)

# Lấy giá trị cho North 3
value3 = pivot3.loc['NORTH 3 - Do Van Nam', 'Across all category']

gauge_max3 = 100
gauge_min3 = 0

# Xác định các mức fill
level1_3 = 33
level2_3 = 66

steps3 = []
if value3 > 0:
    steps3.append({'range': [0, min(value3, level1_3)], 'color': '#b7f7b7'})
if value3 > level1_3:
    steps3.append({'range': [level1_3, min(value3, level2_3)], 'color': '#ffe082'})
if value3 > level2_3:
    steps3.append({'range': [level2_3, min(value3, gauge_max3)], 'color': '#ffb3b3'})
if value3 < gauge_max3:
    steps3.append({'range': [value3, gauge_max3], 'color': '#eeeeee'})

fig_gauge3 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value3,
    gauge={
        'axis': {'range': [gauge_min3, gauge_max3]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps3,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge3.update_layout(
    annotations=[
        dict(
            x=0.5, y=0,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W) cho North 3
idx_w3 = len(week_ends) - 1
idx_w1_3 = idx_w3 - 1
end_w3 = week_ends[idx_w3]
end_w1_3 = week_ends[idx_w1_3]

# W-1
mask_w1_3 = (
    (df_north3['create_date'] <= end_w1_3) &
    (
        (df_north3['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end_w1_3)
    )
)
count_w1_3 = df_north3[mask_w1_3].shape[0]

# W
mask_w3 = (
    (df_north3['create_date'] <= end_w3) &
    (
        (df_north3['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end_w3)
    )
)
count_w3 = df_north3[mask_w3].shape[0]

# Tính % thay đổi
if count_w1_3 == 0:
    percent3 = 100 if count_w3 > 0 else 0
else:
    percent3 = ((count_w3 - count_w1_3) / count_w1_3) * 100

if percent3 > 0:
    percent_text3 = f"W vs W-1: +{percent3:.1f}%"
    bgcolor3 = "#f2c795"
elif percent3 < 0:
    percent_text3 = f"W vs W-1: -{abs(percent3):.1f}%"
    bgcolor3 = "#abf3ab"
else:
    percent_text3 = "W vs W-1: 0.0%"
    bgcolor3 = "#f2c795"

percent_value3 = f"{percent3:+.1f}%" if percent3 != 0 else "0.0%"

col1_3, col2_3 = st.columns([1, 0.9])
with col1_3:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor3}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value3}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2_3:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge3)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts3 = []
solved_counts3 = []
for start, end in zip(week_starts, week_ends):
    created = df_north3[(df_north3['create_date'] >= start) & (df_north3['create_date'] <= end)].shape[0]
    solved = -df_north3[(pd.to_datetime(df_north3['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts3.append(created)
    solved_counts3.append(solved)

fig3 = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts3,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts3],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts3,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts3],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig3.update_layout(
    barmode='group',
    title={
        'text': "NORTH 3 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Stacked Bar Chart theo Category cho North 3 ---
category_names_north3 = df_north3['category_name'].dropna().unique()
table_data_north3 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_north3:
        mask = (
            (df_north3['category_name'] == cat) &
            (df_north3['create_date'] <= end) &
            (
                (df_north3['custom_end_date'] == "not yet end") |
                (
                    (df_north3['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_north3[mask].shape[0]
        row[cat] = count
    table_data_north3.append(row)
df_table_north3 = pd.DataFrame(table_data_north3)

fig_stack_north3 = go.Figure()
for cat in category_names_north3:
    y_values = df_table_north3[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_north3.add_trace(go.Bar(
        name=cat,
        x=df_table_north3["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals3 = df_table_north3[category_names_north3].sum(axis=1)
totals_offset3 = totals3 + totals3 * 0.04
fig_stack_north3.add_trace(go.Scatter(
    x=df_table_north3["Tuần"],
    y=totals_offset3,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_north3["Tuần"], totals_offset3, totals3)):
    fig_stack_north3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w3 = len(week_ends) - 1
idx_w1_3 = idx_w3 - 1
w_label3 = df_table_north3["Tuần"].iloc[idx_w3]

active_categories3 = []
percent_changes3 = {}
category_positions3 = {}
cumulative_height3 = 0
for cat in category_names_north3:
    count_w3 = float(df_table_north3[cat].iloc[idx_w3])
    if count_w3 <= 0:
        continue
    count_w1_3 = float(df_table_north3[cat].iloc[idx_w1_3])
    if count_w1_3 == 0:
        percent = 100 if count_w3 > 0 else 0
    else:
        percent = ((count_w3 - count_w1_3) / count_w1_3) * 100
    active_categories3.append(cat)
    percent_changes3[cat] = percent
    category_positions3[cat] = cumulative_height3 + count_w3 / 2
    cumulative_height3 += count_w3

if active_categories3:
    total_height3 = cumulative_height3
    x_vals3 = list(df_table_north3["Tuần"])
    x_idx3 = x_vals3.index(w_label3)
    x_offset3 = x_idx3 + 2
    sorted_categories3 = sorted(active_categories3, key=lambda x: category_positions3[x])
    for i, cat in enumerate(sorted_categories3):
        percent = percent_changes3[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions3[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height3 * spacing_factor * (i - len(sorted_categories3)/2))
        fig_stack_north3.add_annotation(
            x=w_label3, y=y_col,
            ax=x_offset3, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_north3.add_annotation(
            x=x_offset3, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_north3.update_layout(
    barmode='stack',
    title=dict(
        text="NORTH 3 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_north3)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Stacked Bar Chart theo Priority cho North 3 ---
priority_cols3 = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors3 = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_north3 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_north3['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_north3['priority'].isna()) |
            (df_north3['priority'].astype(str).str.strip() == '0') |
            (df_north3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north3['create_date'] <= end) &
        (
            (df_north3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_north3[mask_low].shape[0]
    mask_medium = (
        (df_north3['helpdesk_ticket_tag_id'] != 3) &
        (df_north3['priority'].fillna(0).astype(int) == 2) &
        (df_north3['create_date'] <= end) &
        (
            (df_north3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_north3[mask_medium].shape[0]
    mask_high = (
        (df_north3['helpdesk_ticket_tag_id'] != 3) &
        (df_north3['priority'].fillna(0).astype(int) == 3) &
        (df_north3['create_date'] <= end) &
        (
            (df_north3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_north3[mask_high].shape[0]
    mask_emergency = (
        (df_north3['helpdesk_ticket_tag_id'] == 3) &
        (df_north3['create_date'] <= end) &
        (
            (df_north3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_north3[mask_emergency].shape[0]
    table_data_priority_north3.append(row)
df_table_priority_north3 = pd.DataFrame(table_data_priority_north3)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w3 = len(week_ends) - 1
idx_w1_3 = idx_w3 - 1
w_label3 = df_table_priority_north3["Tuần"].iloc[idx_w3]
active_priorities3 = []
percent_changes3 = {}
priority_positions3 = {}
cumulative_height3 = 0
for pri in priority_cols3:
    count_w3 = float(df_table_priority_north3[pri].iloc[idx_w3])
    if count_w3 <= 0:
        continue
    count_w1_3 = float(df_table_priority_north3[pri].iloc[idx_w1_3])
    if count_w1_3 == 0:
        percent = 100 if count_w3 > 0 else 0
    else:
        percent = ((count_w3 - count_w1_3) / count_w1_3) * 100
    active_priorities3.append(pri)
    percent_changes3[pri] = percent
    priority_positions3[pri] = cumulative_height3 + count_w3 / 2
    cumulative_height3 += count_w3

fig_stack_priority_north3 = go.Figure()
for priority in priority_cols3:
    y_values = df_table_priority_north3[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_north3.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_north3["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors3[priority]
    ))
if active_priorities3:
    total_height3 = cumulative_height3
    x_vals3 = list(df_table_priority_north3["Tuần"])
    x_idx3 = x_vals3.index(w_label3)
    x_offset3 = x_idx3 + 2
    sorted_priorities3 = sorted(active_priorities3, key=lambda x: priority_positions3[x])
    for i, pri in enumerate(sorted_priorities3):
        percent = percent_changes3[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions3[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height3 * spacing_factor * (i - len(sorted_priorities3)/2))
        fig_stack_priority_north3.add_annotation(
            x=w_label3, y=y_col,
            ax=x_offset3, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_north3.add_annotation(
            x=x_offset3, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals3 = df_table_priority_north3[priority_cols3].sum(axis=1)
totals_offset3 = totals3 + totals3 * 0.04
fig_stack_priority_north3.add_trace(go.Scatter(
    x=df_table_priority_north3["Tuần"],
    y=totals_offset3,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_north3["Tuần"], totals_offset3, totals3)):
    fig_stack_priority_north3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_north3.update_layout(
    barmode='stack',
    title={
        'text': "NORTH 3 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_north3)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency) ---
created_counts_high3 = []
solved_counts_high3 = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_north3[
        (df_north3['create_date'] >= start) &
        (df_north3['create_date'] <= end) &
        (df_north3['priority'].fillna(0).astype(int) == 3) &
        (df_north3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_north3[
        (df_north3['create_date'] >= start) &
        (df_north3['create_date'] <= end) &
        (df_north3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_north3[
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') <= end) &
        (df_north3['priority'].fillna(0).astype(int) == 3) &
        (df_north3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_north3[
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') <= end) &
        (df_north3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts_high3.append(created)
    solved_counts_high3.append(solved)

fig_high3 = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts_high3,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts_high3],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts_high3,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts_high3],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig_high3.update_layout(
    barmode='group',
    title={
        'text': "NORTH 3 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig_high3, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --- Clustered Chart: Created/Solved ticket Low & Medium Priority ---
created_counts_lowmed3 = []
solved_counts_lowmed3 = []
for start, end in zip(week_starts, week_ends):
    created_low = df_north3[
        (df_north3['create_date'] >= start) &
        (df_north3['create_date'] <= end) &
        (
            df_north3['priority'].isna() |
            (df_north3['priority'].astype(str).str.strip() == '0') |
            (df_north3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_north3[
        (df_north3['create_date'] >= start) &
        (df_north3['create_date'] <= end) &
        (df_north3['priority'].fillna(0).astype(int) == 2) &
        (df_north3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_north3[
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') <= end) &
        (
            df_north3['priority'].isna() |
            (df_north3['priority'].astype(str).str.strip() == '0') |
            (df_north3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_north3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_north3[
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') <= end) &
        (df_north3['priority'].fillna(0).astype(int) == 2) &
        (df_north3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts_lowmed3.append(created)
    solved_counts_lowmed3.append(solved)

fig_lowmed3 = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts_lowmed3,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts_lowmed3],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts_lowmed3,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts_lowmed3],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig_lowmed3.update_layout(
    barmode='group',
    title={
        'text': "NORTH 3 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig_lowmed3, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho North 3 (cột 2: số ticket chưa end)
st.markdown("<h3 style='text-align: center;'>NORTH 3 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names3 = [

    "CBS Columbia Lotte Lieu Giai (2763M8)", "CBS Columbia Lotte Mall Ha Noi (2763Q2)", "CBS Crocs Aeon Ha Dong (2763M9)",
    "CBS Crocs Aeon Long Bien (276342)", "CBS Crocs Big C Thang Long (276348)", "CBS Crocs Chua Boc (2763R3)",
    "CBS Crocs Hang Bong (276334)", "CBS Crocs Indochina Plaza Ha Noi (2763P8)", "CBS Crocs Lotte Lieu Giai (2763M7)",
    "CBS Crocs Lotte Mall Ha Noi (2763P9)", "CBS Crocs Savico Long Bien (2763R2)", "CBS Crocs The Garden (2763O3)",
    "CBS Crocs Vincom Ba Trieu (2763E2)", "CBS Crocs Vincom Bac Giang (2763U4)", "CBS Crocs Vincom Bac Ninh (2763T4)",
    "CBS Crocs Vincom Long Bien (2763O8)", "CBS Crocs Vincom Metropolis (2763E4)", "CBS Crocs Vincom Ocean Park (2763E8)",
    "CBS Crocs Vincom Pham Ngoc Thach (2763E1)", "CBS Crocs Vincom Royal City (2763E3)", "CBS Crocs Vincom Smart City (2763G8)",
    "CBS Crocs Vincom Star City (2763G5)", "CBS Crocs Vincom Times City (276315)", "CBS Dyson Lotte Lieu Giai (2763AN)",
    "CBS Dyson Lotte Mall Ha Noi (2763BL)", "CBS Dyson Nguyen Kim Trang Thi (2763AU)", "CBS Dyson Vincom Ba Trieu (2763AT)",
    "CBS Dyson Vincom Star City (2763BF)", "CBS Fila Aeon Ha Dong (2763M6)", "CBS Fila Lotte Lieu Giai (2763H7)",
    "CBS Fila Lotte Mall Ha Noi (2763Q3)", "CBS Fila Vincom Pham Ngoc Thach (2763D2)", "CBS Fila Vincom Star City (2763G6)",
    "CBS Fila Vincom Times City (2763G4)", "CBS Fitflop Lotte DS (2763S6)", "CBS Supersports Aeon Long Bien (276341)",
    "CBS Supersports Big C Thang Long (276305)", "CBS Supersports Lotte Lieu Giai (276319)", "CBS Supersports Lotte Mall Ha Noi (2763Q1)",
    "CBS Under Armour Lotte Lieu Giai (2763N1)", "GO Mall Bac Giang (BGG)", "GO Mall Ha Dong (HDO)",
    "GO Mall Le Trong Tan (LTT)", "GO Mall Long Bien (LBN)", "GO Mall Nguyen Xien (NXN)",
    "GO Mall Thang Long (TLG)", "Hyper Bac Giang (BGG)", "Hyper Long Bien (LBN)",
    "Hyper Me Linh (MLH)", "Hyper Thang Long (TLG)", "KUBO NANO Bac Giang (6414)",
    "KUBO NANO Le Trong Tan (6432)", "KUBO NANO Long Bien (6428)", "KUBO NANO Smart City (6440)",
    "KUBO NANO Thang Long (6421)", "Nguyen Kim Ba Dinh (HN02)", "Nguyen Kim Ba Dinh (Mac Plaza) (Hn16)",
    "Nguyen Kim Ha Dong (HN04)", "Nguyen Kim Long Bien (HN13)", "Nguyen Kim Thang Long (HN07)",
    "Nguyen Kim Trang Thi (HN01)", "Tops Ha Dong (HDO)", "Tops Le Trong Tan (LTT)",
    "Tops Nguyen Xien (NXN)", "Tops Park City (PCY)", "Tops The Garden (TGN)"

]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company3 = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company3 = True

df_special_sites3 = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names3)
    & mask_company3
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites3 = df_special_sites3.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today3 = pd.Timestamp.now().normalize()
seven_days_ago3 = today3 - pd.Timedelta(days=7)
seventy_days_ago3 = today3 - pd.Timedelta(days=70)

# Khởi tạo các list lưu kết quả
site_ticket_not_end3 = []
site_ticket_7days3 = []
site_ticket_70days3 = []
site_ticket_emergency3 = []
site_ticket_high_priority3 = []
site_ticket_medium_priority3 = []
site_ticket_low_priority3 = []

# Lấy danh sách 11 category cần thống kê
category_list3 = df_north3['category_name'].dropna().unique()[:11]
site_ticket_by_category3 = {cat: [] for cat in category_list3}

sites_north3 = df_special_sites3['Sites'].tolist()

for site in sites_north3:
    # Tổng số ticket chưa end
    count_not_end3 = df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end3.append(count_not_end3)

    # CỘT 3 - 7 ngày
    count_old_not_end_7_3 = df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['create_date'] <= seven_days_ago3) &
        (df_north3['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7_3 = df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['create_date'] <= seven_days_ago3) &
        (df_north3['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > seven_days_ago3)
    ].shape[0]
    site_ticket_7days3.append(count_not_end3 - (count_old_not_end_7_3 + count_old_end_late_7_3))

    # CỘT 4 - 70 ngày
    count_old_not_end_70_3 = df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['create_date'] <= seventy_days_ago3) &
        (df_north3['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70_3 = df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['create_date'] <= seventy_days_ago3) &
        (df_north3['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_north3['custom_end_date'], errors='coerce') > seventy_days_ago3)
    ].shape[0]
    site_ticket_70days3.append(count_not_end3 - (count_old_not_end_70_3 + count_old_end_late_70_3))

    # Emergency chưa end
    site_ticket_emergency3.append(df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['custom_end_date'] == "not yet end") &
        (df_north3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    # High priority chưa end, không phải emergency
    site_ticket_high_priority3.append(df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['custom_end_date'] == "not yet end") &
        (df_north3['helpdesk_ticket_tag_id'] != 3) &
        (df_north3['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    # Medium priority chưa end, không phải emergency
    site_ticket_medium_priority3.append(df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['custom_end_date'] == "not yet end") &
        (df_north3['helpdesk_ticket_tag_id'] != 3) &
        (df_north3['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    # Low priority chưa end, không phải emergency
    site_ticket_low_priority3.append(df_north3[
        (df_north3['mall_display_name'] == site) &
        (df_north3['custom_end_date'] == "not yet end") &
        (df_north3['helpdesk_ticket_tag_id'] != 3) &
        (
            df_north3['priority'].isna() |
            (df_north3['priority'].astype(str).str.strip() == '0') |
            (df_north3['priority'].fillna(0).astype(int) == 0) |
            (df_north3['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    # Category columns
    for cat in category_list3:
        site_ticket_by_category3[cat].append(df_north3[
            (df_north3['mall_display_name'] == site) &
            (df_north3['custom_end_date'] == "not yet end") &
            (df_north3['category_name'] == cat)
        ].shape[0])

# Tạo DataFrame
data3 = {
    'Sites': sites_north3,
    'Total OA tickets': site_ticket_not_end3,
    'Vs last 7 days': site_ticket_7days3,
    'Vs last 70 days': site_ticket_70days3,
    'Emergency OA': site_ticket_emergency3,
    'High priority OA': site_ticket_high_priority3,
    'Medium priority OA': site_ticket_medium_priority3,
    'Low priority OA': site_ticket_low_priority3,
}
for cat in category_list3:
    data3[cat] = site_ticket_by_category3[cat]

df_sites_north3 = pd.DataFrame(data3)

# Thêm hàng Total (sum các cột số)
total_row3 = {col: df_sites_north3[col].sum() if df_sites_north3[col].dtype != 'O' else 'TOTAL' for col in df_sites_north3.columns}
df_sites_north3 = pd.concat([df_sites_north3, pd.DataFrame([total_row3])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols3 = [col for col in df_sites_north3.columns if col != 'Sites']
df_no_total3 = df_sites_north3.iloc[:-1][num_cols3]
vmin3 = df_no_total3.min().min()
vmax3 = df_no_total3.max().max()
vmid3 = df_no_total3.stack().quantile(0.5)  # 50th percentile

def color_scale3(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax3 == vmin3:
        norm = 0.5
    elif val <= vmid3:
        norm = (val - vmin3) / (vmid3 - vmin3) / 2 if vmid3 > vmin3 else 0
    else:
        norm = 0.5 + (val - vmid3) / (vmax3 - vmid3) / 2 if vmax3 > vmid3 else 1
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func3(val, row_idx):
    if row_idx == len(df_sites_north3) - 1:
        return ""
    return color_scale3(val)

def apply_color_scale3(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols3:
            styled.at[row_idx, col] = color_scale3(df.at[row_idx, col])
    return styled

styled3 = df_sites_north3.style.apply(lambda s: apply_color_scale3(df_sites_north3), axis=None)

def highlight_total3(s):
    is_total = s.name == len(df_sites_north3) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled3 = styled3.apply(highlight_total3, axis=1)

num_rows3 = df_sites_north3.shape[0]
row_height3 = 35
header_height3 = 38
st.dataframe(styled3, use_container_width=True, height=num_rows3 * row_height3 + header_height3)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 1 - Huynh Bao Dang</h2>", unsafe_allow_html=True)
df_center1 = df[df['team_id'] == 18]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot_center1 = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all_center1 = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot_center1.insert(0, 'Across all category', across_all_center1)
pivot_center1 = pivot_center1.round(0).astype(int)

# Lấy giá trị cho Center 1
value_center1 = pivot_center1.loc['CENTER 1 - Huynh Bao Dang', 'Across all category']

gauge_max = 100
gauge_min = 0

level1 = 33
level2 = 66

steps = []
if value_center1 > 0:
    steps.append({'range': [0, min(value_center1, level1)], 'color': '#b7f7b7'})
if value_center1 > level1:
    steps.append({'range': [level1, min(value_center1, level2)], 'color': '#ffe082'})
if value_center1 > level2:
    steps.append({'range': [level2, min(value_center1, gauge_max)], 'color': '#ffb3b3'})
if value_center1 < gauge_max:
    steps.append({'range': [value_center1, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value_center1,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_center1['create_date'] <= end_w1) &
    (
        (df_center1['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_center1[mask_w1].shape[0]

mask_w = (
    (df_center1['create_date'] <= end_w) &
    (
        (df_center1['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_center1[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_center1[(df_center1['create_date'] >= start) & (df_center1['create_date'] <= end)].shape[0]
    solved = -df_center1[(pd.to_datetime(df_center1['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 1 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho Center 1
category_names_center1 = df_center1['category_name'].dropna().unique()
table_data_center1 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_center1:
        mask = (
            (df_center1['category_name'] == cat) &
            (df_center1['create_date'] <= end) &
            (
                (df_center1['custom_end_date'] == "not yet end") |
                (
                    (df_center1['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_center1[mask].shape[0]
        row[cat] = count
    table_data_center1.append(row)
df_table_center1 = pd.DataFrame(table_data_center1)

fig_stack_center1 = go.Figure()
for cat in category_names_center1:
    y_values = df_table_center1[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_center1.add_trace(go.Bar(
        name=cat,
        x=df_table_center1["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_center1[category_names_center1].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_center1.add_trace(go.Scatter(
    x=df_table_center1["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_center1["Tuần"], totals_offset, totals)):
    fig_stack_center1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_center1["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_center1:
    count_w = float(df_table_center1[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_center1[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_center1["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_center1.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_center1.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_center1.update_layout(
    barmode='stack',
    title=dict(
        text="CENTER 1 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_center1)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho Center 1
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_center1 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_center1['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_center1['priority'].isna()) |
            (df_center1['priority'].astype(str).str.strip() == '0') |
            (df_center1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center1['create_date'] <= end) &
        (
            (df_center1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_center1[mask_low].shape[0]
    mask_medium = (
        (df_center1['helpdesk_ticket_tag_id'] != 3) &
        (df_center1['priority'].fillna(0).astype(int) == 2) &
        (df_center1['create_date'] <= end) &
        (
            (df_center1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_center1[mask_medium].shape[0]
    mask_high = (
        (df_center1['helpdesk_ticket_tag_id'] != 3) &
        (df_center1['priority'].fillna(0).astype(int) == 3) &
        (df_center1['create_date'] <= end) &
        (
            (df_center1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_center1[mask_high].shape[0]
    mask_emergency = (
        (df_center1['helpdesk_ticket_tag_id'] == 3) &
        (df_center1['create_date'] <= end) &
        (
            (df_center1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_center1[mask_emergency].shape[0]
    table_data_priority_center1.append(row)
df_table_priority_center1 = pd.DataFrame(table_data_priority_center1)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_center1["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_center1[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_center1[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_center1 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_center1[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_center1.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_center1["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_center1["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_center1.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_center1.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_center1[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_center1.add_trace(go.Scatter(
    x=df_table_priority_center1["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_center1["Tuần"], totals_offset, totals)):
    fig_stack_priority_center1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_center1.update_layout(
    barmode='stack',
    title={
        'text': "CENTER 1 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_center1)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_center1[
        (df_center1['create_date'] >= start) &
        (df_center1['create_date'] <= end) &
        (df_center1['priority'].fillna(0).astype(int) == 3) &
        (df_center1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_center1[
        (df_center1['create_date'] >= start) &
        (df_center1['create_date'] <= end) &
        (df_center1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_center1[
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') <= end) &
        (df_center1['priority'].fillna(0).astype(int) == 3) &
        (df_center1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_center1[
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') <= end) &
        (df_center1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 1 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_center1[
        (df_center1['create_date'] >= start) &
        (df_center1['create_date'] <= end) &
        (
            df_center1['priority'].isna() |
            (df_center1['priority'].astype(str).str.strip() == '0') |
            (df_center1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_center1[
        (df_center1['create_date'] >= start) &
        (df_center1['create_date'] <= end) &
        (df_center1['priority'].fillna(0).astype(int) == 2)
        &
        (df_center1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_center1[
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') <= end) &
        (
            df_center1['priority'].isna() |
            (df_center1['priority'].astype(str).str.strip() == '0') |
            (df_center1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_center1[
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') <= end) &
        (df_center1['priority'].fillna(0).astype(int) == 2)
        &
        (df_center1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 1 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho Center 1
st.markdown("<h3 style='text-align: center;'>CENTER 1 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs Ba Thang Hai Da Lat (2763R1)", "CBS Crocs Buon Ma Thuot (2763O4)", "CBS Crocs Go Da Lat (2763K7)",
    "GO Mall Buon Ma Thuot (BMT)", "GO Mall Da Lat (DLT)", "Hyper Buon Ma Thuot (BMT)",
    "Hyper Da Lat (DLT)", "KUBO NANO Buon Ma Thuot (6418)", "KUBO NANO Da Lat (6419)",
    "KUBO NANO Kon Tum (6434)", "Nguyen Kim Buon Ma Thuot (DL01)", "Nguyen Kim Buon Ma Thuot (DL04)"

]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_center1['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_center1 = df_special_sites['Sites'].tolist()

for site in sites_center1:
    count_not_end = df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['create_date'] <= seven_days_ago) &
        (df_center1['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['create_date'] <= seven_days_ago) &
        (df_center1['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['create_date'] <= seventy_days_ago) &
        (df_center1['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['create_date'] <= seventy_days_ago) &
        (df_center1['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center1['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['custom_end_date'] == "not yet end") &
        (df_center1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['custom_end_date'] == "not yet end") &
        (df_center1['helpdesk_ticket_tag_id'] != 3) &
        (df_center1['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['custom_end_date'] == "not yet end") &
        (df_center1['helpdesk_ticket_tag_id'] != 3) &
        (df_center1['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_center1[
        (df_center1['mall_display_name'] == site) &
        (df_center1['custom_end_date'] == "not yet end") &
        (df_center1['helpdesk_ticket_tag_id'] != 3) &
        (
            df_center1['priority'].isna() |
            (df_center1['priority'].astype(str).str.strip() == '0') |
            (df_center1['priority'].fillna(0).astype(int) == 0) |
            (df_center1['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_center1[
            (df_center1['mall_display_name'] == site) &
            (df_center1['custom_end_date'] == "not yet end") &
            (df_center1['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_center1,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_center1 = pd.DataFrame(data)

total_row = {col: df_sites_center1[col].sum() if df_sites_center1[col].dtype != 'O' else 'TOTAL' for col in df_sites_center1.columns}
df_sites_center1 = pd.concat([df_sites_center1, pd.DataFrame([total_row])], ignore_index=True)

num_cols = [col for col in df_sites_center1.columns if col != 'Sites']
df_no_total = df_sites_center1.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    if row_idx == len(df_sites_center1) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_center1.style.apply(lambda s: apply_color_scale(df_sites_center1), axis=None)

def highlight_total(s):
    is_total = s.name == len(df_sites_center1) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_center1.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 2 - Luu Duc Thach</h2>", unsafe_allow_html=True)
df_center2 = df[df['team_id'] == 3]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot_center2 = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all_center2 = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot_center2.insert(0, 'Across all category', across_all_center2)
pivot_center2 = pivot_center2.round(0).astype(int)

# Lấy giá trị cho Center 2
value = pivot_center2.loc['CENTER 2 - Luu Duc Thach', 'Across all category']

gauge_max = 100
gauge_min = 0

level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_center2['create_date'] <= end_w1) &
    (
        (df_center2['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_center2[mask_w1].shape[0]

mask_w = (
    (df_center2['create_date'] <= end_w) &
    (
        (df_center2['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_center2[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_center2[(df_center2['create_date'] >= start) & (df_center2['create_date'] <= end)].shape[0]
    solved = -df_center2[(pd.to_datetime(df_center2['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 2 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho Center 2
category_names_center2 = df_center2['category_name'].dropna().unique()
table_data_center2 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_center2:
        mask = (
            (df_center2['category_name'] == cat) &
            (df_center2['create_date'] <= end) &
            (
                (df_center2['custom_end_date'] == "not yet end") |
                (
                    (df_center2['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_center2[mask].shape[0]
        row[cat] = count
    table_data_center2.append(row)
df_table_center2 = pd.DataFrame(table_data_center2)

fig_stack_center2 = go.Figure()
for cat in category_names_center2:
    y_values = df_table_center2[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_center2.add_trace(go.Bar(
        name=cat,
        x=df_table_center2["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_center2[category_names_center2].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_center2.add_trace(go.Scatter(
    x=df_table_center2["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_center2["Tuần"], totals_offset, totals)):
    fig_stack_center2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_center2["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_center2:
    count_w = float(df_table_center2[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_center2[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_center2["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_center2.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_center2.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_center2.update_layout(
    barmode='stack',
    title=dict(
        text="CENTER 2 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_center2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho Center 2
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_center2 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_center2['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_center2['priority'].isna()) |
            (df_center2['priority'].astype(str).str.strip() == '0') |
            (df_center2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center2['create_date'] <= end) &
        (
            (df_center2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_center2[mask_low].shape[0]
    mask_medium = (
        (df_center2['helpdesk_ticket_tag_id'] != 3) &
        (df_center2['priority'].fillna(0).astype(int) == 2) &
        (df_center2['create_date'] <= end) &
        (
            (df_center2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_center2[mask_medium].shape[0]
    mask_high = (
        (df_center2['helpdesk_ticket_tag_id'] != 3) &
        (df_center2['priority'].fillna(0).astype(int) == 3) &
        (df_center2['create_date'] <= end) &
        (
            (df_center2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_center2[mask_high].shape[0]
    mask_emergency = (
        (df_center2['helpdesk_ticket_tag_id'] == 3) &
        (df_center2['create_date'] <= end) &
        (
            (df_center2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_center2[mask_emergency].shape[0]
    table_data_priority_center2.append(row)
df_table_priority_center2 = pd.DataFrame(table_data_priority_center2)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_center2["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_center2[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_center2[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_center2 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_center2[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_center2.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_center2["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_center2["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_center2.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_center2.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_center2[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_center2.add_trace(go.Scatter(
    x=df_table_priority_center2["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_center2["Tuần"], totals_offset, totals)):
    fig_stack_priority_center2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_center2.update_layout(
    barmode='stack',
    title={
        'text': "CENTER 2 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_center2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_center2[
        (df_center2['create_date'] >= start) &
        (df_center2['create_date'] <= end) &
        (df_center2['priority'].fillna(0).astype(int) == 3) &
        (df_center2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_center2[
        (df_center2['create_date'] >= start) &
        (df_center2['create_date'] <= end) &
        (df_center2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_center2[
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') <= end) &
        (df_center2['priority'].fillna(0).astype(int) == 3) &
        (df_center2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_center2[
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') <= end) &
        (df_center2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 2 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_center2[
        (df_center2['create_date'] >= start) &
        (df_center2['create_date'] <= end) &
        (
            df_center2['priority'].isna() |
            (df_center2['priority'].astype(str).str.strip() == '0') |
            (df_center2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_center2[
        (df_center2['create_date'] >= start) &
        (df_center2['create_date'] <= end) &
        (df_center2['priority'].fillna(0).astype(int) == 2)
        &
        (df_center2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_center2[
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') <= end) &
        (
            df_center2['priority'].isna() |
            (df_center2['priority'].astype(str).str.strip() == '0') |
            (df_center2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_center2[
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') <= end) &
        (df_center2['priority'].fillna(0).astype(int) == 2)
        &
        (df_center2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 2 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho Center 2
st.markdown("<h3 style='text-align: center;'>CENTER 2 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Columbia Gold Coast Nha Trang (2763Y8)", "CBS Crocs Gold Coast Nha Trang (2763H5)", "CBS Crocs Lotte Mart Phan Thiet (2763S1)",
    "CBS Crocs Vincom Nha Trang (2763N6)", "CBS FitFlop Gold Coast Mall (2763T1)", "CBS Speedo Gold Coast Nha Trang (2763Z2)",
    "GO Mall Nha Trang (NTG)", "GO Mall Quy Nhon (QNN)", "Hyper Nha Trang (NTG)",
    "Hyper Ninh Thuan (NTN)", "Hyper Quy Nhon (QNN)", "KUBO NANO An Nhon (6452)",
    "KUBO NANO Cam Ranh (6441)", "KUBO NANO Nha Trang (6425)", "KUBO NANO Ninh Thuan (6454)",
    "KUBO NANO Quy Nhon (6435)", "mini go! An Nhon (1510)", "Nguyen Kim Nha Trang (KH02)",
    "Nguyen Kim Phan Thiet (BT01)", "Nguyen Kim Quy Nhon (BI01)"

]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_center2['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_center2 = df_special_sites['Sites'].tolist()

for site in sites_center2:
    count_not_end = df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['create_date'] <= seven_days_ago) &
        (df_center2['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['create_date'] <= seven_days_ago) &
        (df_center2['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['create_date'] <= seventy_days_ago) &
        (df_center2['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['create_date'] <= seventy_days_ago) &
        (df_center2['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center2['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['custom_end_date'] == "not yet end") &
        (df_center2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['custom_end_date'] == "not yet end") &
        (df_center2['helpdesk_ticket_tag_id'] != 3) &
        (df_center2['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['custom_end_date'] == "not yet end") &
        (df_center2['helpdesk_ticket_tag_id'] != 3) &
        (df_center2['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_center2[
        (df_center2['mall_display_name'] == site) &
        (df_center2['custom_end_date'] == "not yet end") &
        (df_center2['helpdesk_ticket_tag_id'] != 3) &
        (
            df_center2['priority'].isna() |
            (df_center2['priority'].astype(str).str.strip() == '0') |
            (df_center2['priority'].fillna(0).astype(int) == 0) |
            (df_center2['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_center2[
            (df_center2['mall_display_name'] == site) &
            (df_center2['custom_end_date'] == "not yet end") &
            (df_center2['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_center2,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_center2 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_center2[col].sum() if df_sites_center2[col].dtype != 'O' else 'TOTAL' for col in df_sites_center2.columns}
df_sites_center2 = pd.concat([df_sites_center2, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_center2.columns if col != 'Sites']
df_no_total = df_sites_center2.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_center2) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_center2.style.apply(lambda s: apply_color_scale(df_sites_center2), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_center2) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_center2.shape[0]
row_height = 35  # hoặc 32, tuỳ font
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 3 - Le Duc Thanh</h2>", unsafe_allow_html=True)
df_center3 = df[df['team_id'] == 19]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Tạo lại pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho Center 3
value = pivot.loc['CENTER 3 - Le Duc Thanh', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_center3['create_date'] <= end_w1) &
    (
        (df_center3['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_center3[mask_w1].shape[0]

mask_w = (
    (df_center3['create_date'] <= end_w) &
    (
        (df_center3['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_center3[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_center3[(df_center3['create_date'] >= start) & (df_center3['create_date'] <= end)].shape[0]
    solved = -df_center3[(pd.to_datetime(df_center3['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 3 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho Center 3
category_names_center3 = df_center3['category_name'].dropna().unique()
table_data_center3 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_center3:
        mask = (
            (df_center3['category_name'] == cat) &
            (df_center3['create_date'] <= end) &
            (
                (df_center3['custom_end_date'] == "not yet end") |
                (
                    (df_center3['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_center3[mask].shape[0]
        row[cat] = count
    table_data_center3.append(row)
df_table_center3 = pd.DataFrame(table_data_center3)

fig_stack_center3 = go.Figure()
for cat in category_names_center3:
    y_values = df_table_center3[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_center3.add_trace(go.Bar(
        name=cat,
        x=df_table_center3["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_center3[category_names_center3].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_center3.add_trace(go.Scatter(
    x=df_table_center3["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_center3["Tuần"], totals_offset, totals)):
    fig_stack_center3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_center3["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_center3:
    count_w = float(df_table_center3[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_center3[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_center3["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_center3.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_center3.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_center3.update_layout(
    barmode='stack',
    title=dict(
        text="CENTER 3 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_center3)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho Center 3
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_center3 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_center3['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_center3['priority'].isna()) |
            (df_center3['priority'].astype(str).str.strip() == '0') |
            (df_center3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center3['create_date'] <= end) &
        (
            (df_center3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_center3[mask_low].shape[0]
    mask_medium = (
        (df_center3['helpdesk_ticket_tag_id'] != 3) &
        (df_center3['priority'].fillna(0).astype(int) == 2) &
        (df_center3['create_date'] <= end) &
        (
            (df_center3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_center3[mask_medium].shape[0]
    mask_high = (
        (df_center3['helpdesk_ticket_tag_id'] != 3) &
        (df_center3['priority'].fillna(0).astype(int) == 3) &
        (df_center3['create_date'] <= end) &
        (
            (df_center3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_center3[mask_high].shape[0]
    mask_emergency = (
        (df_center3['helpdesk_ticket_tag_id'] == 3) &
        (df_center3['create_date'] <= end) &
        (
            (df_center3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_center3[mask_emergency].shape[0]
    table_data_priority_center3.append(row)
df_table_priority_center3 = pd.DataFrame(table_data_priority_center3)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_center3["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_center3[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_center3[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_center3 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_center3[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_center3.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_center3["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_center3["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_center3.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_center3.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_center3[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_center3.add_trace(go.Scatter(
    x=df_table_priority_center3["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_center3["Tuần"], totals_offset, totals)):
    fig_stack_priority_center3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_center3.update_layout(
    barmode='stack',
    title={
        'text': "CENTER 3 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_center3)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_center3[
        (df_center3['create_date'] >= start) &
        (df_center3['create_date'] <= end) &
        (df_center3['priority'].fillna(0).astype(int) == 3) &
        (df_center3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_center3[
        (df_center3['create_date'] >= start) &
        (df_center3['create_date'] <= end) &
        (df_center3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_center3[
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') <= end) &
        (df_center3['priority'].fillna(0).astype(int) == 3) &
        (df_center3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_center3[
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') <= end) &
        (df_center3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 3 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_center3[
        (df_center3['create_date'] >= start) &
        (df_center3['create_date'] <= end) &
        (
            df_center3['priority'].isna() |
            (df_center3['priority'].astype(str).str.strip() == '0') |
            (df_center3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_center3[
        (df_center3['create_date'] >= start) &
        (df_center3['create_date'] <= end) &
        (df_center3['priority'].fillna(0).astype(int) == 2)
        &
        (df_center3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_center3[
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') <= end) &
        (
            df_center3['priority'].isna() |
            (df_center3['priority'].astype(str).str.strip() == '0') |
            (df_center3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_center3[
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') <= end) &
        (df_center3['priority'].fillna(0).astype(int) == 2)
        &
        (df_center3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 3 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho Center 3
st.markdown("<h3 style='text-align: center;'>CENTER 3 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs Aeon Mall Hue (2763U5)", "CBS Crocs Da Nang (2763P5)", "CBS Crocs Go Hue (2763K2)",
    "CBS Crocs Hoi An (2763P6)", "CBS Crocs Lotte Da Nang (2763X9)", "CBS Crocs Vincom Da Nang (276316)",
    "CBS Dyson Vincom Da Nang (2763BK)", "CBS Fila Vincom Da Nang (2763K3)", "CBS FitFlop Vincom Da Nang (2763CC)",
    "CBS Supersports Big C Da Nang (276378)", "GO Mall Da Nang (DNG)", "GO Mall Hue (HUE)",
    "GO Mall Quang Ngai (QNI)", "Hyper Da Nang (DNG)", "Hyper Hue (HUE)",
    "Hyper Quang Ngai (QNI)", "KUBO NANO Da Nang (6411)", "KUBO NANO Dien Ban (6444)",
    "KUBO NANO HUE (6405)", "KUBO NANO Quang Ngai (6404)", "KUBO NANO Tam Ky (6417)",
    "mini go! Dien Ban (1505)", "mini go! Huong Tra (1511)", "mini go! Tam Ky (1500)",
    "Nguyen Kim Mien Trung (DN01)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_center3['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_center3 = df_special_sites['Sites'].tolist()

for site in sites_center3:
    count_not_end = df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['create_date'] <= seven_days_ago) &
        (df_center3['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['create_date'] <= seven_days_ago) &
        (df_center3['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['create_date'] <= seventy_days_ago) &
        (df_center3['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['create_date'] <= seventy_days_ago) &
        (df_center3['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center3['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['custom_end_date'] == "not yet end") &
        (df_center3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['custom_end_date'] == "not yet end") &
        (df_center3['helpdesk_ticket_tag_id'] != 3) &
        (df_center3['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['custom_end_date'] == "not yet end") &
        (df_center3['helpdesk_ticket_tag_id'] != 3) &
        (df_center3['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_center3[
        (df_center3['mall_display_name'] == site) &
        (df_center3['custom_end_date'] == "not yet end") &
        (df_center3['helpdesk_ticket_tag_id'] != 3) &
        (
            df_center3['priority'].isna() |
            (df_center3['priority'].astype(str).str.strip() == '0') |
            (df_center3['priority'].fillna(0).astype(int) == 0) |
            (df_center3['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_center3[
            (df_center3['mall_display_name'] == site) &
            (df_center3['custom_end_date'] == "not yet end") &
            (df_center3['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_center3,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_center3 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_center3[col].sum() if df_sites_center3[col].dtype != 'O' else 'TOTAL' for col in df_sites_center3.columns}
df_sites_center3 = pd.concat([df_sites_center3, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_center3.columns if col != 'Sites']
df_no_total = df_sites_center3.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_center3) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_center3.style.apply(lambda s: apply_color_scale(df_sites_center3), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_center3) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_center3.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 4 & NINH BINH - Le Anh Sinh</h2>", unsafe_allow_html=True)
df_center4 = df[df['team_id'] == 20]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho Center 4 & Ninh Binh
value = pivot.loc['CENTER 4 & NINH BINH - Le Anh Sinh', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_center4['create_date'] <= end_w1) &
    (
        (df_center4['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_center4[mask_w1].shape[0]

mask_w = (
    (df_center4['create_date'] <= end_w) &
    (
        (df_center4['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_center4[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_center4[(df_center4['create_date'] >= start) & (df_center4['create_date'] <= end)].shape[0]
    solved = -df_center4[(pd.to_datetime(df_center4['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 4 & NINH BINH - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho Center 4 & Ninh Binh
category_names_center4 = df_center4['category_name'].dropna().unique()
table_data_center4 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_center4:
        mask = (
            (df_center4['category_name'] == cat) &
            (df_center4['create_date'] <= end) &
            (
                (df_center4['custom_end_date'] == "not yet end") |
                (
                    (df_center4['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_center4[mask].shape[0]
        row[cat] = count
    table_data_center4.append(row)
df_table_center4 = pd.DataFrame(table_data_center4)

fig_stack_center4 = go.Figure()
for cat in category_names_center4:
    y_values = df_table_center4[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_center4.add_trace(go.Bar(
        name=cat,
        x=df_table_center4["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_center4[category_names_center4].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_center4.add_trace(go.Scatter(
    x=df_table_center4["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_center4["Tuần"], totals_offset, totals)):
    fig_stack_center4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_center4["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_center4:
    count_w = float(df_table_center4[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_center4[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_center4["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_center4.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_center4.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_center4.update_layout(
    barmode='stack',
    title=dict(
        text="CENTER 4 & NINH BINH - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_center4)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho Center 4 & Ninh Binh
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_center4 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_center4['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_center4['priority'].isna()) |
            (df_center4['priority'].astype(str).str.strip() == '0') |
            (df_center4['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center4['create_date'] <= end) &
        (
            (df_center4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_center4[mask_low].shape[0]
    mask_medium = (
        (df_center4['helpdesk_ticket_tag_id'] != 3) &
        (df_center4['priority'].fillna(0).astype(int) == 2) &
        (df_center4['create_date'] <= end) &
        (
            (df_center4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_center4[mask_medium].shape[0]
    mask_high = (
        (df_center4['helpdesk_ticket_tag_id'] != 3) &
        (df_center4['priority'].fillna(0).astype(int) == 3) &
        (df_center4['create_date'] <= end) &
        (
            (df_center4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_center4[mask_high].shape[0]
    mask_emergency = (
        (df_center4['helpdesk_ticket_tag_id'] == 3) &
        (df_center4['create_date'] <= end) &
        (
            (df_center4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_center4[mask_emergency].shape[0]
    table_data_priority_center4.append(row)
df_table_priority_center4 = pd.DataFrame(table_data_priority_center4)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_center4["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_center4[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_center4[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_center4 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_center4[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_center4.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_center4["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_center4["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_center4.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_center4.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_center4[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_center4.add_trace(go.Scatter(
    x=df_table_priority_center4["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_center4["Tuần"], totals_offset, totals)):
    fig_stack_priority_center4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_center4.update_layout(
    barmode='stack',
    title={
        'text': "CENTER 4 & NINH BINH - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_center4)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_center4[
        (df_center4['create_date'] >= start) &
        (df_center4['create_date'] <= end) &
        (df_center4['priority'].fillna(0).astype(int) == 3) &
        (df_center4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_center4[
        (df_center4['create_date'] >= start) &
        (df_center4['create_date'] <= end) &
        (df_center4['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_center4[
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') <= end) &
        (df_center4['priority'].fillna(0).astype(int) == 3) &
        (df_center4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_center4[
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') <= end) &
        (df_center4['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 4 & NINH BINH - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_center4[
        (df_center4['create_date'] >= start) &
        (df_center4['create_date'] <= end) &
        (
            df_center4['priority'].isna() |
            (df_center4['priority'].astype(str).str.strip() == '0') |
            (df_center4['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_center4[
        (df_center4['create_date'] >= start) &
        (df_center4['create_date'] <= end) &
        (df_center4['priority'].fillna(0).astype(int) == 2)
        &
        (df_center4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_center4[
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') <= end) &
        (
            df_center4['priority'].isna() |
            (df_center4['priority'].astype(str).str.strip() == '0') |
            (df_center4['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_center4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_center4[
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') <= end) &
        (df_center4['priority'].fillna(0).astype(int) == 2)
        &
        (df_center4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "CENTER 4 & NINH BINH - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho Center 4 & Ninh Binh
st.markdown("<h3 style='text-align: center;'>CENTER 4 & NINH BINH - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs Lotte Vinh (2763M5)", "CBS Crocs Vincom Thanh Hoa (2763O7)", "GO Mall Thanh Hoa (THA)",
    "GO Mall Vinh (VIN)", "Hyper Ninh Binh (NBH)", "Hyper Thanh Hoa (THA)",
    "Hyper Vinh (VIN)", "KUBO NANO Ninh Binh (6416)", "KUBO NANO Thanh Hoa (6446)",
    "KUBO NANO Vinh (6407)", "Nguyen Kim Nghe An (NA01)", "Nguyen Kim Thanh Hoa (TH01)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_center4['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_center4 = df_special_sites['Sites'].tolist()

for site in sites_center4:
    count_not_end = df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['create_date'] <= seven_days_ago) &
        (df_center4['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['create_date'] <= seven_days_ago) &
        (df_center4['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['create_date'] <= seventy_days_ago) &
        (df_center4['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['create_date'] <= seventy_days_ago) &
        (df_center4['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_center4['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['custom_end_date'] == "not yet end") &
        (df_center4['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['custom_end_date'] == "not yet end") &
        (df_center4['helpdesk_ticket_tag_id'] != 3) &
        (df_center4['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['custom_end_date'] == "not yet end") &
        (df_center4['helpdesk_ticket_tag_id'] != 3) &
        (df_center4['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_center4[
        (df_center4['mall_display_name'] == site) &
        (df_center4['custom_end_date'] == "not yet end") &
        (df_center4['helpdesk_ticket_tag_id'] != 3) &
        (
            df_center4['priority'].isna() |
            (df_center4['priority'].astype(str).str.strip() == '0') |
            (df_center4['priority'].fillna(0).astype(int) == 0) |
            (df_center4['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_center4[
            (df_center4['mall_display_name'] == site) &
            (df_center4['custom_end_date'] == "not yet end") &
            (df_center4['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_center4,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_center4 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_center4[col].sum() if df_sites_center4[col].dtype != 'O' else 'TOTAL' for col in df_sites_center4.columns}
df_sites_center4 = pd.concat([df_sites_center4, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_center4.columns if col != 'Sites']
df_no_total = df_sites_center4.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_center4) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_center4.style.apply(lambda s: apply_color_scale(df_sites_center4), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_center4) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_center4.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>SOUTH 1 - Nguyen Duc Tuan</h2>", unsafe_allow_html=True)
df_south1 = df[df['team_id'] == 21]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho South 1
value = pivot.loc['SOUTH 1 - Nguyen Duc Tuan', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_south1['create_date'] <= end_w1) &
    (
        (df_south1['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_south1[mask_w1].shape[0]

mask_w = (
    (df_south1['create_date'] <= end_w) &
    (
        (df_south1['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_south1[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_south1[(df_south1['create_date'] >= start) & (df_south1['create_date'] <= end)].shape[0]
    solved = -df_south1[(pd.to_datetime(df_south1['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 1 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho South 1
category_names_south1 = df_south1['category_name'].dropna().unique()
table_data_south1 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_south1:
        mask = (
            (df_south1['category_name'] == cat) &
            (df_south1['create_date'] <= end) &
            (
                (df_south1['custom_end_date'] == "not yet end") |
                (
                    (df_south1['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_south1[mask].shape[0]
        row[cat] = count
    table_data_south1.append(row)
df_table_south1 = pd.DataFrame(table_data_south1)

fig_stack_south1 = go.Figure()
for cat in category_names_south1:
    y_values = df_table_south1[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_south1.add_trace(go.Bar(
        name=cat,
        x=df_table_south1["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_south1[category_names_south1].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_south1.add_trace(go.Scatter(
    x=df_table_south1["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_south1["Tuần"], totals_offset, totals)):
    fig_stack_south1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_south1["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_south1:
    count_w = float(df_table_south1[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_south1[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_south1["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_south1.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_south1.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_south1.update_layout(
    barmode='stack',
    title=dict(
        text="SOUTH 1 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_south1)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho South 1
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_south1 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_south1['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_south1['priority'].isna()) |
            (df_south1['priority'].astype(str).str.strip() == '0') |
            (df_south1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south1['create_date'] <= end) &
        (
            (df_south1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_south1[mask_low].shape[0]
    mask_medium = (
        (df_south1['helpdesk_ticket_tag_id'] != 3) &
        (df_south1['priority'].fillna(0).astype(int) == 2) &
        (df_south1['create_date'] <= end) &
        (
            (df_south1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_south1[mask_medium].shape[0]
    mask_high = (
        (df_south1['helpdesk_ticket_tag_id'] != 3) &
        (df_south1['priority'].fillna(0).astype(int) == 3) &
        (df_south1['create_date'] <= end) &
        (
            (df_south1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_south1[mask_high].shape[0]
    mask_emergency = (
        (df_south1['helpdesk_ticket_tag_id'] == 3) &
        (df_south1['create_date'] <= end) &
        (
            (df_south1['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_south1[mask_emergency].shape[0]
    table_data_priority_south1.append(row)
df_table_priority_south1 = pd.DataFrame(table_data_priority_south1)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_south1["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_south1[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_south1[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_south1 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_south1[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_south1.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_south1["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_south1["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_south1.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_south1.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_south1[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_south1.add_trace(go.Scatter(
    x=df_table_priority_south1["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_south1["Tuần"], totals_offset, totals)):
    fig_stack_priority_south1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_south1.update_layout(
    barmode='stack',
    title={
        'text': "SOUTH 1 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_south1)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_south1[
        (df_south1['create_date'] >= start) &
        (df_south1['create_date'] <= end) &
        (df_south1['priority'].fillna(0).astype(int) == 3) &
        (df_south1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_south1[
        (df_south1['create_date'] >= start) &
        (df_south1['create_date'] <= end) &
        (df_south1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_south1[
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') <= end) &
        (df_south1['priority'].fillna(0).astype(int) == 3) &
        (df_south1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_south1[
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') <= end) &
        (df_south1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 1 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_south1[
        (df_south1['create_date'] >= start) &
        (df_south1['create_date'] <= end) &
        (
            df_south1['priority'].isna() |
            (df_south1['priority'].astype(str).str.strip() == '0') |
            (df_south1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_south1[
        (df_south1['create_date'] >= start) &
        (df_south1['create_date'] <= end) &
        (df_south1['priority'].fillna(0).astype(int) == 2)
        &
        (df_south1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_south1[
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') <= end) &
        (
            df_south1['priority'].isna() |
            (df_south1['priority'].astype(str).str.strip() == '0') |
            (df_south1['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_south1[
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') <= end) &
        (df_south1['priority'].fillna(0).astype(int) == 2)
        &
        (df_south1['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 1 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho South 1
st.markdown("<h3 style='text-align: center;'>SOUTH 1 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs Can Tho (2763P3)", "CBS Crocs Vincom Xuan Khanh (2763T9)", "GO Mall Can Tho (CTO)",
    "Hyper Bac Lieu (BLU)", "Hyper Can Tho (CTO)", "KUBO NANO Bac Lieu (6453)",
    "KUBO NANO Ca Mau (6433)", "KUBO NANO Can Tho (6413)", "mini go! Rach Gia (1507)",
    "Nguyen Kim Bac Lieu (BL01)", "Nguyen Kim Ca Mau (CM01)", "Nguyen Kim Can Tho (CT01)",
    "Nguyen Kim Kien Giang (KG01)", "Nguyen Kim Long Xuyen (AG01)", "Nguyen Kim Vinh Long (VL01)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_south1['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_south1 = df_special_sites['Sites'].tolist()

for site in sites_south1:
    count_not_end = df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['create_date'] <= seven_days_ago) &
        (df_south1['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['create_date'] <= seven_days_ago) &
        (df_south1['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['create_date'] <= seventy_days_ago) &
        (df_south1['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['create_date'] <= seventy_days_ago) &
        (df_south1['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south1['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['custom_end_date'] == "not yet end") &
        (df_south1['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['custom_end_date'] == "not yet end") &
        (df_south1['helpdesk_ticket_tag_id'] != 3) &
        (df_south1['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['custom_end_date'] == "not yet end") &
        (df_south1['helpdesk_ticket_tag_id'] != 3) &
        (df_south1['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_south1[
        (df_south1['mall_display_name'] == site) &
        (df_south1['custom_end_date'] == "not yet end") &
        (df_south1['helpdesk_ticket_tag_id'] != 3) &
        (
            df_south1['priority'].isna() |
            (df_south1['priority'].astype(str).str.strip() == '0') |
            (df_south1['priority'].fillna(0).astype(int) == 0) |
            (df_south1['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_south1[
            (df_south1['mall_display_name'] == site) &
            (df_south1['custom_end_date'] == "not yet end") &
            (df_south1['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_south1,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_south1 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_south1[col].sum() if df_sites_south1[col].dtype != 'O' else 'TOTAL' for col in df_sites_south1.columns}
df_sites_south1 = pd.concat([df_sites_south1, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_south1.columns if col != 'Sites']
df_no_total = df_sites_south1.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_south1) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_south1.style.apply(lambda s: apply_color_scale(df_sites_south1), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_south1) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_south1.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>SOUTH 2 - Nguyen Huu Hau</h2>", unsafe_allow_html=True)
df_south2 = df[df['team_id'] == 5]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho South 2
value = pivot.loc['SOUTH 2 - Nguyen Huu Hau', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_south2['create_date'] <= end_w1) &
    (
        (df_south2['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_south2[mask_w1].shape[0]

mask_w = (
    (df_south2['create_date'] <= end_w) &
    (
        (df_south2['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_south2[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_south2[(df_south2['create_date'] >= start) & (df_south2['create_date'] <= end)].shape[0]
    solved = -df_south2[(pd.to_datetime(df_south2['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 2 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho South 2
category_names_south2 = df_south2['category_name'].dropna().unique()
table_data_south2 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_south2:
        mask = (
            (df_south2['category_name'] == cat) &
            (df_south2['create_date'] <= end) &
            (
                (df_south2['custom_end_date'] == "not yet end") |
                (
                    (df_south2['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_south2[mask].shape[0]
        row[cat] = count
    table_data_south2.append(row)
df_table_south2 = pd.DataFrame(table_data_south2)

fig_stack_south2 = go.Figure()
for cat in category_names_south2:
    y_values = df_table_south2[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_south2.add_trace(go.Bar(
        name=cat,
        x=df_table_south2["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_south2[category_names_south2].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_south2.add_trace(go.Scatter(
    x=df_table_south2["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_south2["Tuần"], totals_offset, totals)):
    fig_stack_south2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_south2["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_south2:
    count_w = float(df_table_south2[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_south2[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_south2["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_south2.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_south2.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_south2.update_layout(
    barmode='stack',
    title=dict(
        text="SOUTH 2 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_south2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho South 2
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_south2 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_south2['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_south2['priority'].isna()) |
            (df_south2['priority'].astype(str).str.strip() == '0') |
            (df_south2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south2['create_date'] <= end) &
        (
            (df_south2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_south2[mask_low].shape[0]
    mask_medium = (
        (df_south2['helpdesk_ticket_tag_id'] != 3) &
        (df_south2['priority'].fillna(0).astype(int) == 2) &
        (df_south2['create_date'] <= end) &
        (
            (df_south2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_south2[mask_medium].shape[0]
    mask_high = (
        (df_south2['helpdesk_ticket_tag_id'] != 3) &
        (df_south2['priority'].fillna(0).astype(int) == 3) &
        (df_south2['create_date'] <= end) &
        (
            (df_south2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_south2[mask_high].shape[0]
    mask_emergency = (
        (df_south2['helpdesk_ticket_tag_id'] == 3) &
        (df_south2['create_date'] <= end) &
        (
            (df_south2['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_south2[mask_emergency].shape[0]
    table_data_priority_south2.append(row)
df_table_priority_south2 = pd.DataFrame(table_data_priority_south2)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_south2["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_south2[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_south2[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_south2 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_south2[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_south2.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_south2["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_south2["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_south2.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_south2.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_south2[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_south2.add_trace(go.Scatter(
    x=df_table_priority_south2["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_south2["Tuần"], totals_offset, totals)):
    fig_stack_priority_south2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_south2.update_layout(
    barmode='stack',
    title={
        'text': "SOUTH 2 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_south2)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_south2[
        (df_south2['create_date'] >= start) &
        (df_south2['create_date'] <= end) &
        (df_south2['priority'].fillna(0).astype(int) == 3) &
        (df_south2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_south2[
        (df_south2['create_date'] >= start) &
        (df_south2['create_date'] <= end) &
        (df_south2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_south2[
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') <= end) &
        (df_south2['priority'].fillna(0).astype(int) == 3) &
        (df_south2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_south2[
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') <= end) &
        (df_south2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 2 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_south2[
        (df_south2['create_date'] >= start) &
        (df_south2['create_date'] <= end) &
        (
            df_south2['priority'].isna() |
            (df_south2['priority'].astype(str).str.strip() == '0') |
            (df_south2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_south2[
        (df_south2['create_date'] >= start) &
        (df_south2['create_date'] <= end) &
        (df_south2['priority'].fillna(0).astype(int) == 2)
        &
        (df_south2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_south2[
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') <= end) &
        (
            df_south2['priority'].isna() |
            (df_south2['priority'].astype(str).str.strip() == '0') |
            (df_south2['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_south2[
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') <= end) &
        (df_south2['priority'].fillna(0).astype(int) == 2)
        &
        (df_south2['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 2 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho South 2
st.markdown("<h3 style='text-align: center;'>SOUTH 2 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs Aeon Binh Duong (2763E5)", "CBS Crocs Go Ba Ria (2763H4)", "CBS Crocs Vincom Bien Hoa (2763G2)",
    "CBS Crocs Vung Tau (2763P4)", "CBS Dyson Glam Beautique Aeon Binh Duong (2763CI)", "CBS Dyson Nguyen Kim Binh Duong (2763BA)",
    "CBS Fila Aeon Binh Duong (2763G3)", "GO Mall Ba Ria (BRA)", "GO Mall Dong Nai (DNI)",
    "GO Mall Tan Hiep (THP)", "Hyper Ba Ria (BRA)",
    "Hyper Binh Duong (BDG)", "Hyper Dong Nai (DNI)", "Hyper Tan Hiep (THP)",
    "KUBO NANO Ba Ria (6430)", "KUBO NANO Binh Duong (6410)", "KUBO NANO Dong Nai (6424)",
    "KUBO NANO Nhon Trach (6442)", "KUBO NANO Tay Ninh - Go Dau (6436)", "KUBO NANO Tay Ninh - Hoa Thanh (6447)",
    "mini go! Go Dau (1501)", "mini go! Hoa Thanh (1506)", "mini go! Loc Ninh (1513)",
    "mini go! Nhon Trach (1503)", "mini go! Phu My (1502)", "mini go! Tan Uyen (1504)",
    "Nguyen Kim Ba Ria (VT02)", "Nguyen Kim Bien Hoa (BH01)", "Nguyen Kim Binh Duong (BD01)",
    "Nguyen Kim Dong Nai (BH02)", "Nguyen Kim Trang Bom (BH04)", "Nguyen Kim Vung Tau (VT01)"

]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_south2['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_south2 = df_special_sites['Sites'].tolist()

for site in sites_south2:
    count_not_end = df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['create_date'] <= seven_days_ago) &
        (df_south2['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['create_date'] <= seven_days_ago) &
        (df_south2['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['create_date'] <= seventy_days_ago) &
        (df_south2['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['create_date'] <= seventy_days_ago) &
        (df_south2['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south2['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['custom_end_date'] == "not yet end") &
        (df_south2['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['custom_end_date'] == "not yet end") &
        (df_south2['helpdesk_ticket_tag_id'] != 3) &
        (df_south2['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['custom_end_date'] == "not yet end") &
        (df_south2['helpdesk_ticket_tag_id'] != 3) &
        (df_south2['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_south2[
        (df_south2['mall_display_name'] == site) &
        (df_south2['custom_end_date'] == "not yet end") &
        (df_south2['helpdesk_ticket_tag_id'] != 3) &
        (
            df_south2['priority'].isna() |
            (df_south2['priority'].astype(str).str.strip() == '0') |
            (df_south2['priority'].fillna(0).astype(int) == 0) |
            (df_south2['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_south2[
            (df_south2['mall_display_name'] == site) &
            (df_south2['custom_end_date'] == "not yet end") &
            (df_south2['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_south2,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_south2 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_south2[col].sum() if df_sites_south2[col].dtype != 'O' else 'TOTAL' for col in df_sites_south2.columns}
df_sites_south2 = pd.concat([df_sites_south2, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_south2.columns if col != 'Sites']
df_no_total = df_sites_south2.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_south2) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_south2.style.apply(lambda s: apply_color_scale(df_sites_south2), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_south2) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_south2.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>SOUTH 3 - Mai Thanh Long</h2>", unsafe_allow_html=True)
df_south3 = df[df['team_id'] == 4]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho South 3
value = pivot.loc['SOUTH 3 - Mai Thanh Long', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_south3['create_date'] <= end_w1) &
    (
        (df_south3['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_south3[mask_w1].shape[0]

mask_w = (
    (df_south3['create_date'] <= end_w) &
    (
        (df_south3['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_south3[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_south3[(df_south3['create_date'] >= start) & (df_south3['create_date'] <= end)].shape[0]
    solved = -df_south3[(pd.to_datetime(df_south3['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 3 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho South 3
category_names_south3 = df_south3['category_name'].dropna().unique()
table_data_south3 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_south3:
        mask = (
            (df_south3['category_name'] == cat) &
            (df_south3['create_date'] <= end) &
            (
                (df_south3['custom_end_date'] == "not yet end") |
                (
                    (df_south3['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_south3[mask].shape[0]
        row[cat] = count
    table_data_south3.append(row)
df_table_south3 = pd.DataFrame(table_data_south3)

fig_stack_south3 = go.Figure()
for cat in category_names_south3:
    y_values = df_table_south3[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_south3.add_trace(go.Bar(
        name=cat,
        x=df_table_south3["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_south3[category_names_south3].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_south3.add_trace(go.Scatter(
    x=df_table_south3["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_south3["Tuần"], totals_offset, totals)):
    fig_stack_south3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_south3["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_south3:
    count_w = float(df_table_south3[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_south3[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_south3["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_south3.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_south3.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_south3.update_layout(
    barmode='stack',
    title=dict(
        text="SOUTH 3 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_south3)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho South 3
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_south3 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_south3['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_south3['priority'].isna()) |
            (df_south3['priority'].astype(str).str.strip() == '0') |
            (df_south3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south3['create_date'] <= end) &
        (
            (df_south3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_south3[mask_low].shape[0]
    mask_medium = (
        (df_south3['helpdesk_ticket_tag_id'] != 3) &
        (df_south3['priority'].fillna(0).astype(int) == 2) &
        (df_south3['create_date'] <= end) &
        (
            (df_south3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_south3[mask_medium].shape[0]
    mask_high = (
        (df_south3['helpdesk_ticket_tag_id'] != 3) &
        (df_south3['priority'].fillna(0).astype(int) == 3) &
        (df_south3['create_date'] <= end) &
        (
            (df_south3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_south3[mask_high].shape[0]
    mask_emergency = (
        (df_south3['helpdesk_ticket_tag_id'] == 3) &
        (df_south3['create_date'] <= end) &
        (
            (df_south3['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_south3[mask_emergency].shape[0]
    table_data_priority_south3.append(row)
df_table_priority_south3 = pd.DataFrame(table_data_priority_south3)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_south3["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_south3[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_south3[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_south3 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_south3[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_south3.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_south3["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_south3["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_south3.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_south3.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_south3[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_south3.add_trace(go.Scatter(
    x=df_table_priority_south3["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_south3["Tuần"], totals_offset, totals)):
    fig_stack_priority_south3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_south3.update_layout(
    barmode='stack',
    title={
        'text': "SOUTH 3 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_south3)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_south3[
        (df_south3['create_date'] >= start) &
        (df_south3['create_date'] <= end) &
        (df_south3['priority'].fillna(0).astype(int) == 3) &
        (df_south3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_south3[
        (df_south3['create_date'] >= start) &
        (df_south3['create_date'] <= end) &
        (df_south3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_south3[
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') <= end) &
        (df_south3['priority'].fillna(0).astype(int) == 3) &
        (df_south3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_south3[
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') <= end) &
        (df_south3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 3 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_south3[
        (df_south3['create_date'] >= start) &
        (df_south3['create_date'] <= end) &
        (
            df_south3['priority'].isna() |
            (df_south3['priority'].astype(str).str.strip() == '0') |
            (df_south3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_south3[
        (df_south3['create_date'] >= start) &
        (df_south3['create_date'] <= end) &
        (df_south3['priority'].fillna(0).astype(int) == 2)
        &
        (df_south3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_south3[
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') <= end) &
        (
            df_south3['priority'].isna() |
            (df_south3['priority'].astype(str).str.strip() == '0') |
            (df_south3['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_south3[
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') <= end) &
        (df_south3['priority'].fillna(0).astype(int) == 2)
        &
        (df_south3['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 3 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho South 3
st.markdown("<h3 style='text-align: center;'>SOUTH 3 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs Vincom My Tho (2763K4)", "GO Mall Ben Tre (BTE)", "GO Mall My Tho (MTO)",
    "GO Mall Tra Vinh (TVH)", "Hyper Ben Tre (BTE)", "Hyper My Tho (MTO)",
    "Hyper Tra Vinh (TVH)", "KUBO NANO Ben Tre (6420)", "KUBO NANO Hong Ngu (6449)",
    "KUBO NANO My Tho (6427)", "KUBO NANO Thanh Binh (6450)", "KUBO NANO Tra Vinh (6406)",
    "mini go! Hong Ngu (1508)", "mini go! Lap Vo (1512)", "mini go! Thanh Binh (1509)",
    "Nguyen Kim Ben Tre (BE01)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_south3['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_south3 = df_special_sites['Sites'].tolist()

for site in sites_south3:
    count_not_end = df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['create_date'] <= seven_days_ago) &
        (df_south3['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['create_date'] <= seven_days_ago) &
        (df_south3['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['create_date'] <= seventy_days_ago) &
        (df_south3['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['create_date'] <= seventy_days_ago) &
        (df_south3['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south3['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['custom_end_date'] == "not yet end") &
        (df_south3['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['custom_end_date'] == "not yet end") &
        (df_south3['helpdesk_ticket_tag_id'] != 3) &
        (df_south3['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['custom_end_date'] == "not yet end") &
        (df_south3['helpdesk_ticket_tag_id'] != 3) &
        (df_south3['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_south3[
        (df_south3['mall_display_name'] == site) &
        (df_south3['custom_end_date'] == "not yet end") &
        (df_south3['helpdesk_ticket_tag_id'] != 3) &
        (
            df_south3['priority'].isna() |
            (df_south3['priority'].astype(str).str.strip() == '0') |
            (df_south3['priority'].fillna(0).astype(int) == 0) |
            (df_south3['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_south3[
            (df_south3['mall_display_name'] == site) &
            (df_south3['custom_end_date'] == "not yet end") &
            (df_south3['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_south3,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_south3 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_south3[col].sum() if df_sites_south3[col].dtype != 'O' else 'TOTAL' for col in df_sites_south3.columns}
df_sites_south3 = pd.concat([df_sites_south3, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_south3.columns if col != 'Sites']
df_no_total = df_sites_south3.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_south3) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_south3.style.apply(lambda s: apply_color_scale(df_sites_south3), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_south3) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_south3.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>SOUTH 4 - Dang Thanh Danh</h2>", unsafe_allow_html=True)
df_south4 = df[df['team_id'] == 22]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho South 4
value = pivot.loc['SOUTH 4 - Dang Thanh Danh', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_south4['create_date'] <= end_w1) &
    (
        (df_south4['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_south4[mask_w1].shape[0]

mask_w = (
    (df_south4['create_date'] <= end_w) &
    (
        (df_south4['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_south4[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_south4[(df_south4['create_date'] >= start) & (df_south4['create_date'] <= end)].shape[0]
    solved = -df_south4[(pd.to_datetime(df_south4['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 4 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho South 4
category_names_south4 = df_south4['category_name'].dropna().unique()
table_data_south4 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_south4:
        mask = (
            (df_south4['category_name'] == cat) &
            (df_south4['create_date'] <= end) &
            (
                (df_south4['custom_end_date'] == "not yet end") |
                (
                    (df_south4['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_south4[mask].shape[0]
        row[cat] = count
    table_data_south4.append(row)
df_table_south4 = pd.DataFrame(table_data_south4)

fig_stack_south4 = go.Figure()
for cat in category_names_south4:
    y_values = df_table_south4[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_south4.add_trace(go.Bar(
        name=cat,
        x=df_table_south4["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_south4[category_names_south4].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_south4.add_trace(go.Scatter(
    x=df_table_south4["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_south4["Tuần"], totals_offset, totals)):
    fig_stack_south4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_south4["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_south4:
    count_w = float(df_table_south4[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_south4[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_south4["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_south4.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_south4.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_south4.update_layout(
    barmode='stack',
    title=dict(
        text="SOUTH 4 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_south4)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho South 4
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_south4 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_south4['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_south4['priority'].isna()) |
            (df_south4['priority'].astype(str).str.strip() == '0') |
            (df_south4['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south4['create_date'] <= end) &
        (
            (df_south4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_south4[mask_low].shape[0]
    mask_medium = (
        (df_south4['helpdesk_ticket_tag_id'] != 3) &
        (df_south4['priority'].fillna(0).astype(int) == 2) &
        (df_south4['create_date'] <= end) &
        (
            (df_south4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_south4[mask_medium].shape[0]
    mask_high = (
        (df_south4['helpdesk_ticket_tag_id'] != 3) &
        (df_south4['priority'].fillna(0).astype(int) == 3) &
        (df_south4['create_date'] <= end) &
        (
            (df_south4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_south4[mask_high].shape[0]
    mask_emergency = (
        (df_south4['helpdesk_ticket_tag_id'] == 3) &
        (df_south4['create_date'] <= end) &
        (
            (df_south4['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_south4[mask_emergency].shape[0]
    table_data_priority_south4.append(row)
df_table_priority_south4 = pd.DataFrame(table_data_priority_south4)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_south4["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_south4[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_south4[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_south4 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_south4[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_south4.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_south4["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_south4["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_south4.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_south4.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_south4[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_south4.add_trace(go.Scatter(
    x=df_table_priority_south4["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_south4["Tuần"], totals_offset, totals)):
    fig_stack_priority_south4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_south4.update_layout(
    barmode='stack',
    title={
        'text': "SOUTH 4 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_south4)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_south4[
        (df_south4['create_date'] >= start) &
        (df_south4['create_date'] <= end) &
        (df_south4['priority'].fillna(0).astype(int) == 3) &
        (df_south4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_south4[
        (df_south4['create_date'] >= start) &
        (df_south4['create_date'] <= end) &
        (df_south4['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_south4[
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') <= end) &
        (df_south4['priority'].fillna(0).astype(int) == 3) &
        (df_south4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_south4[
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') <= end) &
        (df_south4['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 4 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_south4[
        (df_south4['create_date'] >= start) &
        (df_south4['create_date'] <= end) &
        (
            df_south4['priority'].isna() |
            (df_south4['priority'].astype(str).str.strip() == '0') |
            (df_south4['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_south4[
        (df_south4['create_date'] >= start) &
        (df_south4['create_date'] <= end) &
        (df_south4['priority'].fillna(0).astype(int) == 2)
        &
        (df_south4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_south4[
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') <= end) &
        (
            df_south4['priority'].isna() |
            (df_south4['priority'].astype(str).str.strip() == '0') |
            (df_south4['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_south4[
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') <= end) &
        (df_south4['priority'].fillna(0).astype(int) == 2)
        &
        (df_south4['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 4 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho South 4
st.markdown("<h3 style='text-align: center;'>SOUTH 4 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    # Thay đổi danh sách site cho South 4 nếu cần
    "CBS Columbia Crescent Mall (2763N8)", "CBS Crocs Aeon Binh Tan (276344)", "CBS Crocs Aeon Tan Phu (2763C8)",
    "CBS Crocs Crescent Mall (276330)", "CBS Crocs Hai Ba Trung (276337)", "CBS Crocs Lotte District 7 (2763Y9)",
    "CBS Crocs Parc Mall (2763U2)", "CBS Crocs Vincom Dong Khoi (2763D9)", "CBS Dyson Aeon Tan Phu (2763BH)",
    "CBS Dyson Cellphones Tran Quang Khai Quan 1 (2763AZ)", "CBS Dyson Crescent Mall (2763BG)", "CBS Dyson Diamond Plaza (2763AY)",
    "CBS Dyson Vincom Dong Khoi (2763AK)", "CBS Fila Crescent Mall (2763C7)", "CBS FitFlop Aeon Mall Binh Tan (2763CG)",
    "CBS Fitflop Crescent Mall (2763S9)", "CBS Fitflop Lotte Mart District 7 (2763CF)", "CBS Fitflop Vincom Dong Khoi (2763T3)",
    "CBS Supersports Robins Crescent Mall (276320)", "CBS Under Armour Crescent Mall (2763H3)", "GO Mall An Lac (ALC)",
    "GO Mall Au Co (ACO)", "GO Mall Nguyen Thi Thap (NTT)", "GO Mall Phu Thanh (PTH)",
    "Hyper An Lac (ALC)", "Hyper Mien Dong (MDG)", "Hyper Nguyen Thi Thap (NTT)",
    "Hyper Phu Thanh (PTH)", "KUBO NANO An Lac (6403)", "KUBO NANO Nguyen Thi Thap (6415)",
    "KUBO Premium Lotte Cong Hoa (6448)", "Nguyen Kim An Lac (SG15)", "Nguyen Kim Binh Tan (SG18)",
    "Nguyen Kim Cao Thang Mall (CTM)", "Nguyen Kim Lac Long Quan (A003)", "Nguyen Kim Quan 7 (SG21)",
    "Nguyen Kim Sai Gon (SG01)", "Tops Au Co (ACO)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_south4['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_south4 = df_special_sites['Sites'].tolist()

for site in sites_south4:
    count_not_end = df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['create_date'] <= seven_days_ago) &
        (df_south4['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['create_date'] <= seven_days_ago) &
        (df_south4['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['create_date'] <= seventy_days_ago) &
        (df_south4['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['create_date'] <= seventy_days_ago) &
        (df_south4['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south4['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['custom_end_date'] == "not yet end") &
        (df_south4['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['custom_end_date'] == "not yet end") &
        (df_south4['helpdesk_ticket_tag_id'] != 3) &
        (df_south4['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['custom_end_date'] == "not yet end") &
        (df_south4['helpdesk_ticket_tag_id'] != 3) &
        (df_south4['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_south4[
        (df_south4['mall_display_name'] == site) &
        (df_south4['custom_end_date'] == "not yet end") &
        (df_south4['helpdesk_ticket_tag_id'] != 3) &
        (
            df_south4['priority'].isna() |
            (df_south4['priority'].astype(str).str.strip() == '0') |
            (df_south4['priority'].fillna(0).astype(int) == 0) |
            (df_south4['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_south4[
            (df_south4['mall_display_name'] == site) &
            (df_south4['custom_end_date'] == "not yet end") &
            (df_south4['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_south4,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_south4 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_south4[col].sum() if df_sites_south4[col].dtype != 'O' else 'TOTAL' for col in df_sites_south4.columns}
df_sites_south4 = pd.concat([df_sites_south4, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_south4.columns if col != 'Sites']
df_no_total = df_sites_south4.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_south4) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_south4.style.apply(lambda s: apply_color_scale(df_sites_south4), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_south4) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_south4.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>SOUTH 5 - Nguyen Ngoc Ho</h2>", unsafe_allow_html=True)
df_south5 = df[df['team_id'] == 23]
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

# Pivot cho gauge
pivot = pd.pivot_table(
    df,
    values='processing_time',
    index='team_name',
    columns='category_name',
    aggfunc='mean',
    fill_value=0
)
across_all = df.groupby('team_name')['processing_time'].mean().round(0).astype(int)
pivot.insert(0, 'Across all category', across_all)
pivot = pivot.round(0).astype(int)

# Lấy giá trị cho South 5
value = pivot.loc['SOUTH 5 - Nguyen Ngoc Ho', 'Across all category']

gauge_max = 100
gauge_min = 0
level1 = 33
level2 = 66

steps = []
if value > 0:
    steps.append({'range': [0, min(value, level1)], 'color': '#b7f7b7'})
if value > level1:
    steps.append({'range': [level1, min(value, level2)], 'color': '#ffe082'})
if value > level2:
    steps.append({'range': [level2, min(value, gauge_max)], 'color': '#ffb3b3'})
if value < gauge_max:
    steps.append({'range': [value, gauge_max], 'color': '#eeeeee'})

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=value,
    gauge={
        'axis': {'range': [gauge_min, gauge_max]},
        'bar': {'color': 'rgba(0,0,0,0)'},
        'steps': steps,
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))

fig_gauge.update_layout(
    annotations=[
        dict(
            x=0.5, y=0.01,
            text="(days)",
            showarrow=False,
            font=dict(size=22, color="gray"),
            xanchor="center"
        )
    ],
    width=350, height=250,
    margin=dict(l=10, r=10, t=40, b=10),
)

# Tính số lượng ticket tồn tuần trước (W-1) và tuần hiện tại (W)
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
end_w = week_ends[idx_w]
end_w1 = week_ends[idx_w1]

mask_w1 = (
    (df_south5['create_date'] <= end_w1) &
    (
        (df_south5['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end_w1)
    )
)
count_w1 = df_south5[mask_w1].shape[0]

mask_w = (
    (df_south5['create_date'] <= end_w) &
    (
        (df_south5['custom_end_date'] == "not yet end") |
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end_w)
    )
)
count_w = df_south5[mask_w].shape[0]

if count_w1 == 0:
    percent = 100 if count_w > 0 else 0
else:
    percent = ((count_w - count_w1) / count_w1) * 100

if percent > 0:
    percent_text = f"W vs W-1: +{percent:.1f}%"
    bgcolor = "#f2c795"
elif percent < 0:
    percent_text = f"W vs W-1: -{abs(percent):.1f}%"
    bgcolor = "#abf3ab"
else:
    percent_text = "W vs W-1: 0.0%"
    bgcolor = "#f2c795"

percent_value = f"{percent:+.1f}%" if percent != 0 else "0.0%"

col1, col2 = st.columns([1, 0.9])
with col1:
    st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style='display: flex; justify-content: center; margin-bottom: 2rem;'>
            <div style='padding: 0.5rem 1.2rem; background: {bgcolor}; border: 2px solid #888; border-radius: 10px; font-size: 1.1rem; font-weight: bold; color: #222; min-width: 180px; text-align: center;'>
                <div style='font-size:1.7rem; font-weight: bold;'>W vs W-1</div>
                <div style='font-size:1.3rem; font-weight: bold; margin-top: 0.2rem;'>{percent_value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        """
        <div style='text-align:left; font-size:1.5rem; font-weight:bold; margin-bottom: 1.5rem; margin-left: 35px;'>
            Avg. Processing Time<br>Across All Category
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div style='margin-left: 40px;'>", unsafe_allow_html=True)
    st.plotly_chart(fig_gauge)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height: 7rem'></div>", unsafe_allow_html=True)

# Clustered column chart: Created vs Solved ticket per week
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created = df_south5[(df_south5['create_date'] >= start) & (df_south5['create_date'] <= end)].shape[0]
    solved = -df_south5[(pd.to_datetime(df_south5['custom_end_date'], errors='coerce') >= start) & (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') <= end)].shape[0]
    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 5 - ON ASSESSMENT TICKET OVER WEEKS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Category cho South 5
category_names_south5 = df_south5['category_name'].dropna().unique()
table_data_south5 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    for cat in category_names_south5:
        mask = (
            (df_south5['category_name'] == cat) &
            (df_south5['create_date'] <= end) &
            (
                (df_south5['custom_end_date'] == "not yet end") |
                (
                    (df_south5['custom_end_date'] != "not yet end") &
                    (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end)
                )
            )
        )
        count = df_south5[mask].shape[0]
        row[cat] = count
    table_data_south5.append(row)
df_table_south5 = pd.DataFrame(table_data_south5)

fig_stack_south5 = go.Figure()
for cat in category_names_south5:
    y_values = df_table_south5[cat].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_south5.add_trace(go.Bar(
        name=cat,
        x=df_table_south5["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
    ))
totals = df_table_south5[category_names_south5].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_south5.add_trace(go.Scatter(
    x=df_table_south5["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_south5["Tuần"], totals_offset, totals)):
    fig_stack_south5.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

# % thay đổi giữa tuần hiện tại và tuần trước cho từng category
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_south5["Tuần"].iloc[idx_w]

active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
for cat in category_names_south5:
    count_w = float(df_table_south5[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_south5[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_categories.append(cat)
    percent_changes[cat] = percent
    category_positions[cat] = cumulative_height + count_w / 2
    cumulative_height += count_w

if active_categories:
    total_height = cumulative_height
    x_vals = list(df_table_south5["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_south5.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_south5.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )

fig_stack_south5.update_layout(
    barmode='stack',
    title=dict(
        text="SOUTH 5 - OVERALL EVOLUTION OA TICKETS PER CATEGORY",
        y=1,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=850,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.45,
        xanchor="left",
        x=0
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black')),
        automargin=False
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    ),
    margin=dict(r=50, b=5),
)
st.plotly_chart(fig_stack_south5)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Stacked Bar Chart theo Priority cho South 5
priority_cols = ['Low priority', 'Medium priority', 'High priority', 'Emergency']
priority_colors = {
    'Low priority': '#b7f7b7',
    'Medium priority': '#fff9b1',
    'High priority': '#ffd6a0',
    'Emergency': '#ff2222'
}
table_data_priority_south5 = []
for i, end in enumerate(week_ends):
    row = {"Tuần": week_labels[i]}
    mask_low = (
        (df_south5['helpdesk_ticket_tag_id'] != 3) &
        (
            (df_south5['priority'].isna()) |
            (df_south5['priority'].astype(str).str.strip() == '0') |
            (df_south5['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south5['create_date'] <= end) &
        (
            (df_south5['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Low priority'] = df_south5[mask_low].shape[0]
    mask_medium = (
        (df_south5['helpdesk_ticket_tag_id'] != 3) &
        (df_south5['priority'].fillna(0).astype(int) == 2) &
        (df_south5['create_date'] <= end) &
        (
            (df_south5['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Medium priority'] = df_south5[mask_medium].shape[0]
    mask_high = (
        (df_south5['helpdesk_ticket_tag_id'] != 3) &
        (df_south5['priority'].fillna(0).astype(int) == 3) &
        (df_south5['create_date'] <= end) &
        (
            (df_south5['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end)
        )
    )
    row['High priority'] = df_south5[mask_high].shape[0]
    mask_emergency = (
        (df_south5['helpdesk_ticket_tag_id'] == 3) &
        (df_south5['create_date'] <= end) &
        (
            (df_south5['custom_end_date'] == "not yet end") |
            (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > end)
        )
    )
    row['Emergency'] = df_south5[mask_emergency].shape[0]
    table_data_priority_south5.append(row)
df_table_priority_south5 = pd.DataFrame(table_data_priority_south5)

# % thay đổi giữa tuần hiện tại và tuần trước cho từng priority
idx_w = len(week_ends) - 1
idx_w1 = idx_w - 1
w_label = df_table_priority_south5["Tuần"].iloc[idx_w]
active_priorities = []
percent_changes = {}
priority_positions = {}
cumulative_height = 0
for pri in priority_cols:
    count_w = float(df_table_priority_south5[pri].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = float(df_table_priority_south5[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_priorities.append(pri)
    percent_changes[pri] = percent
    priority_positions[pri] = cumulative_height + count_w / 2
    cumulative_height += count_w

fig_stack_priority_south5 = go.Figure()
for priority in priority_cols:
    y_values = df_table_priority_south5[priority].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority_south5.add_trace(go.Bar(
        name=priority,
        x=df_table_priority_south5["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=priority_colors[priority]
    ))
if active_priorities:
    total_height = cumulative_height
    x_vals = list(df_table_priority_south5["Tuần"])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 2
    sorted_priorities = sorted(active_priorities, key=lambda x: priority_positions[x])
    for i, pri in enumerate(sorted_priorities):
        percent = percent_changes[pri]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = priority_positions[pri]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_priorities)/2))
        fig_stack_priority_south5.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_priority_south5.add_annotation(
            x=x_offset, y=y_box,
            text=f"<b>{percent_text}</b>",
            showarrow=False,
            font=dict(size=11, color="black"),
            align="left",
            xanchor="left",
            yanchor="middle",
            bgcolor=bgcolor,
            borderpad=3,
            bordercolor="black",
            borderwidth=1
        )
totals = df_table_priority_south5[priority_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
fig_stack_priority_south5.add_trace(go.Scatter(
    x=df_table_priority_south5["Tuần"],
    y=totals_offset,
    textposition="top center",
    textfont=dict(size=16),
    showlegend=False,
    hoverinfo="skip",
    texttemplate="%{text}"
))
for i, (x, y, t) in enumerate(zip(df_table_priority_south5["Tuần"], totals_offset, totals)):
    fig_stack_priority_south5.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=10, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_priority_south5.update_layout(
    barmode='stack',
    title={
        'text': "SOUTH 5 - OVERALL EVOLUTION OA TICKETS PER PRIORITY",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of OA Tickets",
    width=1400,
    height=850,
    legend=dict(orientation="h", yanchor="top", y=1.05, xanchor="center", x=0.5),
    xaxis=dict(tickfont=dict(color='black')),
    yaxis=dict(tickfont=dict(color='black'))
)
st.plotly_chart(fig_stack_priority_south5)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket High Priority (Emergency & Non-Emergency)
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_high_non_emergency = df_south5[
        (df_south5['create_date'] >= start) &
        (df_south5['create_date'] <= end) &
        (df_south5['priority'].fillna(0).astype(int) == 3) &
        (df_south5['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_high_emergency = df_south5[
        (df_south5['create_date'] >= start) &
        (df_south5['create_date'] <= end) &
        (df_south5['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    created = created_high_non_emergency + created_high_emergency

    solved_high_non_emergency = df_south5[
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') <= end) &
        (df_south5['priority'].fillna(0).astype(int) == 3) &
        (df_south5['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_high_emergency = df_south5[
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') <= end) &
        (df_south5['helpdesk_ticket_tag_id'] == 3)
    ].shape[0]
    solved = -(solved_high_non_emergency + solved_high_emergency)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 5 - OVERALL EVOLUTION EMERGENCY & HIGH PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Clustered Chart: Created/Solved ticket Low & Medium Priority 
created_counts = []
solved_counts = []
for start, end in zip(week_starts, week_ends):
    created_low = df_south5[
        (df_south5['create_date'] >= start) &
        (df_south5['create_date'] <= end) &
        (
            df_south5['priority'].isna() |
            (df_south5['priority'].astype(str).str.strip() == '0') |
            (df_south5['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south5['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created_medium = df_south5[
        (df_south5['create_date'] >= start) &
        (df_south5['create_date'] <= end) &
        (df_south5['priority'].fillna(0).astype(int) == 2)
        &
        (df_south5['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    created = created_low + created_medium

    solved_low = df_south5[
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') <= end) &
        (
            df_south5['priority'].isna() |
            (df_south5['priority'].astype(str).str.strip() == '0') |
            (df_south5['priority'].fillna(0).astype(int) == 1)
        ) &
        (df_south5['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved_medium = df_south5[
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') <= end) &
        (df_south5['priority'].fillna(0).astype(int) == 2)
        &
        (df_south5['helpdesk_ticket_tag_id'] != 3)
    ].shape[0]
    solved = -(solved_low + solved_medium)

    created_counts.append(created)
    solved_counts.append(solved)

fig = go.Figure(data=[
    go.Bar(
        name='Created ticket',
        x=week_labels,
        y=created_counts,
        marker_color='#ffb3b3',
        text=[str(v) if v != 0 else "" for v in created_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    ),
    go.Bar(
        name='Solved ticket',
        x=week_labels,
        y=solved_counts,
        marker_color='#b7f7b7',
        text=[str(v) if v != 0 else "" for v in solved_counts],
        textposition="outside",
        textfont=dict(size=14, color="black")
    )
])
fig.update_layout(
    barmode='group',
    title={
        'text': "SOUTH 5 - OVERALL EVOLUTION MEDIUM & LOW PRIORITY TICKETS",
        'y': 1,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    xaxis_title="Weeks",
    yaxis_title="Number of Tickets",
    width=1200,
    height=750,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black')
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# Bảng Sites cho South 5
st.markdown("<h3 style='text-align: center;'>SOUTH 5 - DETAIL VIEW PER SITE</h3>", unsafe_allow_html=True)
special_display_names = [
    
    "CBS Crocs 61 Nguyen Trai (276329)", "CBS Crocs Cao Thang (276339)", "CBS Crocs Estella (276366)",
    "CBS Crocs Giga Mall Thu Duc (2763P2)", "CBS Crocs Go Di An (2763H9)", "CBS Crocs Mac Thi Buoi (2763O9)",
    "CBS Crocs Phan Van Tri (2763R9)", "CBS Crocs Saigon Centre (2763K1)", "CBS Crocs Thiso Mall (2763U3)",
    "CBS Crocs Van Hanh Mall (276310)", "CBS Crocs Vincom Grand Park (2763T5)", "CBS Crocs Vincom Landmark 81 (2763Z8)",
    "CBS Crocs Vincom Thao Dien (276311)", "CBS Crocs Vincom Thu Duc (2763T6)", "CBS Dyson Beauty Box Saigon Center (2763CL)",
    "CBS Dyson Estella (2763K9)", "CBS Dyson Landmark 81 (2763AV)", "CBS Dyson Maison Online (2763CM)",
    "CBS Dyson Nguyen Kim Go Vap (2763M2)", "CBS Dyson Nguyen Kim Sai Gon 1 (2763AD)", "CBS Dyson Nguyen Kim Thu Duc (2763M1)",
    "CBS Dyson Takashimaya (2763AH)", "CBS Dyson Thiso Mall (2763CH)", "CBS Fila Saigon Centre (2763U1)",
    "CBS Fila Vincom Landmark 81 (2763H2)", "CBS Fitflop Estella (2763Y1)", "CBS Fitflop Takashimaya (2763S5)",
    "CBS Hoka Saigon Centre (2763X6)", "CBS Supersports Estella (276307)", "CBS Supersports Le Van Sy (2763X7)",
    "CBS Supersports Takashimaya (2763C6)", "CBS Under Armour Vincom Thao Dien (2763Q5)", "GO Mall Di An (DAN)",
    "GO Mall SSC (SSC)", "GO Mall Truong Chinh (TCH)", "Hyper Di An (DAN)",
    "Hyper Go Vap (GVP)", "Hyper Truong Chinh (TCH)", "KUBO NANO Di An (6401)",
    "KUBO NANO Thang Loi (6431)", "KUBO Premium Lotte Phu Tho (6443)", "Nguyen Kim Binh Thanh (SG27)",
    "Nguyen Kim Cu Chi (SG29)", "Nguyen Kim Di An (BD02)", "Nguyen Kim Go Vap (SG09)",
    "Nguyen Kim Le Van Viet (SG30)", "Nguyen Kim Nguyen Duy Trinh (SG17)", "Nguyen Kim Phan Van Hon (SG22)",
    "Nguyen Kim Phu Nhuan (SG10)", "Nguyen Kim Quang Trung (SG19)", "Nguyen Kim Sai Gon Mall (SGM)",
    "Nguyen Kim Thu Duc (SG06)", "Nguyen Kim Thuan An (BD03)", "Nguyen Kim Truong Chinh (SG37)",
    "Tops An Phu (APU)", "Tops Moonlight (MLT)", "Tops Thao Dien (TDN)"
]

df_res_partner['display_name'] = df_res_partner['display_name'].astype(str)
if 'is_company' in df_res_partner.columns:
    mask_company = (df_res_partner['is_company'] == True) | (df_res_partner['is_company'] == 1)
else:
    mask_company = True

df_special_sites = df_res_partner[
    df_res_partner['display_name'].isin(special_display_names)
    & mask_company
    & (df_res_partner['helpdesk_team_id'] != 12)
    & (df_res_partner['helpdesk_team_id'] != 25)
    & (df_res_partner['active'] == True)
][['display_name', 'mall_code']].drop_duplicates().sort_values('display_name')

df_special_sites = df_special_sites.rename(columns={'display_name': 'Sites', 'mall_code': 'Mall Code'})

today = pd.Timestamp.now().normalize()
seven_days_ago = today - pd.Timedelta(days=7)
seventy_days_ago = today - pd.Timedelta(days=70)

site_ticket_not_end = []
site_ticket_7days = []
site_ticket_70days = []
site_ticket_emergency = []
site_ticket_high_priority = []
site_ticket_medium_priority = []
site_ticket_low_priority = []

category_list = df_south5['category_name'].dropna().unique()[:11]
site_ticket_by_category = {cat: [] for cat in category_list}

sites_south5 = df_special_sites['Sites'].tolist()

for site in sites_south5:
    count_not_end = df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['custom_end_date'] == "not yet end")
    ].shape[0]
    site_ticket_not_end.append(count_not_end)

    count_old_not_end_7 = df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['create_date'] <= seven_days_ago) &
        (df_south5['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_7 = df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['create_date'] <= seven_days_ago) &
        (df_south5['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > seven_days_ago)
    ].shape[0]
    site_ticket_7days.append(count_not_end - (count_old_not_end_7 + count_old_end_late_7))

    count_old_not_end_70 = df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['create_date'] <= seventy_days_ago) &
        (df_south5['custom_end_date'] == "not yet end")
    ].shape[0]
    count_old_end_late_70 = df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['create_date'] <= seventy_days_ago) &
        (df_south5['custom_end_date'] != "not yet end") &
        (pd.to_datetime(df_south5['custom_end_date'], errors='coerce') > seventy_days_ago)
    ].shape[0]
    site_ticket_70days.append(count_not_end - (count_old_not_end_70 + count_old_end_late_70))

    site_ticket_emergency.append(df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['custom_end_date'] == "not yet end") &
        (df_south5['helpdesk_ticket_tag_id'] == 3)
    ].shape[0])

    site_ticket_high_priority.append(df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['custom_end_date'] == "not yet end") &
        (df_south5['helpdesk_ticket_tag_id'] != 3) &
        (df_south5['priority'].fillna(0).astype(int) == 3)
    ].shape[0])

    site_ticket_medium_priority.append(df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['custom_end_date'] == "not yet end") &
        (df_south5['helpdesk_ticket_tag_id'] != 3) &
        (df_south5['priority'].fillna(0).astype(int) == 2)
    ].shape[0])

    site_ticket_low_priority.append(df_south5[
        (df_south5['mall_display_name'] == site) &
        (df_south5['custom_end_date'] == "not yet end") &
        (df_south5['helpdesk_ticket_tag_id'] != 3) &
        (
            df_south5['priority'].isna() |
            (df_south5['priority'].astype(str).str.strip() == '0') |
            (df_south5['priority'].fillna(0).astype(int) == 0) |
            (df_south5['priority'].fillna(0).astype(int) == 1)
        )
    ].shape[0])

    for cat in category_list:
        site_ticket_by_category[cat].append(df_south5[
            (df_south5['mall_display_name'] == site) &
            (df_south5['custom_end_date'] == "not yet end") &
            (df_south5['category_name'] == cat)
        ].shape[0])

data = {
    'Sites': sites_south5,
    'Total OA tickets': site_ticket_not_end,
    'Vs last 7 days': site_ticket_7days,
    'Vs last 70 days': site_ticket_70days,
    'Emergency OA': site_ticket_emergency,
    'High priority OA': site_ticket_high_priority,
    'Medium priority OA': site_ticket_medium_priority,
    'Low priority OA': site_ticket_low_priority,
}
for cat in category_list:
    data[cat] = site_ticket_by_category[cat]

df_sites_south5 = pd.DataFrame(data)

# Thêm hàng Total (sum các cột số)
total_row = {col: df_sites_south5[col].sum() if df_sites_south5[col].dtype != 'O' else 'TOTAL' for col in df_sites_south5.columns}
df_sites_south5 = pd.concat([df_sites_south5, pd.DataFrame([total_row])], ignore_index=True)

# Conditional formatting 3-Color Scale (chỉ áp dụng cho các hàng, không áp dụng cho hàng Total)
num_cols = [col for col in df_sites_south5.columns if col != 'Sites']
df_no_total = df_sites_south5.iloc[:-1][num_cols]
vmin = df_no_total.min().min()
vmax = df_no_total.max().max()
vmid = df_no_total.stack().quantile(0.5)  # 50th percentile

def color_scale(val):
    try:
        val = float(val)
    except:
        return ""
    if vmax == vmin:
        norm = 0.5
    elif val <= vmid:
        norm = (val - vmin) / (vmid - vmin) / 2 if vmid > vmin else 0
    else:
        norm = 0.5 + (val - vmid) / (vmax - vmid) / 2 if vmax > vmid else 1
    # Xanh lá nhạt (#b7f7b7) -> trắng (#ffffff) -> đỏ nhạt (#ffb3b3)
    if norm <= 0.5:
        r = int(183 + (255-183)*norm*2)
        g = int(247 + (255-247)*norm*2)
        b = int(183 + (255-183)*norm*2)
    else:
        r = int(255)
        g = int(255 - (255-179)*(norm-0.5)*2)
        b = int(255 - (255-179)*(norm-0.5)*2)
    return f'background-color: rgb({r},{g},{b})'

def style_func(val, row_idx):
    # Không tô màu cho hàng Total (hàng cuối)
    if row_idx == len(df_sites_south5) - 1:
        return ""
    return color_scale(val)

def apply_color_scale(df):
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    for row_idx in range(len(df)):
        if row_idx == len(df) - 1:
            continue
        for col in num_cols:
            styled.at[row_idx, col] = color_scale(df.at[row_idx, col])
    return styled

styled = df_sites_south5.style.apply(lambda s: apply_color_scale(df_sites_south5), axis=None)

# Format hàng Total: màu đỏ, in đậm
def highlight_total(s):
    is_total = s.name == len(df_sites_south5) - 1
    return ['font-weight: bold; color: red;' if is_total else '' for _ in s]

styled = styled.apply(highlight_total, axis=1)

num_rows = df_sites_south5.shape[0]
row_height = 35
header_height = 38
st.dataframe(styled, use_container_width=True, height=num_rows * row_height + header_height)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)
