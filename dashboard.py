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

st_autorefresh(interval=86400000, key="datarefresh")

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
    query = "SELECT * FROM helpdesk_ticket LIMIT 50"
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

# 9. Giao diện
page = st.sidebar.radio("Chọn trang", ["Dashboard", "Xem dữ liệu", "North 1", "North 2", "Others"])

if page == "Dashboard":
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
        ),
        width=1200,
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
        ),
        width=1200,
        height=900,
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
        ),
        width=1200,
        height=900,
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
        ),
        width=1200,
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

    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        name="Created",
        x=week_labels,
        y=created_counts,
        measure=["relative"] * len(week_labels),
        text=[str(c) for c in created_counts],
        textposition="outside",
        increasing={"marker": {"color": "#e74c3c"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        connector={"line": {"color": "rgba(0,0,0,0)"}},
    ))
    solved_counts_negative = [-s for s in solved_counts]
    fig.add_trace(go.Waterfall(
        name="Solved",
        x=week_labels,
        y=solved_counts_negative,
        measure=["relative"] * len(week_labels),
        text=[str(s) for s in solved_counts_negative],
        textposition="outside",
        increasing={"marker": {"color": "#27ae60"}},
        decreasing={"marker": {"color": "#27ae60"}},
        connector={"line": {"color": "rgba(0,0,0,0)"}},
    ))
    fig.update_layout(
        title=dict(
            text="NATIONWIDE ON ASSESSMENT TICKET OVER WEEKS",
            x=0.5,
            xanchor='center',
            yanchor='top',
        ),
        waterfallgap=0.2,
        barmode='group',
        width=1200,
        height=700,
        xaxis=dict(
            tickfont=dict(color='black'),
            title=dict(text="Weeks", font=dict(color='black'))
        ),
        yaxis=dict(
            tickfont=dict(color='black'),
            title=dict(text="Number of OA Tickets", font=dict(color='black'))
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

    

elif page == "Xem dữ liệu":
    st.title("Xem dữ liệu và cột tính mới")
    
    # Hiển thị thông tin thời gian
    st.markdown("""
    <style>
    .date-info {
        padding: 20px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='date-info'>
        <h3>Thông tin thời gian:</h3>
        <p><strong>Start date:</strong> {start_date.strftime('%d-%m-%y %H:%M:%S')}</p>
        <p><strong>End date:</strong> {end_date.strftime('%d-%m-%y %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("Ngày create_date mới nhất trong dữ liệu:", df['create_date'].max())

    st.write("Dữ liệu với cột tính mới:")
    AgGrid(df[['id','number','stage_id','approved_date','last_stage_update','custom_end_date','category_id','category_name','team_id','team_name','priority','helpdesk_ticket_tag_id','mall_id','mall_display_name','processing_time','Under this month report','Carry over ticket']].head(100), key="main_table")
    
    with st.expander("Bảng dữ liệu mẫu helpdesk_ticket_category"):
        st.dataframe(df_category.head(20))
    with st.expander("Bảng dữ liệu mẫu helpdesk_ticket_team"):
        st.dataframe(df_team.head(20))
    with st.expander("Bảng dữ liệu mẫu res_partner"):
        st.dataframe(df_res_partner)

    with st.expander("Bảng dữ liệu mẫu helpdesk_ticket"):
        df_helpdesk_ticket = load_helpdesk_ticket()
        st.dataframe(df_helpdesk_ticket)
    
    st.write("Bảng kiểm tra số lượng ticket tồn theo Category và Tuần:")
    AgGrid(df_table.head(50), key="table_category")
    
    st.write("Bảng kiểm tra số lượng ticket tồn theo TEAM và Tuần:")
    AgGrid(df_table_team.head(50), key="table_team")

    st.write("Bảng kiểm tra số lượng ticket tồn theo PRIORITY và Tuần:")
    AgGrid(df_table_priority.head(50), key="table_priority")

    st.write("Bảng kiểm tra số lượng ticket tồn theo BANNER và Tuần:")
    AgGrid(df_table_banner.head(50), key="table_banner")

    st.write("### Bảng kiểm tra số lượng Created và Solved ticket theo tuần:")
    st.dataframe(result_df)

    with st.expander("Danh sách các bảng trong database"):
        connection = psycopg2.connect(
            host=host, database=database, user=user, password=password
        )
        query = '''
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        '''
        df_tables = pd.read_sql(query, connection)
        connection.close()
        st.dataframe(df_tables)

    df['custom_end_date'] = df['custom_end_date'].fillna("not yet end")

elif page == "North 1":
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
        width=1200,
        height=600,
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
        width=1200,
        height=900,
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
        width=1200,
        height=800,
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
        width=1200,
        height=900,
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
        width=1200,
        height=800,
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
        width=1200,
        height=900,
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
        width=1200,
        height=800,
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
