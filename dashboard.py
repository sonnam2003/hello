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
import requests
from io import BytesIO
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

host = "27.71.237.112"
database = "pbreport"
user = "pbuser"
password = "p0w3rb!"

#Auto refresh mỗi 1h để làm mới cache data
#1 phút = 60 giây = 60.000 ms
#1 giờ = 60 phút = 3.600.000 ms

st_autorefresh(interval=10800000, key="datarefresh")  # 1 tiếng = 3.600.000 ms

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
@st.cache_data  # Cache trong 1 tiếng (3600 giây)
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

@st.cache_data  # Cache trong 1 tiếng (3600 giây)
def load_category():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, name FROM helpdesk_ticket_category"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data  # Cache trong 1 tiếng (3600 giây)
def load_team():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, name FROM helpdesk_ticket_team"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data  # Cache trong 1 tiếng (3600 giây)
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

@st.cache_data  # Cache trong 1 tiếng (3600 giây)
def load_res_partner():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, display_name, create_date, function, street, email, phone, mobile, active, helpdesk_team_id, mall_code, is_company FROM res_partner"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data  # Cache trong 1 tiếng (3600 giây)
def load_res_partner_display_name():
    connection = psycopg2.connect(
        host=host, database=database, user=user, password=password
    )
    query = "SELECT id, display_name FROM res_partner"
    df = pd.read_sql(query, connection)
    connection.close()
    return df

@st.cache_data  # Cache trong 1 tiếng (3600 giây)
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


# Ngày bắt đầu gốc (thứ 2, 19/05/2025)
base_start = datetime(2025, 5, 19)

# Lấy ngày hôm nay (hoặc bạn có thể thay bằng ngày bất kỳ để test)
today = datetime.today()

# Tìm thứ 2 gần nhất trước hoặc bằng hôm nay
def get_monday(d):
    return d - timedelta(days=d.weekday())

# Tìm tuần hiện tại so với base_start
current_monday = get_monday(today)
weeks_since_base = (current_monday - base_start).days // 7

# Nếu chưa đến tuần base_start thì vẫn lấy từ base_start
if weeks_since_base < 0:
    weeks_since_base = 0

# Tạo danh sách 10 tuần, kết thúc ở tuần hiện tại
start_week_index = max(0, weeks_since_base - 9)
week_starts = [base_start + timedelta(weeks=i) for i in range(start_week_index, weeks_since_base + 1)]
week_ends = [start + timedelta(days=6, hours=23, minutes=59, seconds=59) for start in week_starts]

# Tạo nhãn tuần với số tuần trong năm (ISO week number)
week_labels = [
    f"W{start.isocalendar()[1]} ({start.strftime('%d/%m')} - {end.strftime('%d/%m')})"
    for start, end in zip(week_starts, week_ends)
]

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
# WATERFALL CHART - Tính bảng kiểm tra số lượng Created và Solved ticket theo tuần (dùng cho cả hai trang)
created_counts = []
solved_counts = []
waterfall_week_labels = []

# Số tuần sẽ tự động lấy theo độ dài của week_starts (hoặc week_labels)
num_weeks = len(week_starts)

for i in range(num_weeks):
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


# --- Đọc dữ liệu từ file Excel online (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online(excel_url, sheet_name, usecols, skiprows, nrows):
    response = requests.get(excel_url)
    excel_data = BytesIO(response.content)
    df_excel = pd.read_excel(excel_data, sheet_name=sheet_name, usecols=usecols, skiprows=skiprows, nrows=nrows)
    return df_excel

excel_url = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name = "VISUALIZE (fin) by weeks"
usecols = "BF:BQ"
skiprows = 7
nrows = 10
try:
    df_excel = load_excel_online(excel_url, sheet_name, usecols, skiprows, nrows)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online: {e}")



# --- Đọc dữ liệu từ file Excel online thứ 2 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online2(excel_url2, sheet_name2, usecols2, skiprows2, nrows2):
    response2 = requests.get(excel_url2)
    excel_data2 = BytesIO(response2.content)
    df_excel2 = pd.read_excel(excel_data2, sheet_name=sheet_name2, usecols=usecols2, skiprows=skiprows2, nrows=nrows2)
    return df_excel2

excel_url2 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name2 = "VISUALIZE (fin) by weeks"
usecols2 = "BW:CI"
skiprows2 = 7
nrows2 = 10
try:
    df_excel2 = load_excel_online2(excel_url2, sheet_name2, usecols2, skiprows2, nrows2)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 2: {e}")



# --- Đọc dữ liệu từ file Excel online thứ 3 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online3(excel_url3, sheet_name3, usecols3, skiprows3, nrows3):
    response3 = requests.get(excel_url3)
    excel_data3 = BytesIO(response3.content)
    df_excel3 = pd.read_excel(excel_data3, sheet_name=sheet_name3, usecols=usecols3, skiprows=skiprows3, nrows=nrows3)
    return df_excel3

excel_url3 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name3 = "VISUALIZE (fin) by weeks"
usecols3 = "DD:DK"
skiprows3 = 10
nrows3 = 10
try:
    df_excel3 = load_excel_online3(excel_url3, sheet_name3, usecols3, skiprows3, nrows3)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 3: {e}")

# --- Đọc dữ liệu từ file Excel online thứ 4 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online4(excel_url4, sheet_name4, usecols4, skiprows4, nrows4):
    response4 = requests.get(excel_url4)
    excel_data4 = BytesIO(response4.content)
    df_excel4 = pd.read_excel(excel_data4, sheet_name=sheet_name4, usecols=usecols4, skiprows=skiprows4, nrows=nrows4)
    return df_excel4

excel_url4 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name4 = "VISUALIZE (fin) by weeks"
usecols4 = "EF:EN"
skiprows4 = 28
nrows4 = 26
try:
    df_excel4 = load_excel_online4(excel_url4, sheet_name4, usecols4, skiprows4, nrows4)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 4: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 5 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online5(excel_url5, sheet_name5, usecols5, skiprows5, nrows5):
    response5 = requests.get(excel_url5)
    excel_data5 = BytesIO(response5.content)
    df_excel5 = pd.read_excel(excel_data5, sheet_name=sheet_name5, usecols=usecols5, skiprows=skiprows5, nrows=nrows5)
    return df_excel5

excel_url5 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name5 = "VISUALIZE (fin) by months"
usecols5 = "BF:BQ"
skiprows5 = 7
nrows5 = 6
try:
    df_excel5 = load_excel_online5(excel_url5, sheet_name5, usecols5, skiprows5, nrows5)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 5: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 6 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online6(excel_url6, sheet_name6, usecols6, skiprows6, nrows6):
    response6 = requests.get(excel_url6)
    excel_data6 = BytesIO(response6.content)
    df_excel6 = pd.read_excel(excel_data6, sheet_name=sheet_name6, usecols=usecols6, skiprows=skiprows6, nrows=nrows6)
    return df_excel6

excel_url6 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name6 = "VISUALIZE (fin) by months"
usecols6 = "BY:CK"
skiprows6 = 7
nrows6 = 6
try:
    df_excel6 = load_excel_online6(excel_url6, sheet_name6, usecols6, skiprows6, nrows6)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 6: {e}")

# --- Đọc dữ liệu từ file Excel online thứ 7 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online7(excel_url7, sheet_name7, usecols7, skiprows7, nrows7):
    response7 = requests.get(excel_url7)
    excel_data7 = BytesIO(response7.content)
    df_excel7 = pd.read_excel(excel_data7, sheet_name=sheet_name7, usecols=usecols7, skiprows=skiprows7, nrows=nrows7)
    return df_excel7

excel_url7 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name7 = "VISUALIZE (fin) by months"
usecols7 = "DF:DM"
skiprows7 = 10
nrows7 = 6
try:
    df_excel7 = load_excel_online7(excel_url7, sheet_name7, usecols7, skiprows7, nrows7)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 7: {e}")

# --- Đọc dữ liệu từ file Excel online thứ 8 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online8(excel_url8, sheet_name8, usecols8, skiprows8, nrows8):
    response8 = requests.get(excel_url8)
    excel_data8 = BytesIO(response8.content)
    df_excel8 = pd.read_excel(excel_data8, sheet_name=sheet_name8, usecols=usecols8, skiprows=skiprows8, nrows=nrows8)
    return df_excel8

excel_url8 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name8 = "Actual cost per cat"
usecols8 = "Y:AK"
skiprows8 = 3
nrows8 = 11
try:
    df_excel8 = load_excel_online8(excel_url8, sheet_name8, usecols8, skiprows8, nrows8)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 8: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 9 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online9(excel_url9, sheet_name9, usecols9, skiprows9, nrows9):
    response9 = requests.get(excel_url9)
    excel_data9 = BytesIO(response9.content)
    df_excel9 = pd.read_excel(excel_data9, sheet_name=sheet_name9, usecols=usecols9, skiprows=skiprows9, nrows=nrows9)
    return df_excel9

excel_url9 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name9 = "Actual cost per cat"
usecols9 = "Y:AK"
skiprows9 = 44
nrows9 = 12
try:
    df_excel9 = load_excel_online9(excel_url9, sheet_name9, usecols9, skiprows9, nrows9)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 9: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 10 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online10(excel_url10, sheet_name10, usecols10, skiprows10, nrows10):
    response10 = requests.get(excel_url10)
    excel_data10 = BytesIO(response10.content)
    df_excel10 = pd.read_excel(excel_data10, sheet_name=sheet_name10, usecols=usecols10, skiprows=skiprows10, nrows=nrows10)
    return df_excel10

excel_url10 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name10 = "Actual cost per sub region"
usecols10 = "AA:AM"
skiprows10 = 3
nrows10 = 13
try:
    df_excel10 = load_excel_online10(excel_url10, sheet_name10, usecols10, skiprows10, nrows10)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 10: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 11 (OneDrive/SharePoint link chia sẻ) ---
@st.cache_data(show_spinner=True)  # Cache vĩnh viễn - update khi reboot
def load_excel_online11(excel_url11, sheet_name11, usecols11, skiprows11, nrows11):
    response11 = requests.get(excel_url11)
    excel_data11 = BytesIO(response11.content)
    df_excel11 = pd.read_excel(excel_data11, sheet_name=sheet_name11, usecols=usecols11, skiprows=skiprows11, nrows=nrows11)
    return df_excel11

excel_url11 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name11 = "Actual cost per sub region"
usecols11 = "AA:AM"
skiprows11 = 45
nrows11 = 13
try:
    df_excel11 = load_excel_online11(excel_url11, sheet_name11, usecols11, skiprows11, nrows11)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 11: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 12---
@st.cache_data(show_spinner=True) 
def load_excel_online12(excel_url12, sheet_name12, usecols12, skiprows12, nrows12):
    response12 = requests.get(excel_url12)
    excel_data12 = BytesIO(response12.content)
    df_excel12 = pd.read_excel(excel_data12, sheet_name=sheet_name12, usecols=usecols12, skiprows=skiprows12, nrows=nrows12)
    return df_excel12

excel_url12 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name12 = "Top 10 Equip Fre"
usecols12 = "G:L"
skiprows12 = 0
nrows12 = 10
try:
    df_excel12 = load_excel_online12(excel_url12, sheet_name12, usecols12, skiprows12, nrows12)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 12: {e}")


# --- Đọc dữ liệu từ file Excel online thứ 13---
@st.cache_data(show_spinner=True) 
def load_excel_online13(excel_url13, sheet_name13, usecols13, skiprows13, nrows13):
    response13 = requests.get(excel_url13)
    excel_data13 = BytesIO(response13.content)
    df_excel13 = pd.read_excel(excel_data13, sheet_name=sheet_name13, usecols=usecols13, skiprows=skiprows13, nrows=nrows13)
    return df_excel13

excel_url13 = "https://1drv.ms/x/c/982465afa38d44b6/EbHU7h-HDBlOrFY5xavC3JMBqdy9mzqsPhIMVyQWJ8AL3Q?e=8Vy150&download=1"
sheet_name13 = "bud by month"
usecols13 = "A:M"
skiprows13 = 28
nrows13 = 15
try:
    df_excel13 = load_excel_online13(excel_url13, sheet_name13, usecols13, skiprows13, nrows13)
except Exception as e:
    st.warning(f"Không thể đọc dữ liệu từ file Excel online thứ 13: {e}")


toc_html = """
<style>
#toc-float {
    position: fixed;
    top: 30px;
    left: 18px;
    z-index: 9999;
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(231,76,60,0.10), 0 1.5px 8px rgba(44,62,80,0.08);
    padding: 0;
    width: 60px;
    height: 60px;
    transition: width 0.35s, height 0.35s, opacity 0.3s;
    overflow: hidden;
    opacity: 0.85;
    border: none;
}
#toc-float:hover {
    width: 340px;
    height: 780px;
    min-height: 780px;
    opacity: 1;
    overflow: visible;
    box-shadow: 0 8px 32px rgba(231,76,60,0.18), 0 2px 12px rgba(44,62,80,0.10);
}
#toc-float .toc-icon {
    font-size: 38px;
    color: #e74c3c;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60px;
    width: 60px;
    transition: opacity 0.2s;
}
#toc-float:hover .toc-icon {
    opacity: 0;
    pointer-events: none;
}
#toc-float .toc-title {
    font-weight: 700;
    font-size: 22px;
    color: #e74c3c;
    margin: 18px 0 12px 18px;
    display: none;
    letter-spacing: 1px;
    font-family: 'Segoe UI', Arial, sans-serif;
}
#toc-float:hover .toc-title {
    display: block;
}
#toc-float ul {
    list-style: none;
    padding: 0 0 0 10px;
    margin: 0;
    display: none;
}
#toc-float:hover ul {
    display: block;
}
#toc-float li {
    margin-bottom: 6px;
    border-radius: 8px;
    transition: background 0.18s;
}
#toc-float a {
    display: block;
    color: #222;
    text-decoration: none;
    font-size: 17px;
    font-weight: 500;
    padding: 8px 18px 8px 18px;
    border-radius: 8px;
    transition: background 0.18s, color 0.18s;
    font-family: 'Segoe UI', Arial, sans-serif;
    position: relative;
}
#toc-float a:hover, #toc-float li:hover > a {
    background: #ffeaea;
    color: #e74c3c;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(231,76,60,0.07);
}
#toc-float a:before {
    content: "›";
    color: #e74c3c;
    font-size: 15px;
    margin-right: 10px;
    opacity: 0.7;
    transition: opacity 0.18s;
}
#toc-float a:hover:before, #toc-float li:hover > a:before {
    opacity: 1;
}
</style>
<div id="toc-float">
    <div class="toc-icon">☰</div>
    <div class="toc-title">TABLE OF CONTENTS</div>
    <ul>
        <li><a href="#overview">Overview</a></li>
        <li><a href="#north1">North 1</a></li>
        <li><a href="#north2">North 2</a></li>
        <li><a href="#north3">North 3</a></li>
        <li><a href="#center1">Center 1</a></li>
        <li><a href="#center2">Center 2</a></li>
        <li><a href="#center3">Center 3</a></li>
        <li><a href="#center4">Center 4</a></li>
        <li><a href="#south1">South 1</a></li>
        <li><a href="#south2">South 2</a></li>
        <li><a href="#south3">South 3</a></li>
        <li><a href="#south4">South 4</a></li>
        <li><a href="#south5">South 5</a></li>
    </ul>
</div>
<script>
document.querySelectorAll('#toc-float a').forEach(function(link) {
    link.onclick = function(e) {
        e.preventDefault();
        var id = this.getAttribute('href').substring(1);
        var el = document.getElementById(id);
        if (el) {
            el.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
    }
});
</script>
"""
st.markdown(toc_html, unsafe_allow_html=True)
st.markdown('<a id="overview"></a>', unsafe_allow_html=True)
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
    if priority == "Emergency":
        text_labels = [""] * len(y_values)  # Ẩn text mặc định
    else:
        text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_priority.add_trace(go.Bar(
        name=priority,
        x=df_table_priority["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=15),
        marker_color=priority_colors[priority]
    ))

# Thêm annotation ép font cho value "Emergency"
emergency_y = df_table_priority["Emergency"].tolist()
for i, (x, y) in enumerate(zip(df_table_priority["Tuần"], emergency_y)):
    if y != 0:
        # Tính vị trí y (giữa stack Emergency)
        y_stack = y / 2
        # Nếu có các stack phía dưới, cộng dồn lên
        for below_priority in priority_cols:
            if below_priority == "Emergency":
                break
            y_stack += df_table_priority[below_priority].iloc[i]
        fig_stack_priority.add_annotation(
            x=x,
            y=y_stack,
            text=f"<b>{int(y)}</b>",
            showarrow=False,
            font=dict(size=15, color="black"),  # Size lớn, màu trắng nổi bật trên nền đỏ
            align="center",
            xanchor="center",
            yanchor="middle",
            borderpad=2,
            bordercolor="#e74c3c",
            borderwidth=0,
            bgcolor="rgba(0,0,0,0)"  # Không nền
        )

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
    x_offset = x_idx + 1
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
        spacing_factor = 0.1
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
for i, (x, y, t) in enumerate(zip(df_table_priority["Tuần"], totals_offset, totals)):
    fig_stack_priority.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
        y=1.075,
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
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

# --- Tính X%, Z%, Y% và sinh câu mô tả tự động mới ---
idx_w = len(df_table_priority["Tuần"]) - 1
idx_w1 = idx_w - 1

# Tính phần trăm thay đổi cho từng priority
percent_changes = {}
for pri in priority_cols:
    count_w = float(df_table_priority[pri].iloc[idx_w])
    count_w1 = float(df_table_priority[pri].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes[pri] = percent

# X% là số âm lớn nhất (giảm nhiều nhất), A là tên priority đó
neg_percents = {k: v for k, v in percent_changes.items() if v < 0}
if neg_percents:
    A = min(neg_percents, key=neg_percents.get)
    X = neg_percents[A]
else:
    A = min(percent_changes, key=percent_changes.get)
    X = percent_changes[A]

# Z% là số dương lớn nhất (tăng nhiều nhất), B là tên priority đó
pos_percents = {k: v for k, v in percent_changes.items() if v > 0}
if pos_percents:
    B = max(pos_percents, key=pos_percents.get)
    Z = pos_percents[B]
else:
    B = max(percent_changes, key=percent_changes.get)
    Z = percent_changes[B]

# Y% là phần trăm thay đổi tổng số ticket của tất cả priority
sum_w = sum([df_table_priority[pri].iloc[idx_w] for pri in priority_cols])
sum_w1 = sum([df_table_priority[pri].iloc[idx_w1] for pri in priority_cols])
if sum_w1 == 0:
    Y = 100 if sum_w > 0 else 0
else:
    Y = ((sum_w - sum_w1) / sum_w1) * 100

# C là 'increased' nếu Y > 0, ngược lại 'decreased'
C = 'increased' if Y > 0 else 'decreased'

# Hiển thị câu mô tả tự động
A_html = f"<span style='color:#d62728; font-weight:bold'>{A}</span>"
B_html = f"<span style='color:#d62728; font-weight:bold'>{B}</span>"
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"

decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

# Đếm số lượng tăng/giảm/không đổi
change_values = list(percent_changes.values())
num_neg = sum(1 for v in change_values if v < 0)
num_pos = sum(1 for v in change_values if v > 0)
num_zero = sum(1 for v in change_values if abs(v) < 1e-6)

# Sinh câu mô tả theo logic mới
if num_neg == 1 and num_zero == len(priority_cols) - 1:
    # Chỉ có 1 giảm, còn lại không đổi
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while other priorities remains unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_zero == len(priority_cols) - 1:
    # Chỉ có 1 tăng, còn lại không đổi
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%, while other priorities remains unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == len(priority_cols):
    # Tất cả đều giảm
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == len(priority_cols):
    # Tất cả đều tăng
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_zero == len(priority_cols):
    # Tất cả không đổi
    description = f"All priorities remain unchanged compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
else:
    # Trường hợp mặc định như cũ
    if abs(X) < 1e-6:
        decrease_text = f"{A_html} remains unchanged"
    else:
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    if abs(Z) < 1e-6:
        increase_text = f"{B_html} remains unchanged"
    else:
        increase_text = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%"
    description = f"{decrease_text}, while {increase_text}, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"

st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
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
    x_offset = x_idx + 0.8
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
for i, (x, y, t) in enumerate(zip(df_table["Tuần"], totals_offset, totals)):
    fig_stack.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# --- Tính top 2 giảm/tăng mạnh nhất và sinh câu mô tả tự động cho Category ---
idx_w = len(df_table["Tuần"]) - 1
idx_w1 = idx_w - 1

# Tính phần trăm thay đổi cho từng category
percent_changes_cat = {}
for cat in category_names:
    count_w = float(df_table[cat].iloc[idx_w])
    count_w1 = float(df_table[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_cat[cat] = percent

neg_percents = sorted([(k, v) for k, v in percent_changes_cat.items() if v < 0], key=lambda x: x[1])
pos_percents = sorted([(k, v) for k, v in percent_changes_cat.items() if v > 0], key=lambda x: x[1], reverse=True)
num_neg = len(neg_percents)
num_pos = len(pos_percents)
num_zero = sum(1 for v in percent_changes_cat.values() if abs(v) < 1e-6)
n_cat = len(category_names)

# Y% là phần trăm thay đổi tổng số ticket của tất cả category
sum_w = sum([df_table[cat].iloc[idx_w] for cat in category_names])
sum_w1 = sum([df_table[cat].iloc[idx_w1] for cat in category_names])
if sum_w1 == 0:
    Y = 100 if sum_w > 0 else 0
else:
    Y = ((sum_w - sum_w1) / sum_w1) * 100
C = 'increased' if Y > 0 else 'decreased'

# Hiển thị câu mô tả tự động
A, X = neg_percents[0] if num_neg else (None, None)
O, K = neg_percents[1] if num_neg > 1 else (None, None)
B, Z = pos_percents[0] if num_pos else (None, None)
P, H = pos_percents[1] if num_pos > 1 else (None, None)
A_html = f"<span style='color:#d62728; font-weight:bold'>{A}</span>" if A else ""
O_html = f"<span style='color:#d62728; font-weight:bold'>{O}</span>" if O else ""
B_html = f"<span style='color:#d62728; font-weight:bold'>{B}</span>" if B else ""
P_html = f"<span style='color:#d62728; font-weight:bold'>{P}</span>" if P else ""
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

# Logic sinh câu
if num_neg == 1 and num_zero == n_cat - 1:
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while the other categories remain unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_zero == n_cat - 1:
    description = f"{B_html} recorded the largest {increase_html} at {abs(Z):.1f}%, while the other categories remain unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == n_cat:
    if num_neg >= 2:
        description = f"{A_html} and {O_html} recorded the largest {decrease_html} at {abs(X):.1f}% and {abs(K):.1f}%, respectively, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
    else:
        description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == n_cat:
    if num_pos >= 2:
        description = f"{B_html} and {P_html} recorded the largest {increase_html} at {abs(Z):.1f}% and {abs(H):.1f}%, respectively, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
    else:
        description = f"{B_html} recorded the largest {increase_html} at {abs(Z):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_zero == n_cat:
    description = f"All categories remain unchanged compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == 1 and num_pos == 1 and num_zero == n_cat - 2:
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while {B_html} recorded the largest {increase_html} at {abs(Z):.1f}%, respectively, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == 1 and num_pos == 0:
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_neg == 0:
    description = f"{B_html} recorded the largest {increase_html} at {abs(Z):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
else:
    decrease_text = ""
    increase_text = ""
    if num_neg >= 2:
        decrease_text = f"{A_html} and {O_html} recorded the largest {decrease_html} at {abs(X):.1f}% and {abs(K):.1f}%, respectively"
    elif num_neg == 1:
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    if num_pos >= 2:
        increase_text = f"{B_html} and {P_html} recorded the largest {increase_html} at {abs(Z):.1f}% and {abs(H):.1f}%, respectively"
    elif num_pos == 1:
        increase_text = f"{B_html} recorded the largest {increase_html} at {abs(Z):.1f}%"
    if decrease_text and increase_text:
        description = f"{decrease_text}, while {increase_text}, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
    elif decrease_text:
        description = f"{decrease_text} compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
    elif increase_text:
        description = f"{increase_text} compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
    else:
        description = f"Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"


st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# -------------------------Stacked Column Chart theo Team--------------------------

fig_stack_team = go.Figure()
team_colors = [
    '#1f77b4',  # xanh dương
    '#2d5cf4',  # xanh lá
    '#ff7f0e',  # cam
    '#d62728',  # đỏ
    '#9467bd',  # tím
    '#8c564b',  # nâu
    '#e377c2',  # hồng
    '#7f7f7f',  # xám
    '#bcbd22',  # vàng xanh
    '#17becf',  # xanh ngọc
    '#f5b041',  # vàng cam
    '#229954',  # xanh lá đậm
    '#0bf4a3',  # xanh biển nhạt
    '#e74c3c',  # đỏ tươi
    '#f7dc6f',  # vàng nhạt
    '#a569bd',  # tím nhạt
    '#45b39d',  # xanh ngọc nhạt
    '#f1948a',  # hồng nhạt
    '#34495e',  # xanh đen
    '#f39c12',  # cam đậm
]
for i, team in enumerate(df_table_team.columns):
    if team == "Tuần":
        continue
    y_values = df_table_team[team].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = team_colors[i % len(team_colors)]
    fig_stack_team.add_trace(go.Bar(
        name=team,
        x=df_table_team["Tuần"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
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
    x_offset = x_idx + 0.8
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
for i, (x, y, t) in enumerate(zip(df_table_team["Tuần"], totals_offset, totals)):
    fig_stack_team.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)


# --- Tính top 2 giảm/tăng mạnh nhất và sinh câu mô tả tự động cho Region (Team) ---
team_cols = [col for col in df_table_team.columns if col != "Tuần"]
idx_w = len(df_table_team["Tuần"]) - 1
idx_w1 = idx_w - 1

# Tính phần trăm thay đổi cho từng team
percent_changes_team = {}
for team in team_cols:
    count_w = float(df_table_team[team].iloc[idx_w])
    count_w1 = float(df_table_team[team].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_team[team] = percent

# Phân loại các nhóm
neg_percents = sorted([(k, v) for k, v in percent_changes_team.items() if v < -1e-6], key=lambda x: x[1])
pos_percents = sorted([(k, v) for k, v in percent_changes_team.items() if v > 1e-6], key=lambda x: x[1], reverse=True)
zero_percents = [(k, v) for k, v in percent_changes_team.items() if abs(v) <= 1e-6]

def short_team_name(name):
    return str(name).split('-')[0].strip() if '-' in str(name) else str(name).strip()

decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

decrease_text = ""
increase_text = ""

# Xử lý các trường hợp đặc biệt
if len(neg_percents) == 0 and len(pos_percents) == 0:
    # Tất cả không đổi
    description = f"All categories remain unchanged compared to the previous week. Overall, the {total_html} change is <span style='color:#111; font-weight:bold'>unchanged</span>."
else:
    # Xử lý giảm
    if len(neg_percents) == 1:
        A, X = neg_percents[0]
        A_html = f"<span style='color:#d62728; font-weight:bold'>{short_team_name(A)}</span>"
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    elif len(neg_percents) >= 2:
        (A, X), (O, K) = neg_percents[0], neg_percents[1]
        A_html = f"<span style='color:#d62728; font-weight:bold'>{short_team_name(A)}</span>"
        O_html = f"<span style='color:#d62728; font-weight:bold'>{short_team_name(O)}</span>"
        decrease_text = f"{A_html} and {O_html} recorded the largest {decrease_html} at {abs(X):.1f}% and {abs(K):.1f}%, respectively"
    # Xử lý tăng
    if len(pos_percents) == 1:
        B, Z = pos_percents[0]
        B_html = f"<span style='color:#d62728; font-weight:bold'>{short_team_name(B)}</span>"
        increase_text = f"{B_html} recorded the largest {increase_html} at {abs(Z):.1f}%"
    elif len(pos_percents) >= 2:
        (B, Z), (P, H) = pos_percents[0], pos_percents[1]
        B_html = f"<span style='color:#d62728; font-weight:bold'>{short_team_name(B)}</span>"
        P_html = f"<span style='color:#d62728; font-weight:bold'>{short_team_name(P)}</span>"
        increase_text = f"{B_html} and {P_html} recorded the largest {increase_html} at {abs(Z):.1f}% and {abs(H):.1f}%, respectively"

    # Nếu có cả tăng và giảm
    if decrease_text and increase_text:
        description_main = f"{decrease_text}, while {increase_text}, compared to the previous week."
    elif decrease_text:
        description_main = f"{decrease_text} compared to the previous week."
    elif increase_text:
        description_main = f"{increase_text} compared to the previous week."
    else:
        # Trường hợp chỉ có không đổi và 1 tăng/giảm
        unchanged_names = [short_team_name(k) for k, v in zero_percents]
        if unchanged_names:
            unchanged_html = ", ".join([f"<span style='color:#d62728; font-weight:bold'>{name}</span>" for name in unchanged_names])
            description_main = f"{unchanged_html} remain unchanged compared to the previous week."
        else:
            description_main = ""

    # Tính tổng phần trăm thay đổi
    sum_w = sum([df_table_team[team].iloc[idx_w] for team in team_cols])
    sum_w1 = sum([df_table_team[team].iloc[idx_w1] for team in team_cols])
    if sum_w1 == 0:
        Y = 100 if sum_w > 0 else 0
    else:
        Y = ((sum_w - sum_w1) / sum_w1) * 100
    if abs(Y) <= 1e-6:
        C = 'unchanged'
    else:
        C = 'increased' if Y > 0 else 'decreased'
    C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"

    description = f"{description_main} Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"


st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height: 10rem'></div>", unsafe_allow_html=True)

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
    x_offset = x_idx + 1
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
        spacing_factor = 0.2
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
for i, (x, y, t) in enumerate(zip(df_table_banner["Tuần"], totals_offset, totals)):
    fig_stack_banner.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# --- Tính X%, Z%, Y% và sinh câu mô tả tự động cho Banner ---
idx_w = len(df_table_banner["Tuần"]) - 1
idx_w1 = idx_w - 1

# Tính phần trăm thay đổi cho từng banner
percent_changes_banner = {}
for banner in banner_cols:
    count_w = float(df_table_banner[banner].iloc[idx_w])
    count_w1 = float(df_table_banner[banner].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_banner[banner] = percent

# X% là số âm lớn nhất (giảm nhiều nhất), A là tên banner đó
neg_percents = {k: v for k, v in percent_changes_banner.items() if v < 0}
if neg_percents:
    A = min(neg_percents, key=neg_percents.get)
    X = neg_percents[A]
else:
    A = min(percent_changes_banner, key=percent_changes_banner.get)
    X = percent_changes_banner[A]

# Z% là số dương lớn nhất (tăng nhiều nhất), B là tên banner đó
pos_percents = {k: v for k, v in percent_changes_banner.items() if v > 0}
if pos_percents:
    B = max(pos_percents, key=pos_percents.get)
    Z = pos_percents[B]
else:
    B = max(percent_changes_banner, key=percent_changes_banner.get)
    Z = percent_changes_banner[B]

# Y% là phần trăm thay đổi tổng số ticket của tất cả banner
sum_w = sum([df_table_banner[banner].iloc[idx_w] for banner in banner_cols])
sum_w1 = sum([df_table_banner[banner].iloc[idx_w1] for banner in banner_cols])
if sum_w1 == 0:
    Y = 100 if sum_w > 0 else 0
else:
    Y = ((sum_w - sum_w1) / sum_w1) * 100

# C là 'increased' nếu Y > 0, ngược lại 'decreased'
C = 'increased' if Y > 0 else 'decreased'

# Hiển thị câu mô tả tự động
A_html = f"<span style='color:#d62728; font-weight:bold'>{A}</span>"
B_html = f"<span style='color:#d62728; font-weight:bold'>{B}</span>"
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"

decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

# Đếm số lượng tăng/giảm/không đổi
change_values = list(percent_changes_banner.values())
num_neg = sum(1 for v in change_values if v < 0)
num_pos = sum(1 for v in change_values if v > 0)
num_zero = sum(1 for v in change_values if abs(v) < 1e-6)

# Sinh câu mô tả theo logic mới
if num_neg == 1 and num_zero == len(banner_cols) - 1:
    # Chỉ có 1 giảm, còn lại không đổi
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while other banners remains unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_zero == len(banner_cols) - 1:
    # Chỉ có 1 tăng, còn lại không đổi
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%, while other banners remains unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == len(banner_cols):
    # Tất cả đều giảm
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == len(banner_cols):
    # Tất cả đều tăng
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_zero == len(banner_cols):
    # Tất cả không đổi
    description = f"All banners remain unchanged compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
else:
    # Trường hợp mặc định như cũ
    if abs(X) < 1e-6:
        decrease_text = f"{A_html} remains unchanged"
    else:
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    if abs(Z) < 1e-6:
        increase_text = f"{B_html} remains unchanged"
    else:
        increase_text = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%"
    description = f"{decrease_text}, while {increase_text}, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"

st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height: 24rem'></div>", unsafe_allow_html=True)

# -------------------------Stacked Column Chart theo Banner theo tháng--------------------------

import calendar
from datetime import datetime, timedelta

# 1. Tạo danh sách ngày cuối từng tháng từ tháng 1 đến tháng hiện tại
today = datetime.today()
start_month = datetime(today.year, 1, 1)
months = []
month_labels = []
month_ends = []

for m in range(1, today.month + 1):
    last_day = calendar.monthrange(today.year, m)[1]
    end_date = datetime(today.year, m, last_day, 23, 59, 59)
    month_ends.append(end_date)
    month_labels.append(f"{end_date.strftime('%b %Y')}")  # Ví dụ: Jan 2025

# 2. Tạo bảng kiểm tra theo Banner cho từng tháng
banner_names = [
    "GO Mall", "Hyper", "Tops", "CBS", "Nguyen Kim", "KUBO", "mini go!"
]
table_data_banner_month = []
for i, end in enumerate(month_ends):
    row = {"Tháng": month_labels[i]}
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
    table_data_banner_month.append(row)
df_table_banner_month = pd.DataFrame(table_data_banner_month)

# 3. Vẽ stacked column chart theo Banner cho từng tháng
import plotly.graph_objects as go

fig_stack_banner_month = go.Figure()
for banner in banner_names:
    y_values = df_table_banner_month[banner].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    fig_stack_banner_month.add_trace(go.Bar(
        name=banner,
        x=df_table_banner_month["Tháng"],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=11),
    ))

# Thêm tổng trên đầu mỗi cột
totals = df_table_banner_month[banner_names].sum(axis=1)
totals_offset = totals + totals * 0.04
for i, (x, y, t) in enumerate(zip(df_table_banner_month["Tháng"], totals_offset, totals)):
    fig_stack_banner_month.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=18, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )

fig_stack_banner_month.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL EVOLUTION OA TICKETS PER BANNER BY MONTH",
        y=0.97,
        x=0.5,
        xanchor='center',
        yanchor='top',
        font=dict(size=26)
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
        title=dict(text="Months", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Number of OA Tickets", font=dict(color='black'))
    )
)

# --- Thêm box so sánh % giữa tháng hiện tại và tháng trước cho từng banner ---
idx_m = len(df_table_banner_month["Tháng"]) - 1
idx_m1 = idx_m - 1
m_label = df_table_banner_month["Tháng"].iloc[idx_m]
active_banners = []
percent_changes = {}
banner_positions = {}
cumulative_height = 0
for banner in banner_names:
    count_m = float(df_table_banner_month[banner].iloc[idx_m])
    if count_m <= 0:
        continue
    count_m1 = float(df_table_banner_month[banner].iloc[idx_m1])
    if count_m1 == 0:
        percent = 100 if count_m > 0 else 0
    else:
        percent = ((count_m - count_m1) / count_m1) * 100
    active_banners.append(banner)
    percent_changes[banner] = percent
    banner_positions[banner] = cumulative_height + count_m / 2
    cumulative_height += count_m
if active_banners:
    total_height = cumulative_height
    x_vals = list(df_table_banner_month["Tháng"])
    x_idx = x_vals.index(m_label)
    x_offset = x_idx + 0.7
    sorted_banners = sorted(active_banners, key=lambda x: banner_positions[x])
    for i, banner in enumerate(sorted_banners):
        percent = percent_changes[banner]
        if percent > 0:
            percent_text = f"M vs M-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"M vs M-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "M vs M-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = banner_positions[banner]
        spacing_factor = 0.35
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_banners)/2))
        fig_stack_banner_month.add_annotation(
            x=m_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_banner_month.add_annotation(
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
st.plotly_chart(fig_stack_banner_month)

# --- Sinh câu mô tả tự động cho chart banner theo tháng ---
neg_percents = {k: v for k, v in percent_changes.items() if v < 0}
if neg_percents:
    A = min(neg_percents, key=neg_percents.get)
    X = neg_percents[A]
else:
    A = min(percent_changes, key=percent_changes.get)
    X = percent_changes[A]

pos_percents = {k: v for k, v in percent_changes.items() if v > 0}
if pos_percents:
    B = max(pos_percents, key=pos_percents.get)
    Z = pos_percents[B]
else:
    B = max(percent_changes, key=percent_changes.get)
    Z = percent_changes[B]

sum_m = sum([df_table_banner_month[banner].iloc[idx_m] for banner in banner_names])
sum_m1 = sum([df_table_banner_month[banner].iloc[idx_m1] for banner in banner_names])
if sum_m1 == 0:
    Y = 100 if sum_m > 0 else 0
else:
    Y = ((sum_m - sum_m1) / sum_m1) * 100

C = 'increased' if Y > 0 else 'decreased'

A_html = f"<span style='color:#d62728; font-weight:bold'>{A}</span>"
B_html = f"<span style='color:#d62728; font-weight:bold'>{B}</span>"
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

change_values = list(percent_changes.values())
num_neg = sum(1 for v in change_values if v < 0)
num_pos = sum(1 for v in change_values if v > 0)
num_zero = sum(1 for v in change_values if abs(v) < 1e-6)

if num_neg == 1 and num_zero == len(banner_names) - 1:
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while other banners remains unchanged, compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_zero == len(banner_names) - 1:
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%, while other banners remains unchanged, compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == len(banner_names):
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == len(banner_names):
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}% compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_zero == len(banner_names):
    description = f"All banners remain unchanged compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
else:
    if abs(X) < 1e-6:
        decrease_text = f"{A_html} remains unchanged"
    else:
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    if abs(Z) < 1e-6:
        increase_text = f"{B_html} remains unchanged"
    else:
        increase_text = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%"
    description = f"{decrease_text}, while {increase_text}, compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"

st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

#----------------------------- Waterfall Chart BY WEEKS--------------------------------------------

created_counts = []
solved_counts = []

for start, end in zip(week_starts, week_ends):
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

#----------------------------- Waterfall Chart theo MONTH -----------------------------

import calendar
from datetime import datetime

# 1. Tạo danh sách ngày đầu và cuối từng tháng từ tháng 1 đến tháng hiện tại
today = datetime.today()
month_starts = []
month_ends = []
month_labels = []
for m in range(1, today.month + 1):
    start_date = datetime(today.year, m, 1)
    last_day = calendar.monthrange(today.year, m)[1]
    end_date = datetime(today.year, m, last_day, 23, 59, 59)
    month_starts.append(start_date)
    month_ends.append(end_date)
    month_labels.append(f"{start_date.strftime('%b %Y')}")  # Ví dụ: Jan 2025

# 2. Tính số lượng Created và Solved ticket cho từng tháng
created_counts_month = []
solved_counts_month = []
for start, end in zip(month_starts, month_ends):
    created = df[(df['create_date'] >= start) & (df['create_date'] <= end)].shape[0]
    solved = df[
        (pd.to_datetime(df['custom_end_date'], errors='coerce') >= start) &
        (pd.to_datetime(df['custom_end_date'], errors='coerce') <= end)
    ].shape[0]
    created_counts_month.append(created)
    solved_counts_month.append(solved)

# 3. Tạo dữ liệu Waterfall step-by-step: mỗi tháng 2 bước Created/Solved
x_waterfall_month = []
y_waterfall_month = []
measure_waterfall_month = []
text_waterfall_month = []
for i, month in enumerate(month_labels):
    x_waterfall_month.append(f"{month} Created")
    y_waterfall_month.append(created_counts_month[i])
    measure_waterfall_month.append("relative")
    text_waterfall_month.append(str(created_counts_month[i]))
    x_waterfall_month.append(f"{month} Solved")
    y_waterfall_month.append(-solved_counts_month[i])
    measure_waterfall_month.append("relative")
    text_waterfall_month.append(str(-solved_counts_month[i]))
    # Thêm cột trắng (dummy) sau mỗi tháng, trừ tháng cuối
    if i < len(month_labels) - 1:
        x_waterfall_month.append(f"{month} Spacer")
        y_waterfall_month.append(0)
        measure_waterfall_month.append("relative")
        text_waterfall_month.append("")

# 4. Tạo tickvals và ticktext cho trục x (giữa 2 cột Created/Solved, bỏ qua Spacer)
tickvals_month = []
ticktext_month = []
for i, month in enumerate(month_labels):
    tickvals_month.append(i*3 + 0.5)
    ticktext_month.append(month)

# 5. Vẽ Waterfall chart theo tháng
fig_waterfall_month = go.Figure(go.Waterfall(
    x=x_waterfall_month,
    y=y_waterfall_month,
    measure=measure_waterfall_month,
    text=[f'<span style="background-color:#fff9b1; color:#444; padding:2px 6px; border-radius:4px; font-weight:bold;">{v}</span>' if v not in [None, "", "0"] else "" for v in text_waterfall_month],
    textposition="outside",
    connector={"line": {"color": "rgba(0,0,0,0.3)"}},
    increasing={"marker": {"color": "#e74c3c", "line": {"width": 0}}},
    decreasing={"marker": {"color": "#27ae60", "line": {"width": 0}}},
    showlegend=False
))
# Thêm legend thủ công cho Created (đỏ) và Solved (xanh)
fig_waterfall_month.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=14, color='#e74c3c'),
    name='Created ticket',
    showlegend=True
))
fig_waterfall_month.add_trace(go.Scatter(
    x=[None], y=[None],
    mode='markers',
    marker=dict(size=14, color='#27ae60'),
    name='Solved ticket',
    showlegend=True
))

fig_waterfall_month.update_layout(
    title=dict(
        text="NATIONWIDE ON ASSESSMENT TICKET OVER MONTHS",
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
        title=dict(text="Months", font=dict(color='black')),
        tickvals=tickvals_month,
        ticktext=ticktext_month,
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
st.plotly_chart(fig_waterfall_month, use_container_width=True)
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

# --- STACKED COLUMN CHART TỪ EXCEL: OVERALL COST SPENT PER CATEGORY BY WEEKS (MVND)-----------

if 'Tuần' in df_excel.columns:
    x_col = 'Tuần'
else:
    x_col = df_excel.columns[0]
category_cols = [col for col in df_excel.columns if col != x_col]
fig_stack_excel = go.Figure()
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
for i, cat in enumerate(category_cols):
    y_values = df_excel[cat].apply(lambda v: int(round(v)) if pd.notnull(v) else 0).tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette[i % len(color_palette)]
    fig_stack_excel.add_trace(go.Bar(
        name=cat,
        x=df_excel[x_col],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
    ))
totals = df_excel[category_cols].apply(lambda col: col.apply(lambda v: int(round(v)) if pd.notnull(v) else 0)).sum(axis=1)
totals_offset = totals + totals * 0.04

for i, (x, y, t) in enumerate(zip(df_excel[x_col], totals_offset, totals)):
    fig_stack_excel.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_excel.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL COST SPENT PER CATEGORY BY WEEKS (MVND)",
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)

# --- BOX SO SÁNH PHẦN TRĂM CHO STACKED COLUMN CHART TỪ EXCEL (CATEGORY) ---
# idx_w: tuần hiện tại (kế cuối), idx_w1: tuần trước (trước kế cuối)
idx_w = len(df_excel) - 1  # W27
idx_w1 = len(df_excel) - 2 # W26
w_label = df_excel[x_col].iloc[idx_w]
active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
def safe_to_int(v):
    try:
        return int(round(float(v)))
    except:
        return 0
for cat in category_cols:
    count_w = safe_to_int(df_excel[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = safe_to_int(df_excel[cat].iloc[idx_w1])
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
    x_vals = list(df_excel[x_col])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 0.9
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
        spacing_factor = 0.9
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_excel.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_excel.add_annotation(
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
# --- Thêm gridline dọc gạch đứt phân chia các week ---
y_max = totals_offset.max() * 1.05  # hoặc lấy max của các cột, nhân thêm 5% cho đẹp
vertical_lines = []
num_weeks = len(df_excel[x_col])
for i in range(1, num_weeks):
    vertical_lines.append(dict(
        type="line",
        xref="x",
        yref="y",
        x0=i - 0.5,
        x1=i - 0.5,
        y0=0,
        y1=y_max,
        line=dict(
            color="gray",
            width=1,
            dash="dot"
        ),
        layer="below"
    ))
fig_stack_excel.update_layout(shapes=vertical_lines)
st.plotly_chart(fig_stack_excel)
st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

# --- AUTO DESCRIPTION CHO STACKED COLUMN CHART COST THEO CATEGORY (EXCEL) ---
# Tính phần trăm thay đổi cho từng category giữa tuần hiện tại và tuần trước
percent_changes_cat = {}
for cat in category_cols:
    count_w = safe_to_int(df_excel[cat].iloc[idx_w])
    count_w1 = safe_to_int(df_excel[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_cat[cat] = percent

# Phân loại các giá trị
neg_percents = [(k, v) for k, v in percent_changes_cat.items() if v < 0]
pos_percents = [(k, v) for k, v in percent_changes_cat.items() if v > 0]
zero_percents = [(k, v) for k, v in percent_changes_cat.items() if abs(v) < 1e-6]
total_cats = len(percent_changes_cat)

# Tính tổng thay đổi
sum_w = sum([safe_to_int(df_excel[cat].iloc[idx_w]) for cat in category_cols])
sum_w1 = sum([safe_to_int(df_excel[cat].iloc[idx_w1]) for cat in category_cols])
if sum_w1 == 0:
    Y = 100 if sum_w > 0 else 0
else:
    Y = ((sum_w - sum_w1) / sum_w1) * 100
C = 'increased' if Y > 0 else 'decreased'
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"

# Hàm lấy tên đẹp (bạn có thể sửa lại mapping tiếng Việt ở đây)
def get_cat_name(cat):
    mapping = {
        'FFP': 'FFP (Phòng cháy chữa cháy)',
        'LET': 'LET (Thang máy, thang cuốn)',
        'ACMV': 'ACMV (Điều hòa không khí)',
        'Building': 'Building',
        'Electrical': 'Electrical',
        # Thêm các mapping khác nếu cần
    }
    return mapping.get(cat, cat)

if total_cats == 1:
    # Chỉ có 1 box so sánh
    cat, val = list(percent_changes_cat.items())[0]
    change_type = increase_html if val > 0 else decrease_html
    cat_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(cat)}</span>"
    st.markdown(
        f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
        f"{cat_html} recorded the largest {change_type} at <b>{abs(val):.1f}%</b> compared to the previous week. "
        f"Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )
elif len(neg_percents) == 1 and len(pos_percents) == 0:
    # 1 giảm, còn lại không đổi
    cat, val = neg_percents[0]
    cat_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(cat)}</span>"
    st.markdown(
        f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
        f"{cat_html} recorded the largest {decrease_html} at <b>{abs(val):.1f}%</b>, while the other categories remain unchanged, compared to the previous week. "
        f"Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )
elif len(pos_percents) == 1 and len(neg_percents) == 0:
    # 1 tăng, còn lại không đổi
    cat, val = pos_percents[0]
    cat_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(cat)}</span>"
    st.markdown(
        f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
        f"{cat_html} recorded the largest {increase_html} at <b>{abs(val):.1f}%</b>, while the other categories remain unchanged, compared to the previous week. "
        f"Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )
elif len(neg_percents) == 1 and len(pos_percents) == 1:
    # 1 tăng, 1 giảm
    cat_dec, val_dec = neg_percents[0]
    cat_inc, val_inc = pos_percents[0]
    cat_dec_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(cat_dec)}</span>"
    cat_inc_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(cat_inc)}</span>"
    st.markdown(
        f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
        f"{cat_dec_html} recorded the largest {decrease_html} at <b>{abs(val_dec):.1f}%</b>, while {cat_inc_html} showed the largest {increase_html} at <b>{abs(val_inc):.1f}%</b>, respectively, compared to the previous week. "
        f"Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )
elif (len(neg_percents) == total_cats) or (len(pos_percents) == total_cats):
    # Tất cả đều tăng hoặc đều giảm
    if len(neg_percents) >= 2:
        sorted_neg = sorted(neg_percents, key=lambda x: x[1])
        (A, X), (O, K) = sorted_neg[0], sorted_neg[1]
        A_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(A)}</span>"
        O_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(O)}</span>"
        st.markdown(
            f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
            f"{A_html} and {O_html} recorded the largest {decrease_html} at <b>{abs(X):.1f}%</b> and <b>{abs(K):.1f}%</b>, respectively, compared to the previous week. "
            f"Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
            f"</div>",
            unsafe_allow_html=True
        )
    elif len(pos_percents) >= 2:
        sorted_pos = sorted(pos_percents, key=lambda x: x[1], reverse=True)
        (B, Z), (P, H) = sorted_pos[0], sorted_pos[1]
        B_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(B)}</span>"
        P_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(P)}</span>"
        st.markdown(
            f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
            f"{B_html} and {P_html} recorded the largest {increase_html} at <b>{abs(Z):.1f}%</b> and <b>{abs(H):.1f}%</b>, respectively, compared to the previous week. "
            f"Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
            f"</div>",
            unsafe_allow_html=True
        )
else:
    # Trường hợp mặc định: lấy top 2 tăng, top 2 giảm như cũ, có format HTML
    sorted_neg = sorted(neg_percents, key=lambda x: x[1])
    sorted_pos = sorted(pos_percents, key=lambda x: x[1], reverse=True)
    decrease_text = ""
    increase_text = ""
    if len(sorted_neg) >= 2:
        (A, X), (O, K) = sorted_neg[0], sorted_neg[1]
        A_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(A)}</span>"
        O_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(O)}</span>"
        decrease_text = f"{A_html} and {O_html} recorded the largest {decrease_html} at <b>{abs(X):.1f}%</b> and <b>{abs(K):.1f}%</b>, respectively"
    elif len(sorted_neg) == 1:
        (A, X) = sorted_neg[0]
        A_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(A)}</span>"
        decrease_text = f"{A_html} recorded the largest {decrease_html} at <b>{abs(X):.1f}%</b>"
    if len(sorted_pos) >= 2:
        (B, Z), (P, H) = sorted_pos[0], sorted_pos[1]
        B_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(B)}</span>"
        P_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(P)}</span>"
        increase_text = f"{B_html} and {P_html} showed highest {increase_html} with <b>{abs(Z):.1f}%</b> and <b>{abs(H):.1f}%</b>, respectively"
    elif len(sorted_pos) == 1:
        (B, Z) = sorted_pos[0]
        B_html = f"<span style='color:#d62728; font-weight:bold'>{get_cat_name(B)}</span>"
        increase_text = f"{B_html} showed highest {increase_html} with <b>{abs(Z):.1f}%</b>"
    if decrease_text and increase_text:
        text = f"{decrease_text}, while {increase_text}, compared to the previous week. "
    elif decrease_text:
        text = f"{decrease_text} compared to the previous week. "
    elif increase_text:
        text = f"{increase_text} compared to the previous week. "
    else:
        text = "No significant changes compared to the previous week. "
    st.markdown(
        f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>"
        f"{text}Overall, the {total_html} change is {C_html} by <b>{abs(Y):.1f}%</b>"
        f"</div>",
        unsafe_allow_html=True
    )
st.markdown("<div style='height: 25rem'></div>", unsafe_allow_html=True)

# --- STACKED COLUMN CHART TỪ EXCEL: OVERALL COST SPENT PER TEAM BY WEEKS (MVND)-----------

if 'Tuần' in df_excel2.columns:
    x_col2 = 'Tuần'
else:
    x_col2 = df_excel2.columns[0]
team_cols = [col for col in df_excel2.columns if col != x_col2]
fig_stack_excel2 = go.Figure()
color_palette2 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
def safe_to_int2(v):
    try:
        return int(round(float(v)))
    except:
        return 0
for i, team in enumerate(team_cols):
    y_values = df_excel2[team].apply(safe_to_int2).tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette2[i % len(color_palette2)]
    fig_stack_excel2.add_trace(go.Bar(
        name=team,
        x=df_excel2[x_col2],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
    ))
totals2 = df_excel2[team_cols].applymap(safe_to_int2).sum(axis=1)
totals_offset2 = totals2 + totals2 * 0.04
for i, (x, y, t) in enumerate(zip(df_excel2[x_col2], totals_offset2, totals2)):
    fig_stack_excel2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_excel2.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL COST SPENT PER TEAM BY WEEKS (MVND)",
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
# --- BOX SO SÁNH PHẦN TRĂM ---
idx_w2 = len(df_excel2) - 1
idx_w1_2 = len(df_excel2) - 2
w_label2 = df_excel2[x_col2].iloc[idx_w2]
active_teams = []
percent_changes2 = {}
team_positions = {}
cumulative_height2 = 0
for team in team_cols:
    count_w = safe_to_int2(df_excel2[team].iloc[idx_w2])
    if count_w <= 0:
        continue
    count_w1 = safe_to_int2(df_excel2[team].iloc[idx_w1_2])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_teams.append(team)
    percent_changes2[team] = percent
    team_positions[team] = cumulative_height2 + count_w / 2
    cumulative_height2 += count_w
if active_teams:
    total_height2 = cumulative_height2
    x_vals2 = list(df_excel2[x_col2])
    x_idx2 = x_vals2.index(w_label2)
    x_offset2 = x_idx2 + 0.9
    sorted_teams = sorted(active_teams, key=lambda x: team_positions[x])
    for i, team in enumerate(sorted_teams):
        percent = percent_changes2[team]
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
        spacing_factor = 0.9
        y_box = y_col + (total_height2 * spacing_factor * (i - len(sorted_teams)/2))
        fig_stack_excel2.add_annotation(
            x=w_label2, y=y_col,
            ax=x_offset2, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_excel2.add_annotation(
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
# --- Thêm gridline dọc gạch đứt ---
y_max2 = totals_offset2.max() * 1.05
vertical_lines2 = []
num_weeks2 = len(df_excel2[x_col2])
for i in range(1, num_weeks2):
    vertical_lines2.append(dict(
        type="line",
        xref="x",
        yref="y",
        x0=i - 0.5,
        x1=i - 0.5,
        y0=0,
        y1=y_max2,
        line=dict(
            color="gray",
            width=1,
            dash="dot"
        ),
        layer="below"
    ))
fig_stack_excel2.update_layout(shapes=vertical_lines2)
st.plotly_chart(fig_stack_excel2)
st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

# --- AUTO DESCRIPTION CHO STACKED COLUMN CHART COST THEO TEAM (EXCEL) ---
def get_team_name(team):
    # Nếu muốn mapping tên đẹp, sửa ở đây
    return team

total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"

percent_changes_team = {}
for team in team_cols:
    count_w = safe_to_int2(df_excel2[team].iloc[idx_w2])
    count_w1 = safe_to_int2(df_excel2[team].iloc[idx_w1_2])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_team[team] = percent

sum_w = sum([safe_to_int2(df_excel2[team].iloc[idx_w2]) for team in team_cols])
sum_w1 = sum([safe_to_int2(df_excel2[team].iloc[idx_w1_2]) for team in team_cols])
if sum_w1 == 0:
    total_percent = 100 if sum_w > 0 else 0
else:
    total_percent = ((sum_w - sum_w1) / sum_w1) * 100

# Phân loại các giá trị
neg_percents = [(k, v) for k, v in percent_changes_team.items() if v < 0]
pos_percents = [(k, v) for k, v in percent_changes_team.items() if v > 0]
zero_percents = [(k, v) for k, v in percent_changes_team.items() if v == 0]

neg_percents.sort(key=lambda x: x[1])  # tăng âm nhiều nhất
pos_percents.sort(key=lambda x: -x[1]) # tăng dương nhiều nhất

# Format HTML cho tên team và số %
def team_html(team):
    return f"<span style='font-weight:bold; color:#d62728'>{get_team_name(team)}</span>"
def percent_html(val):
    return f"<span style='font-weight:bold; color:#d62728'>{abs(val):.1f}%</span>"
def total_percent_html(val):
    return f"<span style='font-weight:bold; color:#d62728'>{abs(val):.1f}%</span>"

# Sinh câu mô tả
if len(percent_changes_team) == 1:
    # Chỉ có 1 team
    team, val = list(percent_changes_team.items())[0]
    if val > 0:
        desc = f"{team_html(team)} recorded the largest {increase_html} at {percent_html(val)} compared to the previous week. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif val < 0:
        desc = f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)} compared to the previous week. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    else:
        desc = f"{team_html(team)} remains unchanged compared to the previous week. Overall, the {total_html} change is 0%"
else:
    # Nhiều hơn 1 team
    if len(neg_percents) == 1 and len(pos_percents) == 0:
        # 1 giảm, còn lại không đổi
        team, val = neg_percents[0]
        desc = f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)}, while the other teams remain unchanged, compared to the previous week. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    elif len(pos_percents) == 1 and len(neg_percents) == 0:
        # 1 tăng, còn lại không đổi
        team, val = pos_percents[0]
        desc = f"{team_html(team)} recorded the largest {increase_html} at {percent_html(val)}, while the other teams remain unchanged, compared to the previous week. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif len(neg_percents) == 1 and len(pos_percents) == 1:
        # 1 tăng, 1 giảm
        team_dec, val_dec = neg_percents[0]
        team_inc, val_inc = pos_percents[0]
        desc = f"{team_html(team_dec)} recorded the largest {decrease_html} at {percent_html(val_dec)}, while {team_html(team_inc)} showed the largest {increase_html} at {percent_html(val_inc)}, respectively, compared to the previous week. Overall, the {total_html} change is {'decreased' if total_percent < 0 else 'increased'} by {total_percent_html(total_percent)}"
    elif len(neg_percents) == 0 and len(pos_percents) > 0:
        # Tất cả tăng, lấy top 2
        if len(pos_percents) == 1:
            team, val = pos_percents[0]
            desc = f"{team_html(team)} recorded the largest {increase_html} at {percent_html(val)} compared to the previous week. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
        else:
            (team1, val1), (team2, val2) = pos_percents[:2]
            desc = f"{team_html(team1)} and {team_html(team2)} recorded the largest {increase_html} at {percent_html(val1)} and {percent_html(val2)}, respectively, compared to the previous week. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif len(pos_percents) == 0 and len(neg_percents) > 0:
        # Tất cả giảm, lấy top 2
        if len(neg_percents) == 1:
            team, val = neg_percents[0]
            desc = f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)} compared to the previous week. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
        else:
            (team1, val1), (team2, val2) = neg_percents[:2]
            desc = f"{team_html(team1)} and {team_html(team2)} recorded the largest {decrease_html} at {percent_html(val1)} and {percent_html(val2)}, respectively, compared to the previous week. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    else:
        # Trường hợp nhiều hơn 2 tăng/giảm, lấy top 2 mỗi bên
        desc_parts = []
        if len(neg_percents) > 0:
            if len(neg_percents) == 1:
                team, val = neg_percents[0]
                desc_parts.append(f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)}")
            else:
                (team1, val1), (team2, val2) = neg_percents[:2]
                desc_parts.append(f"{team_html(team1)} and {team_html(team2)} recorded the largest {decrease_html} at {percent_html(val1)} and {percent_html(val2)}")
        if len(pos_percents) > 0:
            if len(pos_percents) == 1:
                team, val = pos_percents[0]
                desc_parts.append(f"{team_html(team)} showed the largest {increase_html} at {percent_html(val)}")
            else:
                (team1, val1), (team2, val2) = pos_percents[:2]
                desc_parts.append(f"{team_html(team1)} and {team_html(team2)} showed the largest {increase_html} at {percent_html(val1)} and {percent_html(val2)}")
        desc = ", while ".join(desc_parts) + f", respectively, compared to the previous week. Overall, the {total_html} change is {'decreased' if total_percent < 0 else 'increased'} by {total_percent_html(total_percent)}"

st.markdown(f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{desc}</div>", unsafe_allow_html=True)
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

# --- STACKED COLUMN CHART TỪ EXCEL: OVERALL COST SPENT PER BANNER BY WEEKS (MVND)-----------

if 'Tuần' in df_excel3.columns:
    x_col3 = 'Tuần'
else:
    x_col3 = df_excel3.columns[0]
banner_cols3 = [col for col in df_excel3.columns if col != x_col3]
fig_stack_excel3 = go.Figure()
color_palette3 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
def safe_to_int3(v):
    try:
        return int(round(float(v)))
    except:
        return 0
for i, banner in enumerate(banner_cols3):
    y_values = df_excel3[banner].apply(safe_to_int3).tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette3[i % len(color_palette3)]
    fig_stack_excel3.add_trace(go.Bar(
        name=banner,
        x=df_excel3[x_col3],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
    ))
totals3 = df_excel3[banner_cols3].applymap(safe_to_int3).sum(axis=1)
totals_offset3 = totals3 + totals3 * 0.04
for i, (x, y, t) in enumerate(zip(df_excel3[x_col3], totals_offset3, totals3)):
    fig_stack_excel3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_excel3.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL COST SPENT PER BANNER BY WEEKS (MVND)",
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Weeks", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
idx_w3 = len(df_excel3) - 1
idx_w1_3 = len(df_excel3) - 2
w_label3 = df_excel3[x_col3].iloc[idx_w3]
active_banners3 = []
percent_changes3 = {}
banner_positions3 = {}
cumulative_height3 = 0
for banner in banner_cols3:
    count_w = safe_to_int3(df_excel3[banner].iloc[idx_w3])
    if count_w <= 0:
        continue
    count_w1 = safe_to_int3(df_excel3[banner].iloc[idx_w1_3])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_banners3.append(banner)
    percent_changes3[banner] = percent
    banner_positions3[banner] = cumulative_height3 + count_w / 2
    cumulative_height3 += count_w
if active_banners3:
    total_height3 = cumulative_height3
    x_vals3 = list(df_excel3[x_col3])
    x_idx3 = x_vals3.index(w_label3)
    x_offset3 = x_idx3 + 0.9
    sorted_banners3 = sorted(active_banners3, key=lambda x: banner_positions3[x])
    for i, banner in enumerate(sorted_banners3):
        percent = percent_changes3[banner]
        if percent > 0:
            percent_text = f"W vs W-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"W vs W-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "W vs W-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = banner_positions3[banner]
        spacing_factor = 0.9
        y_box = y_col + (total_height3 * spacing_factor * (i - len(sorted_banners3)/2))
        fig_stack_excel3.add_annotation(
            x=w_label3, y=y_col,
            ax=x_offset3, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_excel3.add_annotation(
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
# Gridline dọc
y_max3 = totals_offset3.max() * 1.05
vertical_lines3 = []
num_weeks3 = len(df_excel3[x_col3])
for i in range(1, num_weeks3):
    vertical_lines3.append(dict(
        type="line",
        xref="x",
        yref="y",
        x0=i - 0.5,
        x1=i - 0.5,
        y0=0,
        y1=y_max3,
        line=dict(
            color="gray",
            width=1,
            dash="dot"
        ),
        layer="below"
    ))
fig_stack_excel3.update_layout(shapes=vertical_lines3)
st.plotly_chart(fig_stack_excel3)
st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

# Tính phần trăm thay đổi cho từng banner giữa tuần hiện tại và tuần trước
percent_changes_banner3 = {}
for banner in banner_cols3:
    count_w = safe_to_int3(df_excel3[banner].iloc[idx_w3])
    count_w1 = safe_to_int3(df_excel3[banner].iloc[idx_w1_3])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_banner3[banner] = percent

# X% là số âm lớn nhất (giảm nhiều nhất), A là tên banner đó
neg_percents = {k: v for k, v in percent_changes_banner3.items() if v < 0}
if neg_percents:
    A = min(neg_percents, key=neg_percents.get)
    X = neg_percents[A]
else:
    A = min(percent_changes_banner3, key=percent_changes_banner3.get)
    X = percent_changes_banner3[A]

# Z% là số dương lớn nhất (tăng nhiều nhất), B là tên banner đó
pos_percents = {k: v for k, v in percent_changes_banner3.items() if v > 0}
if pos_percents:
    B = max(pos_percents, key=pos_percents.get)
    Z = pos_percents[B]
else:
    B = max(percent_changes_banner3, key=percent_changes_banner3.get)
    Z = percent_changes_banner3[B]

# Y% là phần trăm thay đổi tổng số cost của tất cả banner
sum_w = sum([safe_to_int3(df_excel3[banner].iloc[idx_w3]) for banner in banner_cols3])
sum_w1 = sum([safe_to_int3(df_excel3[banner].iloc[idx_w1_3]) for banner in banner_cols3])
if sum_w1 == 0:
    Y = 100 if sum_w > 0 else 0
else:
    Y = ((sum_w - sum_w1) / sum_w1) * 100

C = 'increased' if Y > 0 else 'decreased'

A_html = f"<span style='color:#d62728; font-weight:bold'>{A}</span>"
B_html = f"<span style='color:#d62728; font-weight:bold'>{B}</span>"
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

# Đếm số lượng tăng/giảm/không đổi
change_values = list(percent_changes_banner3.values())
num_neg = sum(1 for v in change_values if v < 0)
num_pos = sum(1 for v in change_values if v > 0)
num_zero = sum(1 for v in change_values if abs(v) < 1e-6)

# Sinh câu mô tả theo logic mới
if num_neg == 1 and num_zero == len(banner_cols3) - 1:
    # Chỉ có 1 giảm, còn lại không đổi
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while other banners remains unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_zero == len(banner_cols3) - 1:
    # Chỉ có 1 tăng, còn lại không đổi
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%, while other banners remains unchanged, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == len(banner_cols3):
    # Tất cả đều giảm
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == len(banner_cols3):
    # Tất cả đều tăng
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}% compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_zero == len(banner_cols3):
    # Tất cả không đổi
    description = f"All banners remain unchanged compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
else:
    # Trường hợp mặc định như cũ
    if abs(X) < 1e-6:
        decrease_text = f"{A_html} remains unchanged"
    else:
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    if abs(Z) < 1e-6:
        increase_text = f"{B_html} remains unchanged"
    else:
        increase_text = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%"
    description = f"{decrease_text}, while {increase_text}, compared to the previous week. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"

st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

# ------------------- 100% STACKED COLUMN CHART TỪ EXCEL------------------------

import plotly.graph_objects as go

# 1. Xác định các cột cần vẽ (giả sử là 8 cột đầu tiên)
column_names = list(df_excel4.columns)
label_col = df_excel4.columns[0]  # Cột đầu tiên (không có header, chứa Jan, Feb,...)
x_cols = df_excel4.columns[1:9]   # 8 cột tiếp theo: GO Mall ... Total
row_labels = df_excel4[label_col].astype(str).tolist()  # Lấy đúng tên tháng

# 2. Lấy nhãn dòng (tuần, hoặc index)
if 'Tuần' in df_excel4.columns:
    row_labels = df_excel4['Tuần'].astype(str).tolist()
    data = df_excel4[x_cols].copy()
else:
    data = df_excel4[x_cols].copy()

# --- THÊM DÒNG NÀY ĐỂ CHUYỂN TOÀN BỘ SANG SỐ ---
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

# 3. Tính phần trăm từng dòng trên tổng của mỗi cột
percent_data = data.div(data.sum(axis=0), axis=1) * 100

# 4. Vẽ 100% stacked column chart
fig_100stack = go.Figure()
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
for i, row_label in enumerate(row_labels):  # Đúng! Lặp qua từng dòng (tháng)
    y_values = percent_data.iloc[i].tolist()  # Phần trăm của từng banner tại dòng i
    text_labels = [row_label if v > 0 else "" for v in y_values]
    color = "#1976d2"
    # Nếu là "remaining year budget" thì xanh lá
    if row_label.strip().lower() == "remaining year budget":
        color = "#27ae60"
    else:
        # Nếu bất kỳ giá trị từ dòng 14 trở xuống của dòng này > 0 thì tô đỏ
        vals = data.iloc[i, :]
        if (vals > 0).any() and i >= 13:
            color = "#e53935"
    fig_100stack.add_trace(go.Bar(
        name=row_label,
        x=x_cols,
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        marker_color=color,
        marker_line_color="black",
        marker_line_width=1,
        textfont=dict(size=10),
        textangle=0,
        hovertemplate=f"<b>{row_label}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>"
    ))

fig_100stack.update_layout(
barmode='stack',
title=dict(
    text="OVERALL COST SPENT PER BANNER",
    x=0.5,
    y=0.97,
    xanchor='center',
    yanchor='top',
    font=dict(size=24)
),
width=1200,
height=700,
showlegend=False,  # Ẩn legend mặc định
yaxis=dict(
    range=[0, 100],
    tickfont=dict(size=13, color='black')
)
)

# Thêm 3 trace dummy để tạo custom legend (dùng visible='legendonly')
fig_100stack.add_trace(go.Bar(
    x=[-999], y=[0],
    name="Actual cost within budget",
    marker_color="#1976d2",  # Xanh nước biển
    showlegend=True,
    visible='legendonly'
))
fig_100stack.add_trace(go.Bar(
    x=[-999], y=[0],
    name="Actual cost exceeding budget",
    marker_color="#e53935",  # Đỏ
    showlegend=True,
    visible='legendonly'
))
fig_100stack.add_trace(go.Bar(
    x=[-999], y=[0],
    name="Remaining budget",
    marker_color="#27ae60",  # Xanh lá
    showlegend=True,
    visible='legendonly'
))

# Xác định vị trí cột "Total" trên trục x
total_idx = list(x_cols).index("Total")
fig_100stack.add_shape(
    type="rect",
    xref="x",
    yref="paper",
    x0=total_idx - 0.5,   # Mở rộng sang trái
    x1=total_idx + 0.5,   # Mở rộng sang phải
    y0=-0.06,
    y1=1.05,
    line=dict(
        color="red",
        width=5,
        dash="solid"
    ),
    fillcolor="rgba(0,0,0,0)",
    layer="above"
)

st.plotly_chart(fig_100stack, use_container_width=True)
st.markdown("""
<div style="width: 100%; display: flex; justify-content: center; margin-bottom: 1.5rem; margin-top: 0.1rem;">
<div style="display: flex; gap: 2.5rem; align-items: center;">
    <div style="display: flex; align-items: center;">
    <span style="width: 32px; height: 18px; background: #1976d2; display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
    <span style="font-size: 18px;">Actual cost within budget</span>
    </div>
    <div style="display: flex; align-items: center;">
    <span style="width: 32px; height: 18px; background: #e53935; display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
    <span style="font-size: 18px;">Actual cost exceeding budget</span>
    </div>
    <div style="display: flex; align-items: center;">
    <span style="width: 32px; height: 18px; background: #27ae60; display: inline-block; border-radius: 3px; margin-right: 8px;"></span>
    <span style="font-size: 18px;">Remaining budget</span>
    </div>
</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem; font-size: 28px;">
        <span style="color: #e53935; font-weight: bold;">*Note:</span>
        The budget for CBS is still a dummy data, real budget will be added in in the future
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

# --- STACKED COLUMN CHART TỪ EXCEL: OVERALL COST SPENT PER CATEGORY BY MONTH (MVND)-----------

if 'Months' in df_excel5.columns:
    x_col_month = 'Months'
else:
    x_col_month = df_excel5.columns[0]
category_cols_month = [col for col in df_excel5.columns if col != x_col_month]
fig_stack_excel_month = go.Figure()
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
for i, cat in enumerate(category_cols_month):
    y_values = df_excel5[cat].apply(lambda v: int(round(v)) if pd.notnull(v) else 0).tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette[i % len(color_palette)]
    fig_stack_excel_month.add_trace(go.Bar(
        name=cat,
        x=df_excel5[x_col_month],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
    ))
totals = df_excel5[category_cols_month].apply(lambda col: col.apply(lambda v: int(round(v)) if pd.notnull(v) else 0)).sum(axis=1)
totals_offset = totals + totals * 0.04
for i, (x, y, t) in enumerate(zip(df_excel5[x_col_month], totals_offset, totals)):
    fig_stack_excel_month.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_excel_month.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL COST SPENT PER CATEGORY BY MONTH (MVND)",
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Months", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
# --- BOX SO SÁNH PHẦN TRĂM ---
idx_w = len(df_excel5) - 1
idx_w1 = len(df_excel5) - 2
w_label = df_excel5[x_col_month].iloc[idx_w]
active_categories = []
percent_changes = {}
category_positions = {}
cumulative_height = 0
def safe_to_int(v):
    try:
        return int(round(float(v)))
    except:
        return 0
for cat in category_cols_month:
    count_w = safe_to_int(df_excel5[cat].iloc[idx_w])
    if count_w <= 0:
        continue
    count_w1 = safe_to_int(df_excel5[cat].iloc[idx_w1])
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
    x_vals = list(df_excel5[x_col_month])
    x_idx = x_vals.index(w_label)
    x_offset = x_idx + 0.8
    sorted_categories = sorted(active_categories, key=lambda x: category_positions[x])
    for i, cat in enumerate(sorted_categories):
        percent = percent_changes[cat]
        if percent > 0:
            percent_text = f"M vs M-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"M vs M-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "M vs M-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = category_positions[cat]
        spacing_factor = 2
        y_box = y_col + (total_height * spacing_factor * (i - len(sorted_categories)/2))
        fig_stack_excel_month.add_annotation(
            x=w_label, y=y_col,
            ax=x_offset, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_excel_month.add_annotation(
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
# --- Thêm gridline dọc gạch đứt phân chia các tháng ---
y_max = totals_offset.max() * 1.05
vertical_lines = []
num_months = len(df_excel5[x_col_month])
for i in range(1, num_months):
    vertical_lines.append(dict(
        type="line",
        xref="x",
        yref="y",
        x0=i - 0.5,
        x1=i - 0.5,
        y0=0,
        y1=y_max,
        line=dict(
            color="gray",
            width=1,
            dash="dot"
        ),
        layer="below"
    ))
fig_stack_excel_month.update_layout(shapes=vertical_lines)
st.plotly_chart(fig_stack_excel_month)
st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

# --- AUTO DESCRIPTION CHO STACKED COLUMN CHART COST THEO CATEGORY BY MONTH (EXCEL) ---
def get_cat_name_month(cat):
    # Mapping tên đẹp nếu muốn
    return cat

total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"

percent_changes_cat = {}
for cat in category_cols_month:
    count_w = safe_to_int(df_excel5[cat].iloc[idx_w])
    count_w1 = safe_to_int(df_excel5[cat].iloc[idx_w1])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_cat[cat] = percent

sum_w = sum([safe_to_int(df_excel5[cat].iloc[idx_w]) for cat in category_cols_month])
sum_w1 = sum([safe_to_int(df_excel5[cat].iloc[idx_w1]) for cat in category_cols_month])
if sum_w1 == 0:
    total_percent = 100 if sum_w > 0 else 0
else:
    total_percent = ((sum_w - sum_w1) / sum_w1) * 100

neg_percents = [(k, v) for k, v in percent_changes_cat.items() if v < 0]
pos_percents = [(k, v) for k, v in percent_changes_cat.items() if v > 0]
zero_percents = [(k, v) for k, v in percent_changes_cat.items() if v == 0]

neg_percents.sort(key=lambda x: x[1])
pos_percents.sort(key=lambda x: -x[1])

def cat_html(cat):
    return f"<span style='font-weight:bold; color:#d62728'>{get_cat_name_month(cat)}</span>"
def percent_html(val):
    return f"<span style='font-weight:bold; color:#d62728'>{abs(val):.1f}%</span>"
def total_percent_html(val):
    return f"<span style='font-weight:bold; color:#d62728'>{abs(val):.1f}%</span>"

if len(percent_changes_cat) == 1:
    # Chỉ có 1 category
    cat, val = list(percent_changes_cat.items())[0]
    if val > 0:
        desc = f"{cat_html(cat)} recorded the largest {increase_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif val < 0:
        desc = f"{cat_html(cat)} recorded the largest {decrease_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    else:
        desc = f"{cat_html(cat)} remains unchanged compared to the previous month. Overall, the {total_html} change is 0%"
else:
    if len(neg_percents) == 1 and len(pos_percents) == 0:
        cat, val = neg_percents[0]
        desc = f"{cat_html(cat)} recorded the largest {decrease_html} at {percent_html(val)}, while the other categories remain unchanged, compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    elif len(pos_percents) == 1 and len(neg_percents) == 0:
        cat, val = pos_percents[0]
        desc = f"{cat_html(cat)} recorded the largest {increase_html} at {percent_html(val)}, while the other categories remain unchanged, compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif len(neg_percents) == 1 and len(pos_percents) == 1:
        cat_dec, val_dec = neg_percents[0]
        cat_inc, val_inc = pos_percents[0]
        desc = f"{cat_html(cat_dec)} recorded the largest {decrease_html} at {percent_html(val_dec)}, while {cat_html(cat_inc)} showed the largest {increase_html} at {percent_html(val_inc)}, respectively, compared to the previous month. Overall, the {total_html} change is {'decreased' if total_percent < 0 else 'increased'} by {total_percent_html(total_percent)}"
    elif len(neg_percents) == 0 and len(pos_percents) > 0:
        if len(pos_percents) == 1:
            cat, val = pos_percents[0]
            desc = f"{cat_html(cat)} recorded the largest {increase_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
        else:
            (cat1, val1), (cat2, val2) = pos_percents[:2]
            desc = f"{cat_html(cat1)} and {cat_html(cat2)} recorded the largest {increase_html} at {percent_html(val1)} and {percent_html(val2)}, respectively, compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif len(pos_percents) == 0 and len(neg_percents) > 0:
        if len(neg_percents) == 1:
            cat, val = neg_percents[0]
            desc = f"{cat_html(cat)} recorded the largest {decrease_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
        else:
            (cat1, val1), (cat2, val2) = neg_percents[:2]
            desc = f"{cat_html(cat1)} and {cat_html(cat2)} recorded the largest {decrease_html} at {percent_html(val1)} and {percent_html(val2)}, respectively, compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    else:
        desc_parts = []
        if len(neg_percents) > 0:
            if len(neg_percents) == 1:
                cat, val = neg_percents[0]
                desc_parts.append(f"{cat_html(cat)} recorded the largest {decrease_html} at {percent_html(val)}")
            else:
                (cat1, val1), (cat2, val2) = neg_percents[:2]
                desc_parts.append(f"{cat_html(cat1)} and {cat_html(cat2)} recorded the largest {decrease_html} at {percent_html(val1)} and {percent_html(val2)}")
        if len(pos_percents) > 0:
            if len(pos_percents) == 1:
                cat, val = pos_percents[0]
                desc_parts.append(f"{cat_html(cat)} showed the largest {increase_html} at {percent_html(val)}")
            else:
                (cat1, val1), (cat2, val2) = pos_percents[:2]
                desc_parts.append(f"{cat_html(cat1)} and {cat_html(cat2)} showed the largest {increase_html} at {percent_html(val1)} and {percent_html(val2)}")
        desc = ", while ".join(desc_parts) + f", respectively, compared to the previous month. Overall, the {total_html} change is {'decreased' if total_percent < 0 else 'increased'} by {total_percent_html(total_percent)}"

st.markdown(f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{desc}</div>", unsafe_allow_html=True)
st.markdown("<div style='height: 19rem'></div>", unsafe_allow_html=True)

# --- STACKED COLUMN CHART TỪ EXCEL: OVERALL COST SPENT PER TEAM BY MONTH (MVND)-----------

if 'Months' in df_excel6.columns:
    x_col_month_team = 'Months'
else:
    x_col_month_team = df_excel6.columns[0]
team_cols_month = [col for col in df_excel6.columns if col != x_col_month_team]
fig_stack_excel_month_team = go.Figure()
color_palette_team = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
def safe_to_int_team(v):
    try:
        return int(round(float(v)))
    except:
        return 0
for i, team in enumerate(team_cols_month):
    y_values = df_excel6[team].apply(safe_to_int_team).tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette_team[i % len(color_palette_team)]
    fig_stack_excel_month_team.add_trace(go.Bar(
        name=team,
        x=df_excel6[x_col_month_team],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
    ))
totals_team = df_excel6[team_cols_month].applymap(safe_to_int_team).sum(axis=1)
totals_offset_team = totals_team + totals_team * 0.04
for i, (x, y, t) in enumerate(zip(df_excel6[x_col_month_team], totals_offset_team, totals_team)):
    fig_stack_excel_month_team.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_excel_month_team.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL COST SPENT PER TEAM BY MONTH (MVND)",
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Months", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
# --- BOX SO SÁNH PHẦN TRĂM ---
idx_w_team = len(df_excel6) - 1
idx_w1_team = len(df_excel6) - 2
w_label_team = df_excel6[x_col_month_team].iloc[idx_w_team]
active_teams = []
percent_changes_team = {}
team_positions = {}
cumulative_height_team = 0
for team in team_cols_month:
    count_w = safe_to_int_team(df_excel6[team].iloc[idx_w_team])
    if count_w <= 0:
        continue
    count_w1 = safe_to_int_team(df_excel6[team].iloc[idx_w1_team])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_teams.append(team)
    percent_changes_team[team] = percent
    team_positions[team] = cumulative_height_team + count_w / 2
    cumulative_height_team += count_w
if active_teams:
    total_height_team = cumulative_height_team
    x_vals_team = list(df_excel6[x_col_month_team])
    x_idx_team = x_vals_team.index(w_label_team)
    x_offset_team = x_idx_team + 0.8
    sorted_teams = sorted(active_teams, key=lambda x: team_positions[x])
    for i, team in enumerate(sorted_teams):
        percent = percent_changes_team[team]
        if percent > 0:
            percent_text = f"M vs M-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"M vs M-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "M vs M-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = team_positions[team]
        spacing_factor = 2
        y_box = y_col + (total_height_team * spacing_factor * (i - len(sorted_teams)/2))
        fig_stack_excel_month_team.add_annotation(
            x=w_label_team, y=y_col,
            ax=x_offset_team, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_excel_month_team.add_annotation(
            x=x_offset_team, y=y_box,
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
# --- Thêm gridline dọc gạch đứt phân chia các tháng ---
y_max_team = totals_offset_team.max() * 1.05
vertical_lines_team = []
num_months_team = len(df_excel6[x_col_month_team])
for i in range(1, num_months_team):
    vertical_lines_team.append(dict(
        type="line",
        xref="x",
        yref="y",
        x0=i - 0.5,
        x1=i - 0.5,
        y0=0,
        y1=y_max_team,
        line=dict(
            color="gray",
            width=1,
            dash="dot"
        ),
        layer="below"
    ))
fig_stack_excel_month_team.update_layout(shapes=vertical_lines_team)
st.plotly_chart(fig_stack_excel_month_team)
st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

# --- AUTO DESCRIPTION CHO STACKED COLUMN CHART COST THEO TEAM BY MONTH (EXCEL) ---
def get_team_name_month(team):
    # Mapping tên đẹp nếu muốn
    return team

total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"

percent_changes_team_desc = {}
for team in team_cols_month:
    count_w = safe_to_int_team(df_excel6[team].iloc[idx_w_team])
    count_w1 = safe_to_int_team(df_excel6[team].iloc[idx_w1_team])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_team_desc[team] = percent

sum_w = sum([safe_to_int_team(df_excel6[team].iloc[idx_w_team]) for team in team_cols_month])
sum_w1 = sum([safe_to_int_team(df_excel6[team].iloc[idx_w1_team]) for team in team_cols_month])
if sum_w1 == 0:
    total_percent = 100 if sum_w > 0 else 0
else:
    total_percent = ((sum_w - sum_w1) / sum_w1) * 100

neg_percents = [(k, v) for k, v in percent_changes_team_desc.items() if v < 0]
pos_percents = [(k, v) for k, v in percent_changes_team_desc.items() if v > 0]
zero_percents = [(k, v) for k, v in percent_changes_team_desc.items() if v == 0]

neg_percents.sort(key=lambda x: x[1])
pos_percents.sort(key=lambda x: -x[1])

def team_html(team):
    return f"<span style='font-weight:bold; color:#d62728'>{get_team_name_month(team)}</span>"
def percent_html(val):
    return f"<span style='font-weight:bold; color:#d62728'>{abs(val):.1f}%</span>"
def total_percent_html(val):
    return f"<span style='font-weight:bold; color:#d62728'>{abs(val):.1f}%</span>"

if len(percent_changes_team_desc) == 1:
    # Chỉ có 1 team
    team, val = list(percent_changes_team_desc.items())[0]
    if val > 0:
        desc = f"{team_html(team)} recorded the largest {increase_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif val < 0:
        desc = f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    else:
        desc = f"{team_html(team)} remains unchanged compared to the previous month. Overall, the {total_html} change is 0%"
else:
    if len(neg_percents) == 1 and len(pos_percents) == 0:
        team, val = neg_percents[0]
        desc = f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)}, while the other teams remain unchanged, compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    elif len(pos_percents) == 1 and len(neg_percents) == 0:
        team, val = pos_percents[0]
        desc = f"{team_html(team)} recorded the largest {increase_html} at {percent_html(val)}, while the other teams remain unchanged, compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif len(neg_percents) == 1 and len(pos_percents) == 1:
        team_dec, val_dec = neg_percents[0]
        team_inc, val_inc = pos_percents[0]
        desc = f"{team_html(team_dec)} recorded the largest {decrease_html} at {percent_html(val_dec)}, while {team_html(team_inc)} showed the largest {increase_html} at {percent_html(val_inc)}, respectively, compared to the previous month. Overall, the {total_html} change is {'decreased' if total_percent < 0 else 'increased'} by {total_percent_html(total_percent)}"
    elif len(neg_percents) == 0 and len(pos_percents) > 0:
        if len(pos_percents) == 1:
            team, val = pos_percents[0]
            desc = f"{team_html(team)} recorded the largest {increase_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
        else:
            (team1, val1), (team2, val2) = pos_percents[:2]
            desc = f"{team_html(team1)} and {team_html(team2)} recorded the largest {increase_html} at {percent_html(val1)} and {percent_html(val2)}, respectively, compared to the previous month. Overall, the {total_html} change is {increase_html} by {total_percent_html(total_percent)}"
    elif len(pos_percents) == 0 and len(neg_percents) > 0:
        if len(neg_percents) == 1:
            team, val = neg_percents[0]
            desc = f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)} compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
        else:
            (team1, val1), (team2, val2) = neg_percents[:2]
            desc = f"{team_html(team1)} and {team_html(team2)} recorded the largest {decrease_html} at {percent_html(val1)} and {percent_html(val2)}, respectively, compared to the previous month. Overall, the {total_html} change is {decrease_html} by {total_percent_html(total_percent)}"
    else:
        desc_parts = []
        if len(neg_percents) > 0:
            if len(neg_percents) == 1:
                team, val = neg_percents[0]
                desc_parts.append(f"{team_html(team)} recorded the largest {decrease_html} at {percent_html(val)}")
            else:
                (team1, val1), (team2, val2) = neg_percents[:2]
                desc_parts.append(f"{team_html(team1)} and {team_html(team2)} recorded the largest {decrease_html} at {percent_html(val1)} and {percent_html(val2)}")
        if len(pos_percents) > 0:
            if len(pos_percents) == 1:
                team, val = pos_percents[0]
                desc_parts.append(f"{team_html(team)} showed the largest {increase_html} at {percent_html(val)}")
            else:
                (team1, val1), (team2, val2) = pos_percents[:2]
                desc_parts.append(f"{team_html(team1)} and {team_html(team2)} showed the largest {increase_html} at {percent_html(val1)} and {percent_html(val2)}")
        desc = ", while ".join(desc_parts) + f", respectively, compared to the previous month. Overall, the {total_html} change is {'decreased' if total_percent < 0 else 'increased'} by {total_percent_html(total_percent)}"

st.markdown(f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{desc}</div>", unsafe_allow_html=True)
st.markdown("<div style='height: 14rem'></div>", unsafe_allow_html=True)

# --- STACKED COLUMN CHART TỪ EXCEL: OVERALL COST SPENT PER BANNER BY MONTH (MVND)-----------

if 'Months' in df_excel7.columns:
    x_col_month_banner = 'Months'
else:
    x_col_month_banner = df_excel7.columns[0]
banner_cols_month = [col for col in df_excel7.columns if col != x_col_month_banner]
fig_stack_excel_month_banner = go.Figure()
color_palette_banner = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
def safe_to_int_banner(v):
    try:
        return int(round(float(v)))
    except:
        return 0
for i, banner in enumerate(banner_cols_month):
    y_values = df_excel7[banner].apply(safe_to_int_banner).tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette_banner[i % len(color_palette_banner)]
    fig_stack_excel_month_banner.add_trace(go.Bar(
        name=banner,
        x=df_excel7[x_col_month_banner],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=9),
        marker_color=color
    ))
totals_banner = df_excel7[banner_cols_month].applymap(safe_to_int_banner).sum(axis=1)
totals_offset_banner = totals_banner + totals_banner * 0.04
for i, (x, y, t) in enumerate(zip(df_excel7[x_col_month_banner], totals_offset_banner, totals_banner)):
    fig_stack_excel_month_banner.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
fig_stack_excel_month_banner.update_layout(
    barmode='stack',
    title=dict(
        text="OVERALL COST SPENT PER BANNER BY MONTH (MVND)",
        x=0.5,
        y=1.0,
        xanchor='center',
        yanchor='top',
        font=dict(size=28)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black'),
        title=dict(text="Months", font=dict(color='black'))
    ),
    yaxis=dict(
        tickfont=dict(color='black')
    )
)
# --- BOX SO SÁNH PHẦN TRĂM ---
idx_w_banner = len(df_excel7) - 1
idx_w1_banner = len(df_excel7) - 2
w_label_banner = df_excel7[x_col_month_banner].iloc[idx_w_banner]
active_banners = []
percent_changes_banner = {}
banner_positions = {}
cumulative_height_banner = 0
for banner in banner_cols_month:
    count_w = safe_to_int_banner(df_excel7[banner].iloc[idx_w_banner])
    if count_w <= 0:
        continue
    count_w1 = safe_to_int_banner(df_excel7[banner].iloc[idx_w1_banner])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    active_banners.append(banner)
    percent_changes_banner[banner] = percent
    banner_positions[banner] = cumulative_height_banner + count_w / 2
    cumulative_height_banner += count_w
if active_banners:
    total_height_banner = cumulative_height_banner
    x_vals_banner = list(df_excel7[x_col_month_banner])
    x_idx_banner = x_vals_banner.index(w_label_banner)
    x_offset_banner = x_idx_banner + 0.8
    sorted_banners = sorted(active_banners, key=lambda x: banner_positions[x])
    for i, banner in enumerate(sorted_banners):
        percent = percent_changes_banner[banner]
        if percent > 0:
            percent_text = f"M vs M-1: +{percent:.1f}%"
            bgcolor = "#f2c795"
        elif percent < 0:
            percent_text = f"M vs M-1: -{abs(percent):.1f}%"
            bgcolor = "#abf3ab"
        else:
            percent_text = "M vs M-1: 0.0%"
            bgcolor = "#f2c795"
        y_col = banner_positions[banner]
        spacing_factor = 2
        y_box = y_col + (total_height_banner * spacing_factor * (i - len(sorted_banners)/2))
        fig_stack_excel_month_banner.add_annotation(
            x=w_label_banner, y=y_col,
            ax=x_offset_banner, ay=y_box,
            xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor="black"
        )
        fig_stack_excel_month_banner.add_annotation(
            x=x_offset_banner, y=y_box,
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
# --- Thêm gridline dọc gạch đứt phân chia các tháng ---
y_max_banner = totals_offset_banner.max() * 1.05
vertical_lines_banner = []
num_months_banner = len(df_excel7[x_col_month_banner])
for i in range(1, num_months_banner):
    vertical_lines_banner.append(dict(
        type="line",
        xref="x",
        yref="y",
        x0=i - 0.5,
        x1=i - 0.5,
        y0=0,
        y1=y_max_banner,
        line=dict(
            color="gray",
            width=1,
            dash="dot"
        ),
        layer="below"
    ))
fig_stack_excel_month_banner.update_layout(shapes=vertical_lines_banner)
st.plotly_chart(fig_stack_excel_month_banner)
st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

# --- AUTO DESCRIPTION ---
percent_changes_banner_desc = {}
for banner in banner_cols_month:
    count_w = safe_to_int_banner(df_excel7[banner].iloc[idx_w_banner])
    count_w1 = safe_to_int_banner(df_excel7[banner].iloc[idx_w1_banner])
    if count_w1 == 0:
        percent = 100 if count_w > 0 else 0
    else:
        percent = ((count_w - count_w1) / count_w1) * 100
    percent_changes_banner_desc[banner] = percent

# X% là số âm lớn nhất (giảm nhiều nhất), A là tên banner đó
neg_percents = {k: v for k, v in percent_changes_banner_desc.items() if v < 0}
if neg_percents:
    A = min(neg_percents, key=neg_percents.get)
    X = neg_percents[A]
else:
    A = min(percent_changes_banner_desc, key=percent_changes_banner_desc.get)
    X = percent_changes_banner_desc[A]

# Z% là số dương lớn nhất (tăng nhiều nhất), B là tên banner đó
pos_percents = {k: v for k, v in percent_changes_banner_desc.items() if v > 0}
if pos_percents:
    B = max(pos_percents, key=pos_percents.get)
    Z = pos_percents[B]
else:
    B = max(percent_changes_banner_desc, key=percent_changes_banner_desc.get)
    Z = percent_changes_banner_desc[B]

sum_w = sum([safe_to_int_banner(df_excel7[banner].iloc[idx_w_banner]) for banner in banner_cols_month])
sum_w1 = sum([safe_to_int_banner(df_excel7[banner].iloc[idx_w1_banner]) for banner in banner_cols_month])
if sum_w1 == 0:
    Y = 100 if sum_w > 0 else 0
else:
    Y = ((sum_w - sum_w1) / sum_w1) * 100

C = 'increased' if Y > 0 else 'decreased'

A_html = f"<span style='color:#d62728; font-weight:bold'>{A}</span>"
B_html = f"<span style='color:#d62728; font-weight:bold'>{B}</span>"
C_html = f"<span style='color:#111; font-weight:bold'>{C}</span>"
decrease_html = "<span style='font-weight:bold; color:#111'>decrease</span>"
increase_html = "<span style='font-weight:bold; color:#111'>increase</span>"
total_html = "<span style='font-weight:bold; color:#d62728'>Total</span>"

# Đếm số lượng tăng/giảm/không đổi
change_values = list(percent_changes_banner_desc.values())
num_neg = sum(1 for v in change_values if v < 0)
num_pos = sum(1 for v in change_values if v > 0)
num_zero = sum(1 for v in change_values if abs(v) < 1e-6)

# Sinh câu mô tả theo logic mới
if num_neg == 1 and num_zero == len(banner_cols_month) - 1:
    # Chỉ có 1 giảm, còn lại không đổi
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%, while other banners remains unchanged, compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == 1 and num_zero == len(banner_cols_month) - 1:
    # Chỉ có 1 tăng, còn lại không đổi
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%, while other banners remains unchanged, compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_neg == len(banner_cols_month):
    # Tất cả đều giảm
    description = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}% compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_pos == len(banner_cols_month):
    # Tất cả đều tăng
    description = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}% compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
elif num_zero == len(banner_cols_month):
    # Tất cả không đổi
    description = f"All banners remain unchanged compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"
else:
    # Trường hợp mặc định như cũ
    if abs(X) < 1e-6:
        decrease_text = f"{A_html} remains unchanged"
    else:
        decrease_text = f"{A_html} recorded the largest {decrease_html} at {abs(X):.1f}%"
    if abs(Z) < 1e-6:
        increase_text = f"{B_html} remains unchanged"
    else:
        increase_text = f"{B_html} show the highest {increase_html} with {abs(Z):.1f}%"
    description = f"{decrease_text}, while {increase_text}, compared to the previous month. Overall, the {total_html} change is {C_html} by {abs(Y):.1f}%"

st.markdown(
    f"<div style='font-size:18px; color:#444; text-align:center; margin-bottom:2rem'>{description}</div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --------------------------- BẢNG Actual cost ---------------------------

import matplotlib.colors as mcolors
import numpy as np

# Làm tròn số cho toàn bộ các cột số, không lấy số thập phân
df_excel8_rounded = df_excel8.copy()
num_cols = df_excel8_rounded.select_dtypes(include='number').columns
df_excel8_rounded[num_cols] = df_excel8_rounded[num_cols].round(0).astype(int)

# Tạo colormap từ đỏ nhạt đến đỏ đậm
red_cmap = mcolors.LinearSegmentedColormap.from_list("red_grad", ["#fff0f0", "#e57373"])

def red_gradient(val, vmin=None, vmax=None):
    try:
        if abs(float(val)) < 1e-8:
            return 'background-color: white; color: black;'
        if vmin is None or vmax is None:
            vmin = df_excel8_rounded.select_dtypes(include='number').min().min()
            vmax = df_excel8_rounded.select_dtypes(include='number').max().max()
        norm = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0
        color = mcolors.to_hex(red_cmap(norm))
        return f'background-color: {color}; color: black;'
    except:
        return ''

# Lấy các cột số
num_cols = df_excel8_rounded.select_dtypes(include='number').columns

# Tính min/max cho gradient
vmin = df_excel8_rounded[num_cols].min().min()
vmax = df_excel8_rounded[num_cols].max().max()

styled_df = df_excel8_rounded.style.applymap(lambda v: red_gradient(v, vmin, vmax), subset=num_cols)

num_rows = df_excel8_rounded.shape[0]
row_height = 35  # hoặc 35 tùy font
total_height = num_rows * row_height + 35  # +38 cho header

st.markdown("""
<h3 style='text-align: center; margin-top: 2rem; margin-bottom: 1rem;'>TOTAL ACTUAL COST CONFIRMED BY CATEGORY (MVND)</h3>
""", unsafe_allow_html=True)
st.dataframe(styled_df, use_container_width=True, height=total_height)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --------------------------- STACKED BAR CHART CHO BẢNG Actual cost ---------------------------

if df_excel8.shape[1] > 1:
    # Xác định cột category (cột đầu tiên) và các cột tháng
    cat_col = df_excel8.columns[0]
    month_cols = list(df_excel8.columns[1:])
    # Làm tròn số cho các cột số
    df_excel8_chart = df_excel8.copy()
    df_excel8_chart[month_cols] = df_excel8_chart[month_cols].round(0).astype(int)
    # Tạo stacked bar chart
import plotly.graph_objects as go
fig_excel8 = go.Figure()
color_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
    '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
    '#45b39d', '#f1948a', '#34495e', '#f39c12'
]
for i, month in enumerate(month_cols):
    y_values = df_excel8_chart[month].tolist()
    text_labels = [str(v) if v != 0 else "" for v in y_values]
    color = color_palette[i % len(color_palette)]
    fig_excel8.add_trace(go.Bar(
        name=month,
        x=df_excel8_chart[cat_col],
        y=y_values,
        text=text_labels,
        textposition="inside",
        texttemplate="%{text}",
        textangle=0,
        textfont=dict(size=10),
        marker_color=color
    ))
fig_excel8.update_layout(
    barmode='stack',
    title=dict(
        text="TOTAL ACTUAL COST CONFIRMED BY CATEGORY (MVND)",
        x=0.5,
        y=1,
        xanchor='center',
        yanchor='top',
        font=dict(size=24)
    ),
    width=1400,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.055,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
    tickangle=-45,  # hoặc -60 nếu muốn nghiêng nhiều hơn
    tickfont=dict(size=13, color='black')
    )
)
    # --- Thêm giá trị total trên đầu mỗi cột (category) ---
totals = df_excel8_chart[month_cols].sum(axis=1)
totals_offset = totals + totals * 0.04
for i, (x, y, t) in enumerate(zip(df_excel8_chart[cat_col], totals_offset, totals)):
    fig_excel8.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
        borderpad=4,
        bordercolor="#e74c3c",
        borderwidth=0
    )
st.plotly_chart(fig_excel8, use_container_width=True)
st.markdown("<div style='height: 5rem'></div>", unsafe_allow_html=True)

# --------------------------- BẢNG Actual cost per cat 2 ---------------------------

import matplotlib.colors as mcolors
import numpy as np

# Làm tròn số cho toàn bộ các cột số, không lấy số thập phân
df_excel9_rounded = df_excel9.copy().fillna(0)
num_cols9 = df_excel9_rounded.select_dtypes(include='number').columns
df_excel9_rounded[num_cols9] = df_excel9_rounded[num_cols9].round(1)  # Giữ 1 chữ số thập phân

# Tạo colormap từ đỏ nhạt đến đỏ đậm
red_cmap9 = mcolors.LinearSegmentedColormap.from_list("red_grad9", ["#fff0f0", "#e57373"])

def red_gradient9(val, vmin=None, vmax=None):
    try:
        if abs(float(val)) < 1e-8:
            return 'background-color: white; color: black;'
        if vmin is None or vmax is None:
            vmin = df_excel9_rounded.select_dtypes(include='number').min().min()
            vmax = df_excel9_rounded.select_dtypes(include='number').max().max()
        norm = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0
        color = mcolors.to_hex(red_cmap9(norm))
        return f'background-color: {color}; color: black;'
    except:
        return ''

# Lấy các cột số
num_cols9 = df_excel9_rounded.select_dtypes(include='number').columns

# Tính min/max cho gradient
vmin9 = df_excel9_rounded[num_cols9].min().min()
vmax9 = df_excel9_rounded[num_cols9].max().max()

styled_df9 = (
    df_excel9_rounded
    .style
    .format({col: (lambda x: "0" if x == 0 else f"{x:.1f}") for col in num_cols9})
    .applymap(lambda v: red_gradient9(v, vmin9, vmax9), subset=num_cols9)
)

num_rows9 = df_excel9_rounded.shape[0]
row_height9 = 35
total_height9 = num_rows9 * row_height9 + 35

st.markdown("""
<h3 style='text-align: center; margin-top: 2rem; margin-bottom: 1rem;'>AVERAGE ACTUAL COST PER CONFIRMED TICKET BY CATEGORY (MVND)</h3>
""", unsafe_allow_html=True)
st.dataframe(styled_df9, use_container_width=True, height=total_height9)
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

# --------------------------- BẢNG Actual cost per sub region 1 ---------------------------

df_excel10_rounded = df_excel10.copy()
num_cols10 = df_excel10_rounded.select_dtypes(include='number').columns
df_excel10_rounded[num_cols10] = df_excel10_rounded[num_cols10].round(0).astype(int)

red_cmap10 = mcolors.LinearSegmentedColormap.from_list("red_grad10", ["#fff0f0", "#e57373"])

def red_gradient10(val, vmin=None, vmax=None):
    try:
        if abs(float(val)) < 1e-8:
            return 'background-color: white; color: black;'
        if vmin is None or vmax is None:
            vmin = df_excel10_rounded.select_dtypes(include='number').min().min()
            vmax = df_excel10_rounded.select_dtypes(include='number').max().max()
        norm = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0
        color = mcolors.to_hex(red_cmap10(norm))
        return f'background-color: {color}; color: black;'
    except:
        return ''

num_cols10 = df_excel10_rounded.select_dtypes(include='number').columns
vmin10 = df_excel10_rounded[num_cols10].min().min()
vmax10 = df_excel10_rounded[num_cols10].max().max()

styled_df10 = df_excel10_rounded.style.applymap(lambda v: red_gradient10(v, vmin10, vmax10), subset=num_cols10)

num_rows10 = df_excel10_rounded.shape[0]
row_height10 = 35
total_height10 = num_rows10 * row_height10 + 35

st.markdown("""
<h3 style='text-align: center; margin-top: 2rem; margin-bottom: 1rem;'>TOTAL ACTUAL COST CONFIRMED BY SUB-REGION (MVND)</h3>
""", unsafe_allow_html=True)
st.dataframe(styled_df10, use_container_width=True, height=total_height10)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --------------------------- STACKED BAR CHART CHO BẢNG Actual cost per sub region 1 ---------------------------

if df_excel10.shape[1] > 1:
    cat_col10 = df_excel10.columns[0]
    month_cols10 = list(df_excel10.columns[1:])
    df_excel10_chart = df_excel10.copy()
    df_excel10_chart[month_cols10] = df_excel10_chart[month_cols10].round(0).astype(int)
    import plotly.graph_objects as go
    fig_excel10 = go.Figure()
    color_palette10 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f5b041',
        '#229954', '#0bf4a3', '#e74c3c', '#f7dc6f', '#a569bd',
        '#45b39d', '#f1948a', '#34495e', '#f39c12'
    ]
    for i, month in enumerate(month_cols10):
        y_values = df_excel10_chart[month].tolist()
        text_labels = [str(v) if v != 0 else "" for v in y_values]
        color = color_palette10[i % len(color_palette10)]
        fig_excel10.add_trace(go.Bar(
            name=month,
            x=df_excel10_chart[cat_col10],
            y=y_values,
            text=text_labels,
            textposition="inside",
            texttemplate="%{text}",
            textangle=0,
            textfont=dict(size=10),
            marker_color=color
        ))
    fig_excel10.update_layout(
        barmode='stack',
        title=dict(
            text="TOTAL ACTUAL COST CONFIRMED BY SUB-REGION (MVND)",
            x=0.5,
            y=1,
            xanchor='center',
            yanchor='top',
            font=dict(size=24)
        ),
        width=1400,
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.055,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=13, color='black')
        )
    )
    # --- Thêm giá trị total trên đầu mỗi cột (category) ---
    totals10 = df_excel10_chart[month_cols10].sum(axis=1)
    totals_offset10 = totals10 + totals10 * 0.04
    for i, (x, y, t) in enumerate(zip(df_excel10_chart[cat_col10], totals_offset10, totals10)):
        fig_excel10.add_annotation(
            x=x,
            y=y,
            text=f"<span style='color:#e74c3c; font-weight:bold'>{int(t)}</span>",
            showarrow=False,
            font=dict(size=20, color="#e74c3c"),
            align="center",
            xanchor="center",
            yanchor="bottom",
            #bgcolor="rgba(255,255,0,0.77)",
            borderpad=4,
            bordercolor="#e74c3c",
            borderwidth=0
        )
    st.plotly_chart(fig_excel10, use_container_width=True)
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

# --------------------------- BẢNG Actual cost per sub region 2 ---------------------------

df_excel11_rounded = df_excel11.copy().fillna(0)
num_cols11 = df_excel11_rounded.select_dtypes(include='number').columns
df_excel11_rounded[num_cols11] = df_excel11_rounded[num_cols11].round(1)

# Tạo colormap từ đỏ nhạt đến đỏ đậm
red_cmap11 = mcolors.LinearSegmentedColormap.from_list("red_grad11", ["#fff0f0", "#e57373"])

def red_gradient11(val, vmin=None, vmax=None):
    try:
        if abs(float(val)) < 1e-8:
            return 'background-color: white; color: black;'
        if vmin is None or vmax is None:
            vmin = df_excel11_rounded.select_dtypes(include='number').min().min()
            vmax = df_excel11_rounded.select_dtypes(include='number').max().max()
        norm = (float(val) - vmin) / (vmax - vmin) if vmax > vmin else 0
        color = mcolors.to_hex(red_cmap11(norm))
        return f'background-color: {color}; color: black;'
    except:
        return ''

num_cols11 = df_excel11_rounded.select_dtypes(include='number').columns
vmin11 = df_excel11_rounded[num_cols11].min().min()
vmax11 = df_excel11_rounded[num_cols11].max().max()

styled_df11 = (
    df_excel11_rounded
    .style
    .format({col: (lambda x: "0" if x == 0 else f"{x:.1f}") for col in num_cols11})
    .applymap(lambda v: red_gradient11(v, vmin11, vmax11), subset=num_cols11)
)

num_rows11 = df_excel11_rounded.shape[0]
row_height11 = 35
total_height11 = num_rows11 * row_height11 + 35

st.markdown("""
<h3 style='text-align: center; margin-top: 2rem; margin-bottom: 1rem;'>ACTUAL AVERAGE COST PER CONFIRMED TICKET BY SUB-REGION (MVND)</h3>
""", unsafe_allow_html=True)
st.dataframe(styled_df11, use_container_width=True, height=total_height11)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

# --------------------------- BẢNG HELPDESK TICKET D-1 ---------------------------

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

# --- Tạo hàng "Avg. across all" ---
avg_row = {}
for col in pivot.columns:
    avg_row[col] = int(round(df['processing_time'].mean())) if col == 'Across all category' else int(round(df[df['category_name'] == col]['processing_time'].mean()))
avg_row = pd.DataFrame([avg_row], index=['Avg. across all'])

# --- Sort theo 'Across all category' tăng dần, rồi nối hàng "Avg. across all" vào cuối ---
pivot = pivot.sort_values('Across all category', ascending=True)
pivot = pd.concat([pivot, avg_row])

# Đổi tên index thành cột 'Regions' (và đặt heading rõ ràng)
pivot = pivot.reset_index()
pivot.rename(columns={pivot.columns[0]: 'Regions'}, inplace=True)

# Chỉ lấy các cột số để tính min/max
num_cols = pivot.select_dtypes(include=[np.number]).columns
vmin = pivot[num_cols].min().min()
vmax = pivot[num_cols].max().max()

def style_regions(val):
    return 'color: black;'

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

styled = pivot.style.applymap(lambda v: color_scale(v, vmin, vmax), subset=pivot.columns.difference(['Regions'])) \
                .applymap(style_regions, subset=['Regions']) \
                .set_properties(**{'text-align': 'center', 'color': 'black'})
num_rows = len(pivot.index)
row_height = 35
total_height = (num_rows + 1) * row_height
st.dataframe(styled, use_container_width=True, height=total_height)
st.markdown("<div style='height: 22rem'></div>", unsafe_allow_html=True)


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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)


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
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)


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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

#------------------BẢNG EQUIPMENT FREQUENCY-----------------------------------------------

latest_ticket_col = df_excel12.columns[0]
mall_col = df_excel12.columns[1]
cate_col = df_excel12.columns[2]
equip_col = df_excel12.columns[4]
freq_col = df_excel12.columns[5]

# Làm tròn cột freq_col về 1 chữ số thập phân (chuẩn round-half-up)
freq_rounded = df_excel12[freq_col].apply(lambda x: round(x, 1) if pd.notnull(x) else x)

df_latest_ticket = pd.DataFrame({
    "Latest incident ticket": df_excel12[latest_ticket_col],
    "Mall": df_excel12[mall_col],
    "Category": df_excel12[cate_col],
    "Equipment": df_excel12[equip_col],
    "Average incident frequency by month": freq_rounded,
})

st.markdown("<h3 style='text-align: center; margin-top: 2rem;'>TOP EQUIPMENT BY MONTHLY INCIDENT RATE</h3>", unsafe_allow_html=True)
st.dataframe(df_latest_ticket, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

#------------------------ GO MALL - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1 = df_excel13.iloc[0, 1:]
row2 = df_excel13.iloc[1, 1:]
x_labels = df_excel13.columns[1:].tolist()
y_line = [round(v) if pd.notnull(v) else 0 for v in row1.values.tolist()]
y_bar = [round(v) if pd.notnull(v) else 0 for v in row2.values.tolist()]

fig_combo = go.Figure()

# Line chart cho dòng 1
fig_combo.add_trace(go.Scatter(
    x=x_labels,
    y=y_line,
    mode='lines+markers',
    name='Go Mall - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line],  # Ẩn label nếu = 0
    textposition='top center'
))

# Bar chart cho dòng 2
fig_combo.add_trace(go.Bar(
    x=x_labels,
    y=y_bar,
    name='GO Mall - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar],  # Ẩn label nếu = 0
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo.update_layout(
    title={
        'text': 'Go Mall - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=650,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo, use_container_width=True)
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

#------------------------ HYPER - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1_hyper = df_excel13.iloc[2, 1:]  # Dòng 3 (index 2)
row2_hyper = df_excel13.iloc[3, 1:]  # Dòng 4 (index 3)
x_labels_hyper = df_excel13.columns[1:].tolist()
y_line_hyper = [round(v) if pd.notnull(v) else 0 for v in row1_hyper.values.tolist()]
y_bar_hyper = [round(v) if pd.notnull(v) else 0 for v in row2_hyper.values.tolist()]

fig_combo_hyper = go.Figure()

# Line chart cho dòng 3
fig_combo_hyper.add_trace(go.Scatter(
    x=x_labels_hyper,
    y=y_line_hyper,
    mode='lines+markers',
    name='Hyper - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line_hyper],
    textposition='top center'
))

# Bar chart cho dòng 4
fig_combo_hyper.add_trace(go.Bar(
    x=x_labels_hyper,
    y=y_bar_hyper,
    name='Hyper - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar_hyper],
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo_hyper.update_layout(
    title={
        'text': 'Hyper - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=650,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo_hyper, use_container_width=True)
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

#------------------------ TOP - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1_top = df_excel13.iloc[4, 1:]  # Dòng 5 (index 4) - Budget
row2_top = df_excel13.iloc[5, 1:]  # Dòng 6 (index 5) - Actual Cost
x_labels_top = df_excel13.columns[1:].tolist()
y_line_top = [round(v) if pd.notnull(v) else 0 for v in row1_top.values.tolist()]
y_bar_top = [round(v) if pd.notnull(v) else 0 for v in row2_top.values.tolist()]

fig_combo_top = go.Figure()

# Line chart cho dòng 5
fig_combo_top.add_trace(go.Scatter(
    x=x_labels_top,
    y=y_line_top,
    mode='lines+markers',
    name='Top - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line_top],
    textposition='top center'
))

# Bar chart cho dòng 6
fig_combo_top.add_trace(go.Bar(
    x=x_labels_top,
    y=y_bar_top,
    name='Top - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar_top],
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo_top.update_layout(
    title={
        'text': 'Top - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=650,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo_top, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

#------------------------ MINIGO - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1_minigo = df_excel13.iloc[6, 1:]  # Dòng 7 (index 6) - Budget
row2_minigo = df_excel13.iloc[7, 1:]  # Dòng 8 (index 7) - Actual Cost
x_labels_minigo = df_excel13.columns[1:].tolist()
y_line_minigo = [round(v) if pd.notnull(v) else 0 for v in row1_minigo.values.tolist()]
y_bar_minigo = [round(v) if pd.notnull(v) else 0 for v in row2_minigo.values.tolist()]

fig_combo_minigo = go.Figure()

# Line chart cho dòng 7
fig_combo_minigo.add_trace(go.Scatter(
    x=x_labels_minigo,
    y=y_line_minigo,
    mode='lines+markers',
    name='Minigo - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line_minigo],
    textposition='top center'
))

# Bar chart cho dòng 8
fig_combo_minigo.add_trace(go.Bar(
    x=x_labels_minigo,
    y=y_bar_minigo,
    name='Minigo - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar_minigo],
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo_minigo.update_layout(
    title={
        'text': 'Minigo - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=650,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo_minigo, use_container_width=True)
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

#------------------------ NK - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1_nk = df_excel13.iloc[8, 1:]  # Dòng 9 (index 8) - Budget
row2_nk = df_excel13.iloc[9, 1:]  # Dòng 10 (index 9) - Actual Cost
x_labels_nk = df_excel13.columns[1:].tolist()
y_line_nk = [round(v) if pd.notnull(v) else 0 for v in row1_nk.values.tolist()]
y_bar_nk = [round(v) if pd.notnull(v) else 0 for v in row2_nk.values.tolist()]

fig_combo_nk = go.Figure()

# Line chart cho dòng 9
fig_combo_nk.add_trace(go.Scatter(
    x=x_labels_nk,
    y=y_line_nk,
    mode='lines+markers',
    name='NK - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line_nk],
    textposition='top center'
))

# Bar chart cho dòng 10
fig_combo_nk.add_trace(go.Bar(
    x=x_labels_nk,
    y=y_bar_nk,
    name='NK - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar_nk],
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo_nk.update_layout(
    title={
        'text': 'NGUYEN KIM - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=650,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo_nk, use_container_width=True)
st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)

#------------------------ CBS - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1_cbs = df_excel13.iloc[10, 1:]  # Dòng 11 (index 10) - Budget
row2_cbs = df_excel13.iloc[11, 1:]  # Dòng 12 (index 11) - Actual Cost
x_labels_cbs = df_excel13.columns[1:].tolist()
y_line_cbs = [round(v) if pd.notnull(v) else 0 for v in row1_cbs.values.tolist()]
y_bar_cbs = [round(v) if pd.notnull(v) else 0 for v in row2_cbs.values.tolist()]

fig_combo_cbs = go.Figure()

# Line chart cho dòng 11
fig_combo_cbs.add_trace(go.Scatter(
    x=x_labels_cbs,
    y=y_line_cbs,
    mode='lines+markers',
    name='CBS - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line_cbs],
    textposition='top center'
))

# Bar chart cho dòng 12
fig_combo_cbs.add_trace(go.Bar(
    x=x_labels_cbs,
    y=y_bar_cbs,
    name='CBS - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar_cbs],
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo_cbs.update_layout(
    title={
        'text': 'CBS - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=650,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo_cbs, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

#------------------------ KUBO - CHART ACTUAL COST VS BUDGET -------------------------------------------

import plotly.graph_objects as go

row1_kubo = df_excel13.iloc[12, 1:]  # Dòng 13 (index 12) - Budget
row2_kubo = df_excel13.iloc[13, 1:]  # Dòng 14 (index 13) - Actual Cost
x_labels_kubo = df_excel13.columns[1:].tolist()
y_line_kubo = [round(v) if pd.notnull(v) else 0 for v in row1_kubo.values.tolist()]
y_bar_kubo = [round(v) if pd.notnull(v) else 0 for v in row2_kubo.values.tolist()]

fig_combo_kubo = go.Figure()

# Line chart cho dòng 13
fig_combo_kubo.add_trace(go.Scatter(
    x=x_labels_kubo,
    y=y_line_kubo,
    mode='lines+markers',
    name='KUBO - Budget',
    line=dict(color='#EE0000', width=4),
    marker=dict(size=11, color='#e74c3c'),
    text=[str(v) if v != 0 else "" for v in y_line_kubo],
    textposition='top center'
))

# Bar chart cho dòng 14
fig_combo_kubo.add_trace(go.Bar(
    x=x_labels_kubo,
    y=y_bar_kubo,
    name='KUBO - Actual Cost',
    marker_color='#87CEFA',
    text=[str(v) if v != 0 else "" for v in y_bar_kubo],
    textposition='outside',
    textfont=dict(size=16, color='black', family='Arial', weight='bold')
))

fig_combo_kubo.update_layout(
    title={
        'text': 'KUBO - Actual Cost vs Budget (2025) (MVND)',
        'y': 1.0,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=28)
    },
    width=1300,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.08,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        tickfont=dict(color='black', size=14),
    ),
    yaxis=dict(
        tickfont=dict(color='black', size=14),
    )
)
st.plotly_chart(fig_combo_kubo, use_container_width=True)
st.markdown("<div style='height: 9rem'></div>", unsafe_allow_html=True)

#------------------------------------------------------------------------------------------------------------

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)
st.markdown("<div style='height: 70rem'></div>", unsafe_allow_html=True)

# -------------------------------NORTH 1------------------------------------------------------

st.markdown('<a id="north1"></a>', unsafe_allow_html=True)

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
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)


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
for i, (x, y, t) in enumerate(zip(df_table_north1["Tuần"], totals_offset, totals)):
    fig_stack_north1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_north1["Tuần"], totals_offset, totals)):
    fig_stack_priority_north1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)


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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)



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

st.markdown('<a id="north2"></a>', unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_north2["Tuần"], totals_offset2, totals2)):
    fig_stack_north2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_north2["Tuần"], totals_offset2, totals2)):
    fig_stack_priority_north2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 30rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)


st.markdown('<a id="north3"></a>', unsafe_allow_html=True)
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
for i, (x, y, t) in enumerate(zip(df_table_north3["Tuần"], totals_offset3, totals3)):
    fig_stack_north3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_north3["Tuần"], totals_offset3, totals3)):
    fig_stack_priority_north3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="center1"></a>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 1 - Huynh Bao Dang</h2>", unsafe_allow_html=True)
df_center1 = df[df['team_id'] == 18]
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

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

st.markdown("<div style='height: 4rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_center1["Tuần"], totals_offset, totals)):
    fig_stack_center1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_center1["Tuần"], totals_offset, totals)):
    fig_stack_priority_center1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 32rem'></div>", unsafe_allow_html=True)


st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="center2"></a>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 2 - Luu Duc Thach</h2>", unsafe_allow_html=True)
df_center2 = df[df['team_id'] == 3]
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_center2["Tuần"], totals_offset, totals)):
    fig_stack_center2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_priority_center2["Tuần"], totals_offset, totals)):
    fig_stack_priority_center2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="center3"></a>', unsafe_allow_html=True)
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
for i, (x, y, t) in enumerate(zip(df_table_center3["Tuần"], totals_offset, totals)):
    fig_stack_center3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_center3["Tuần"], totals_offset, totals)):
    fig_stack_priority_center3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 13rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="center4"></a>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>CENTER 4 & NINH BINH - Le Anh Sinh</h2>", unsafe_allow_html=True)
df_center4 = df[df['team_id'] == 20]
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_center4["Tuần"], totals_offset, totals)):
    fig_stack_center4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_priority_center4["Tuần"], totals_offset, totals)):
    fig_stack_priority_center4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 35rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="south1"></a>', unsafe_allow_html=True)

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
    st.plotly_chart(fig_gauge, key="gauge_chart_s1")
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
for i, (x, y, t) in enumerate(zip(df_table_south1["Tuần"], totals_offset, totals)):
    fig_stack_south1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_south1["Tuần"], totals_offset, totals)):
    fig_stack_priority_south1.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="south2"></a>', unsafe_allow_html=True)

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
    st.plotly_chart(fig_gauge, key="gauge_chart_s2")
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
for i, (x, y, t) in enumerate(zip(df_table_south2["Tuần"], totals_offset, totals)):
    fig_stack_south2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_south2["Tuần"], totals_offset, totals)):
    fig_stack_priority_south2.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 17rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="south3"></a>', unsafe_allow_html=True)

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
    st.plotly_chart(fig_gauge, key="gauge_chart_s3")
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
for i, (x, y, t) in enumerate(zip(df_table_south3["Tuần"], totals_offset, totals)):
    fig_stack_south3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_south3["Tuần"], totals_offset, totals)):
    fig_stack_priority_south3.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="south4"></a>', unsafe_allow_html=True)

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
    st.plotly_chart(fig_gauge, key="gauge_chart_s4")
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
for i, (x, y, t) in enumerate(zip(df_table_south4["Tuần"], totals_offset, totals)):
    fig_stack_south4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
for i, (x, y, t) in enumerate(zip(df_table_priority_south4["Tuần"], totals_offset, totals)):
    fig_stack_priority_south4.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

st.markdown(
"<hr style='border: 1.5px solid #222; margin: 30px 0;'>",
unsafe_allow_html=True
)

st.markdown("<div style='height: 0.2rem'></div>", unsafe_allow_html=True)

st.markdown('<a id="south5"></a>', unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;color: #ab3f3f;'>SOUTH 5 - Nguyen Ngoc Ho</h2>", unsafe_allow_html=True)
df_south5 = df[df['team_id'] == 23]
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

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
    st.plotly_chart(fig_gauge, key="gauge_chart_s5")
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
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_south5["Tuần"], totals_offset, totals)):
    fig_stack_south5.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 15rem'></div>", unsafe_allow_html=True)

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
for i, (x, y, t) in enumerate(zip(df_table_priority_south5["Tuần"], totals_offset, totals)):
    fig_stack_priority_south5.add_annotation(
        x=x,
        y=y,
        text=f"<span style='color:#e74c3c; font-weight:bold'>{t}</span>",
        showarrow=False,
        font=dict(size=20, color="#e74c3c"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        #bgcolor="rgba(255,255,0,0.77)",
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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 20rem'></div>", unsafe_allow_html=True)

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
st.markdown("<div style='height: 18rem'></div>", unsafe_allow_html=True)

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
