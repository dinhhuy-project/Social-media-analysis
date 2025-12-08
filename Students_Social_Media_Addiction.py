import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- BƯỚC 1: TẢI VÀ LÀM SẠCH DỮ LIỆU ---
# Đọc file CSV
file_path = 'Students Social Media Addiction.csv'
df = pd.read_csv(file_path)

# Hiển thị thông tin cơ bản
print("--- 5 Dòng đầu tiên của dữ liệu ---")
print(df.head())
print("\n--- Thông tin tổng quan ---")
print(df.info())

# Kiểm tra dữ liệu bị thiếu
print("\n--- Số lượng giá trị null ---")
print(df.isnull().sum())

# Cấu hình giao diện biểu đồ
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- BƯỚC 2: PHÂN TÍCH & TRỰC QUAN HÓA ---

# 1. Phân bố các nền tảng mạng xã hội được sử dụng nhiều nhất
plt.figure(figsize=(10, 6))
sns.countplot(y='Most_Used_Platform', data=df, order=df['Most_Used_Platform'].value_counts().index, palette='viridis')
plt.title('Phân bố nền tảng mạng xã hội được sử dụng nhiều nhất', fontsize=15)
plt.xlabel('Số lượng sinh viên')
plt.ylabel('Nền tảng')
plt.show()

# 2. Mối tương quan giữa Thời gian sử dụng (Usage) và Điểm số nghiện (Addicted Score)
# Xem xét xem dùng càng nhiều thì điểm nghiện càng cao không
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Avg_Daily_Usage_Hours', y='Addicted_Score', data=df, hue='Gender', alpha=0.7)
plt.title('Tương quan giữa Thời gian sử dụng hàng ngày và Điểm độ nghiện', fontsize=15)
plt.xlabel('Thời gian sử dụng trung bình (Giờ/ngày)')
plt.ylabel('Điểm số nghiện (Addicted Score)')
plt.show()

# 3. Ảnh hưởng của MXH đến Giấc ngủ và Sức khỏe tinh thần (Heatmap)
# Chọn các cột số để tính tương quan
numeric_cols = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Addicted_Score']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan giữa các chỉ số sức khỏe và thói quen', fontsize=15)
plt.show()

# 4. So sánh ảnh hưởng đến học tập (Academic Performance)
# Nhóm sinh viên bị ảnh hưởng và không bị ảnh hưởng để xem thời gian sử dụng của họ khác nhau thế nào
plt.figure(figsize=(10, 6))
sns.boxplot(x='Affects_Academic_Performance', y='Avg_Daily_Usage_Hours', data=df, palette="Set2")
plt.title('Tác động đến học tập dựa trên thời gian sử dụng MXH', fontsize=15)
plt.xlabel('Ảnh hưởng đến kết quả học tập?')
plt.ylabel('Thời gian sử dụng trung bình (Giờ)')
plt.show()

# 5. Phân tích nền tảng nào gây ra nhiều xung đột nhất (Conflicts)
plt.figure(figsize=(12, 6))
avg_conflict = df.groupby('Most_Used_Platform')['Conflicts_Over_Social_Media'].mean().sort_values(ascending=False)
sns.barplot(x=avg_conflict.index, y=avg_conflict.values, palette='magma')
plt.title('Trung bình số vụ xung đột/tranh cãi theo từng nền tảng', fontsize=15)
plt.ylabel('Số vụ xung đột trung bình')
plt.xticks(rotation=45)
plt.show()

# --- BƯỚC 3: TỔNG KẾT SỐ LIỆU ---
print("\n--- Thống kê mô tả (Mean, Min, Max) ---")
print(df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']].describe())