import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Bỏ qua các cảnh báo không cần thiết để output sạch hơn
warnings.filterwarnings('ignore')

# Cấu hình giao diện biểu đồ
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. ĐỌC DỮ LIỆU
# ---------------------------------------------------------
df = pd.read_csv('Mental_Health_and_Social_Media_Balance_Dataset.csv')

# Đổi tên cột cho gọn và dễ code hơn (tùy chọn)
df.columns = ['UserID', 'Age', 'Gender', 'ScreenTime', 'Sleep', 
              'Stress', 'DaysNoSocial', 'Exercise', 'Platform', 'Happiness']

print("--- THÔNG TIN CƠ BẢN VỀ DỮ LIỆU ---")
print(df.info())
print("\n--- THỐNG KÊ MÔ TẢ ---")
print(df.describe())

# 2. TRỰC QUAN HÓA DỮ LIỆU (VISUALIZATION)
# ---------------------------------------------------------

# BIỂU ĐỒ 1: Phân bố nền tảng mạng xã hội được sử dụng (Pie Chart)
plt.figure(figsize=(8, 8))
platform_counts = df['Platform'].value_counts()
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Tỷ lệ người dùng theo Nền tảng Mạng xã hội', fontsize=16)
plt.show()

# BIỂU ĐỒ 2: Ma trận tương quan (Correlation Heatmap)
# Để xem mối liên hệ giữa Thời gian màn hình, Giấc ngủ, Stress và Hạnh phúc
plt.figure(figsize=(10, 8))
# Chỉ lấy các cột số
numeric_cols = ['Age', 'ScreenTime', 'Sleep', 'Stress', 'DaysNoSocial', 'Exercise', 'Happiness']
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan giữa các biến số', fontsize=16)
plt.show()

# BIỂU ĐỒ 3: Tác động của Thời gian màn hình đến Chất lượng giấc ngủ (Scatter Plot + Regression Line)
plt.figure(figsize=(10, 6))
sns.regplot(x='ScreenTime', y='Sleep', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Tương quan: Thời gian màn hình vs Chất lượng giấc ngủ', fontsize=14)
plt.xlabel('Thời gian màn hình hàng ngày (giờ)')
plt.ylabel('Chất lượng giấc ngủ (1-10)')
plt.show()

# BIỂU ĐỒ 4: Mức độ Stress theo từng Nền tảng mạng xã hội (Box Plot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Platform', y='Stress', data=df, palette="Set3")
plt.title('Phân bố mức độ Stress theo từng Nền tảng', fontsize=14)
plt.xlabel('Nền tảng mạng xã hội')
plt.ylabel('Mức độ Stress (1-10)')
plt.show()

# BIỂU ĐỒ 5: So sánh chỉ số Hạnh phúc theo Giới tính (Bar Plot)
plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Happiness', data=df, palette="muted", errorbar=None)
plt.title('Chỉ số Hạnh phúc trung bình theo Giới tính', fontsize=14)
plt.ylabel('Chỉ số Hạnh phúc (1-10)')
plt.show()