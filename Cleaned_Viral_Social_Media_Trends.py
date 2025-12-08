import pandas as pd
import matplotlib.pyplot as pd_plt
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình giao diện biểu đồ cho đẹp mắt (phong cách chuyên nghiệp)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Tải dữ liệu
df = pd.read_csv('Cleaned_Viral_Social_Media_Trends.csv')

# 2. Xử lý sơ bộ (Data Cleaning)
# Chuyển đổi cột 'Post_Date' sang định dạng datetime để phân tích theo thời gian
df['Post_Date'] = pd.to_datetime(df['Post_Date'])

# Kiểm tra thông tin tổng quan
print("--- Tổng quan dữ liệu ---")
print(df.info())
print("\n--- 5 dòng đầu tiên ---")
print(df.head())

## 1. Tỷ lệ phân bố bài viết theo Nền tảng (Platform)
# Đếm số lượng bài viết theo từng nền tảng
platform_counts = df['Platform'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Phân bố bài viết theo Nền tảng Mạng xã hội', fontsize=16)
plt.show()

## 2. So sánh mức độ tương tác (Views/Likes) theo Loại nội dung (Content Type)
# Tính trung bình Views và Likes theo Content_Type
content_stats = df.groupby('Content_Type')[['Views', 'Likes']].mean().sort_values(by='Views', ascending=False)

# Vẽ biểu đồ cột ghép
content_stats.plot(kind='bar', figsize=(12, 6), color=['#3498db', '#e74c3c'])
plt.title('Trung bình Lượt xem và Lượt thích theo Loại nội dung', fontsize=16)
plt.ylabel('Số lượng trung bình')
plt.xlabel('Loại nội dung')
plt.xticks(rotation=45)
plt.legend(['Lượt xem (Views)', 'Lượt thích (Likes)'])
plt.show()

## 3. Phân tích xu hướng Hashtag (#) phổ biến và hiệu quả
# Tính tổng tương tác cho mỗi Hashtag
# Giả sử 'Engagement_Level' là phân loại, ta dùng Views làm thước đo chính lượng
hashtag_performance = df.groupby('Hashtag')['Views'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='Views', y='Hashtag', data=hashtag_performance, palette='viridis')
plt.title('Tổng lượt xem theo từng chủ đề Hashtag', fontsize=16)
plt.xlabel('Tổng lượt xem')
plt.ylabel('Hashtag')
plt.show()

## 4. Bản đồ nhiệt tương quan (Correlation Heatmap)
# Chọn các cột số liệu để tính tương quan
numerical_cols = ['Views', 'Likes', 'Shares', 'Comments']
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan giữa các chỉ số tương tác', fontsize=16)
plt.show()

