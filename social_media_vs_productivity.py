import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cấu hình giao diện biểu đồ cho đẹp mắt (chuẩn báo cáo)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# 1. Tải dữ liệu
# Lưu ý: Thay đổi đường dẫn nếu file của bạn nằm ở thư mục khác
df = pd.read_csv('social_media_vs_productivity.csv')

print("\n=== Thông tin tổng quan ===")
df.info()

## Phân tích Nhân khẩu học & Thói quen (Demographics & Habits)
# Tạo khung vẽ (Subplots)
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Biểu đồ 1: Phân bố độ tuổi
sns.histplot(df['age'], kde=True, color='skyblue', ax=ax[0])
ax[0].set_title('Phân bố Độ tuổi người tham gia', fontsize=14)
ax[0].set_xlabel('Tuổi')
ax[0].set_ylabel('Số lượng')

# Biểu đồ 2: Nền tảng mạng xã hội được ưa chuộng nhất
platform_counts = df['social_platform_preference'].value_counts()
colors = sns.color_palette('pastel')[0:5]
ax[1].pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax[1].set_title('Tỷ lệ ưa thích các Nền tảng Mạng xã hội', fontsize=14)

plt.tight_layout()
plt.show()

## Phân tích cốt lõi: Mạng xã hội vs Năng suất làm việc
# Biểu đồ tán xạ (Scatter Plot) với đường hồi quy
plt.figure(figsize=(10, 6))
sns.regplot(
    data=df, 
    x='daily_social_media_time', 
    y='actual_productivity_score', 
    scatter_kws={'alpha':0.5, 'color': 'teal'}, 
    line_kws={'color': 'red'}
)

plt.title('Tương quan: Thời gian dùng MXH vs Điểm năng suất thực tế', fontsize=16)
plt.xlabel('Thời gian dùng MXH mỗi ngày (phút/giờ - tùy đơn vị)', fontsize=12)
plt.ylabel('Điểm năng suất thực tế (Thang 1-10)', fontsize=12)

# Tính hệ số tương quan
corr = df['daily_social_media_time'].corr(df['actual_productivity_score'])
plt.text(x=df['daily_social_media_time'].min(), y=df['actual_productivity_score'].min(), 
         s=f'Hệ số tương quan (r): {corr:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.show()

## Phân tích Sức khỏe tinh thần: Thông báo & Mức độ Stress
plt.figure(figsize=(12, 6))

# Nhóm mức độ stress thành các nhóm để dễ quan sát (nếu stress là biến liên tục)
# Hoặc vẽ boxplot trực tiếp nếu stress là thang điểm rời rạc
sns.boxplot(x='stress_level', y='number_of_notifications', data=df, palette="Reds")

plt.title('Mối liên hệ giữa Mức độ Stress và Số lượng thông báo', fontsize=16)
plt.xlabel('Mức độ Stress (Thấp -> Cao)', fontsize=12)
plt.ylabel('Số lượng thông báo nhận được', fontsize=12)

plt.show()

## So sánh: Hiệu quả của các công cụ hỗ trợ (Focus Apps)
plt.figure(figsize=(10, 6))

# Violin plot kết hợp Box plot bên trong để thấy rõ phân phối
sns.violinplot(data=df, x='uses_focus_apps', y='actual_productivity_score', palette="Set2", split=True)

plt.title('So sánh Năng suất: Có dùng vs Không dùng Ứng dụng tập trung', fontsize=16)
plt.xlabel('Sử dụng ứng dụng tập trung (Focus Apps)', fontsize=12)
plt.ylabel('Điểm năng suất thực tế', fontsize=12)

plt.show()

## Ma trận tương quan tổng thể (Heatmap)
# Chọn các cột dữ liệu số quan trọng
cols_to_analyze = [
    'daily_social_media_time', 'number_of_notifications', 
    'perceived_productivity_score', 'actual_productivity_score', 
    'stress_level', 'sleep_hours', 'weekly_offline_hours'
]

# Tính ma trận tương quan
corr_matrix = df[cols_to_analyze].corr()

# Vẽ Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan giữa các biến số', fontsize=16)
plt.show()