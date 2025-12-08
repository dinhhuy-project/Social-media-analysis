import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Cấu hình hiển thị
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
warnings.filterwarnings('ignore')

# 1. Tải dữ liệu
# Giả sử file csv nằm cùng thư mục
file_path = 'Time-Wasters on Social Media.csv'
try:
    df = pd.read_csv(file_path)
    print(">>> Tải dữ liệu thành công!")
    print(f"Kích thước bộ dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
except FileNotFoundError:
    print(">>> Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn.")

# 2. Kiểm tra và Làm sạch sơ bộ
print("\n--- Thông tin dữ liệu ---")
print(df.info())
print("\n--- Kiểm tra giá trị thiếu ---")
print(df.isnull().sum())

# 3. Trực quan hóa dữ liệu (Visualization)

# --- Biểu đồ 1: Phân bố nền tảng mạng xã hội (Platform) ---
plt.figure(figsize=(10, 6))
platform_counts = df['Platform'].value_counts()
plt.pie(platform_counts, labels=platform_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Tỷ lệ người dùng trên các Nền tảng Mạng xã hội', fontsize=15)
plt.show()

# --- Biểu đồ 2: Tương quan giữa Độ tuổi và Tổng thời gian sử dụng (Total Time Spent) ---
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Age', y='Total Time Spent', hue='Platform', alpha=0.7)
plt.title('Tương quan giữa Độ tuổi và Thời gian sử dụng (theo Nền tảng)', fontsize=15)
plt.xlabel('Tuổi')
plt.ylabel('Tổng thời gian sử dụng (phút)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Biểu đồ 3: Mức độ nghiện (Addiction Level) theo Nghề nghiệp (Profession) ---
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x='Profession', y='Addiction Level', palette='coolwarm')
plt.title('Phân bố Mức độ Nghiện theo Nghề nghiệp', fontsize=15)
plt.xticks(rotation=45)
plt.xlabel('Nghề nghiệp')
plt.ylabel('Mức độ nghiện (thang điểm)')
plt.tight_layout()
plt.show()

# --- Biểu đồ 4: Lý do xem (Watch Reason) và Cảm giác mất năng suất (ProductivityLoss) ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Watch Reason', y='ProductivityLoss', hue='Gender', ci=None, palette='viridis')
plt.title('Mức độ Mất năng suất dựa trên Lý do xem video', fontsize=15)
plt.xlabel('Lý do xem')
plt.ylabel('Điểm mất năng suất trung bình')
plt.show()

# --- Biểu đồ 5: Ma trận tương quan (Correlation Heatmap) ---
# Chỉ lấy các cột số quan trọng để xem tương quan
numeric_cols = ['Age', 'Income', 'Total Time Spent', 'Engagement', 
                'ProductivityLoss', 'Satisfaction', 'Addiction Level', 'Scroll Rate']
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu', fmt=".2f", linewidths=0.5)
plt.title('Ma trận Tương quan giữa các biến số', fontsize=15)
plt.show()