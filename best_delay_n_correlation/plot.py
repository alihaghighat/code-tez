import pandas as pd
import matplotlib.pyplot as plt
import os

# خواندن فایل اصلی و حذف هرگونه فاصله اضافی
df = pd.read_csv('test_results.csv', sep=',', skipinitialspace=True).dropna()

# حذف ردیف‌های حاوی مقدار "Stock" در ستون "Stock"
df = df[df["Stock"] != "Stock"]

# تبدیل مقادیر عددی از نوع رشته به عدد در صورت نیاز
numeric_columns = ["Lookback", "Delay", "N_Top", "MSE", "MAE", "RMSE", "MAPE"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# فیلتر کردن داده‌ها بر اساس حداقل مقدار MSE برای هر سهام
df_filtered = df.loc[df.groupby("Stock")["MSE"].idxmin()]

# مرتب‌سازی بر اساس ستون "Stock" و ریست کردن ایندکس
df_filtered = df_filtered.sort_values(by="Stock").reset_index(drop=True)

# اضافه کردن ستون MSE فردی برای هر سهام از مسیر داده‌ها
individual_mse = []

for stock in df_filtered["Stock"]:
    # مسیر فایل مربوط به هر سهام
    file_path = f'../torch_Split_DataFrame/lookback_30/stock_errors_lookback_30.csv'
    if os.path.exists(file_path):
        # خواندن فایل و دریافت مقدار MSE برای سهام
        stock_df = pd.read_csv(file_path)
        stock_mse = stock_df.loc[stock_df['Stock'] == stock, 'MSE'].values
        individual_mse.append(stock_mse[0] if len(stock_mse) > 0 else None)
    else:
        individual_mse.append(None)

# اضافه کردن MSE فردی به df_filtered
df_filtered['Individual_MSE'] = individual_mse

# ایجاد نمودار با دو سری داده
plt.figure(figsize=(40, 1))
plt.plot(df_filtered["Stock"], df_filtered["MSE"], marker='o', linestyle='-', label="Overall MSE")
plt.plot(df_filtered["Stock"], df_filtered["Individual_MSE"], marker='x', linestyle='--', color='r', label="Individual MSE")

# اضافه کردن برچسب به هر نقطه
for i, row in df_filtered.iterrows():
    label = f"Lookback: {row['Lookback']}, Delay: {row['Delay']}, N_Top: {row['N_Top']}"
    plt.text(i, row["MSE"], label, fontsize=9, ha='right', va='bottom')

# اضافه کردن برچسب به محورها و عنوان
plt.xlabel("Stock")
plt.ylabel("MSE")
plt.title("Comparison of Overall and Individual MSE for Each Stock")
plt.legend()

# ذخیره تصویر به عنوان فایل PNG
plt.savefig('line_plot_with_individual_and_overall_mse.png', format='png', dpi=300)
