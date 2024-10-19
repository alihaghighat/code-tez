import os
import pandas as pd
import matplotlib.pyplot as plt

# تعریف طول پنجره، دیلی‌ها و مسیرهای فایل
lookback = 30
delays = range(-1, -6, -1)  # دیلی‌های -۱ تا -۵
data_paths = {
    'stock_only': "./torch_Split_DataFrame/lookback_{}",
    'stock_only_Seasonal': "./Seasonal_one_stock/lookback_{}",
    'stock_with_correlation': "./Seasonal_one_stock_C/lookback_{}_delay_{}",
}

# دیکشنری برای ذخیره داده‌ها
dataframes = {'stock_only': None, 'stock_only_Seasonal': None}

# چک کردن فایل و خواندن داده‌های stock_only
filename_stock_only = f"{data_paths['stock_only']}/stock_errors_lookback_{lookback}.csv"
if os.path.exists(filename_stock_only.format(lookback)):
    dataframes['stock_only'] = pd.read_csv(filename_stock_only.format(lookback))
    stock_order = dataframes['stock_only']['Stock'].tolist()  # تعریف stock_order
else:
    print(f"Error: File not found: {filename_stock_only.format(lookback)}")

# چک کردن فایل و خواندن داده‌های stock_only_Seasonal
filename_stock_only_seasonal = f"{data_paths['stock_only_Seasonal']}/stock_errors_lookback_{lookback}.csv"
if os.path.exists(filename_stock_only_seasonal.format(lookback)):
    dataframes['stock_only_Seasonal'] = pd.read_csv(filename_stock_only_seasonal.format(lookback))
    # تنظیم ترتیب سهام بر اساس stock_order
    dataframes['stock_only_Seasonal'] = (
        dataframes['stock_only_Seasonal']
        .set_index('Stock')
        .reindex(stock_order)
        .reset_index()
    )
else:
    print(f"Error: File not found: {filename_stock_only_seasonal.format(lookback)}")

# دریافت ترتیب سهام از stock_only
if dataframes['stock_only'] is not None:
    stock_order = dataframes['stock_only']['Stock'].tolist()

    # خواندن و مرتب‌سازی داده‌های stock_with_correlation برای هر دیلی
    for delay in delays:
        # ساخت مسیر جدید بر اساس نام فولدر و نام فایل به‌روز شده
        filename_corr = f"./Seasonal_one_stock_C/lookback_{lookback}_delay_{delay}/lookback_{lookback}_delay_{delay}.csv"

        print(filename_corr)
        if os.path.exists(filename_corr):
            dataframes[f'stock_with_correlation_delay_{delay}'] = pd.read_csv(filename_corr)
            dataframes[f'stock_with_correlation_delay_{delay}'] = (
                dataframes[f'stock_with_correlation_delay_{delay}']
                .set_index('Stock')
                .reindex(stock_order)
                .reset_index()
            )
        else:
            print(f"Error: File not found: {filename_corr}")

# تعریف پارامترهای نمودار
error_types = ['MSE', 'MAE', 'RMSE']
colors = {
    'stock_only': 'blue',
    'stock_only_Seasonal': 'cyan',
    'stock_with_correlation_delay_-1': 'orange',
    'stock_with_correlation_delay_-2': 'green',
    'stock_with_correlation_delay_-3': 'red',
    'stock_with_correlation_delay_-4': 'purple',
    'stock_with_correlation_delay_-5': 'brown'
}
labels = {
    'stock_only': 'Stock Only',
    'stock_only_Seasonal': 'Stock Only Seasonal',
    'stock_with_correlation_delay_-1': 'Stock with Correlation (Delay -1)',
    'stock_with_correlation_delay_-2': 'Stock with Correlation (Delay -2)',
    'stock_with_correlation_delay_-3': 'Stock with Correlation (Delay -3)',
    'stock_with_correlation_delay_-4': 'Stock with Correlation (Delay -4)',
    'stock_with_correlation_delay_-5': 'Stock with Correlation (Delay -5)'
}
group_size = 20  # اندازه گروه‌ها
num_stocks = len(stock_order)
# ایجاد فولدر برای ذخیره نمودارها
output_folder = f"Error_Seasonal_Charts_Lookback_{lookback}_Delays_-1_to_-5"
os.makedirs(output_folder, exist_ok=True)
for error_type in error_types:
    for i in range(0, num_stocks, group_size):
        # تنظیمات محدوده برای گروه ۲۰ تایی
        stock_subset = stock_order[i:i + group_size]
        plt.figure(figsize=(20, 10))

        # رسم نقاط برای هر نوع داده در گروه ۲۰ تایی
        for data_type in dataframes.keys():
            if dataframes[data_type] is not None and error_type in dataframes[data_type].columns:
                data_subset = dataframes[data_type][dataframes[data_type]['Stock'].isin(stock_subset)]
                plt.plot(
                    data_subset['Stock'], 
                    data_subset[error_type], 
                    label=labels[data_type], 
                    color=colors[data_type], 
                    marker='o', 
                    linestyle='-'
                )

        # تنظیمات نمودار
        plt.xlabel('Stock')
        plt.ylabel('Error Value')
        plt.title(f'{error_type} for Stocks {i + 1} to {min(i + group_size, num_stocks)} - Lookback {lookback}, Delays -1 to -5')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # ذخیره نمودار به عنوان فایل تصویری با نام مناسب
        output_path = os.path.join(output_folder, f"{error_type}_Group_{i + 1}_to_{min(i + group_size, num_stocks)}_Lookback_{lookback}_Delays_-1_to_-5.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

print(f"Charts have been saved in the folder: {output_folder}")
