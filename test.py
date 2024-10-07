import pandas as pd
import matplotlib.pyplot as plt

# تعریف طول پنجره و مسیرهای فایل
lookback = 180  # فقط طول پنجره ۳۰ روز
data_paths = {
    'stock_only': "./torch_Split_DataFrame/lookback_{}",
    'stock_with_correlation': "./torch_Split_DataFrame_Correlation/lookback_{}",
    'correlation_only': "./torch_Split_DataFrame_Correlation_without_stock_data/lookback_{}"
}

# دیکشنری برای ذخیره داده‌ها
dataframes = {'stock_only': None, 'stock_with_correlation': None, 'correlation_only': None}

# خواندن داده‌های هر مجموعه برای طول پنجره ۳۰ روز
for data_type, path_template in data_paths.items():
    filename = f"{path_template}/stock_errors_lookback_{lookback}.csv"
    dataframes[data_type] = pd.read_csv(filename.format(lookback))

# دریافت ترتیب سهام از stock_only
stock_order = dataframes['stock_only']['Stock'].tolist()  # دریافت ترتیب سهام

# مرتب‌سازی داده‌های stock_with_correlation بر اساس ترتیب سهام
dataframes['stock_with_correlation'] = dataframes['stock_with_correlation'].set_index('Stock').reindex(stock_order).reset_index()

# تعریف پارامترهای نمودار
error_types = ['MSE', 'MAE', 'RMSE']
colors = {'stock_only': 'blue', 'stock_with_correlation': 'orange', 'correlation_only': 'green'}
labels = {'stock_only': 'Stock Only', 'stock_with_correlation': 'Stock with Correlation', 'correlation_only': 'Correlation Only'}

# رسم نمودارها برای هر نوع ارور
for error_type in error_types:
    plt.figure(figsize=(30, 8))

    # رسم خطوط برای هر نوع داده
    for data_type in dataframes.keys():
        if error_type in dataframes[data_type].columns:
            plt.plot(dataframes[data_type]['Stock'], 
                     dataframes[data_type][error_type], 
                     label=labels[data_type], 
                     color=colors[data_type], 
                     marker='o', 
                     linestyle='-')

    # تنظیمات نمودار
    plt.xlabel('Stock')
    plt.ylabel('Error Value')
    plt.title(f'Comparison of {error_type} for Lookback Window Size of {lookback}')
    plt.xticks(rotation=45)
    plt.legend()

    # ذخیره نمودار به عنوان فایل تصویری
    plt.tight_layout()
    plt.savefig(f"{error_type}_comparison_lookback_{lookback}.png", dpi=300)
    plt.close()