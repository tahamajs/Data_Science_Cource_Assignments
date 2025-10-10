# Notebooks Directory

## توضیحات

این پوشه شامل Jupyter Notebooks برای تحلیل، آزمایش و توسعه است.

## فایل‌ها

### `real_time.ipynb`

Notebook اصلی برای تحلیل و آزمایش real-time داده‌ها.

**محتوا:**

- اتصال به Kafka
- خواندن stream داده‌ها
- تحلیل exploratory
- تست الگوریتم‌های مختلف
- Visualization داده‌ها

**کتابخانه‌های استفاده شده:**

- pandas: برای data manipulation
- matplotlib/seaborn: برای visualization
- confluent-kafka: برای اتصال به Kafka
- numpy: برای محاسبات عددی

## نحوه استفاده

### راه‌اندازی Jupyter

```bash
# نصب jupyter (در صورت نیاز)
pip install jupyter notebook

# اجرای notebook
jupyter notebook real_time.ipynb
```

### در محیط Google Colab

می‌توانید notebook را به Colab آپلود کرده و اجرا کنید.

## بخش‌های Notebook

1. **Setup**: نصب کتابخانه‌ها و import ها
2. **Data Loading**: بارگذاری داده از Kafka/فایل
3. **EDA**: تحلیل اکتشافی داده
4. **Feature Engineering**: ساخت ویژگی‌های جدید
5. **Analysis**: تحلیل‌های مختلف
6. **Visualization**: نمودارها و گراف‌ها
7. **Results**: نتایج و یافته‌ها

## نکات

- قبل از اجرا، Kafka broker باید در حال اجرا باشد
- برای داده‌های بزرگ از sampling استفاده کنید
- نتایج را در قالب تصاویر ذخیره کنید
