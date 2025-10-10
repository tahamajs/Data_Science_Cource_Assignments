"""
پایپلاین اصلی پردازش داده
این اسکریپت مراحل کامل پردازش داده را از بارگذاری تا ذخیره‌سازی انجام می‌دهد
"""
import os
from datetime import datetime
from scripts.load_data import load_data
from scripts.preprocess import preprocess_data
from scripts.feature_engineering import engineer_features


def main():
    """اجرای پایپلاین کامل پردازش داده"""
    
    print("\n" + "="*60)
    print("🚀 شروع پایپلاین پردازش داده")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    
    try:
        # مرحله ۱: بارگذاری داده
        print("📥 مرحله ۱: بارگذاری داده از MySQL")
        print("-" * 60)
        uber_trips, weather_data, taxi_zones = load_data()
        print("✅ داده‌ها با موفقیت بارگذاری شدند.\n")

        # مرحله ۲: پیش‌پردازش
        print("🔄 مرحله ۲: پیش‌پردازش داده‌ها")
        print("-" * 60)
        uber_trips, weather_data, taxi_zones = preprocess_data(
            uber_trips, weather_data, taxi_zones
        )
        print("✅ پیش‌پردازش با موفقیت انجام شد.\n")

        # مرحله ۳: مهندسی ویژگی
        print("⚙️  مرحله ۳: مهندسی ویژگی")
        print("-" * 60)
        uber_trips, weather_data, taxi_zones = engineer_features(
            uber_trips, weather_data, taxi_zones
        )
        print("✅ مهندسی ویژگی با موفقیت انجام شد.\n")

        # مرحله ۴: ذخیره داده‌های پردازش شده
        print("💾 مرحله ۴: ذخیره‌سازی داده‌های پردازش شده")
        print("-" * 60)
        
        # ایجاد پوشه output در صورت عدم وجود
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        # ذخیره فایل‌ها
        uber_trips.to_csv(f'{output_dir}/processed_uber_trips.csv', index=False)
        print(f"   ✓ ذخیره شد: {output_dir}/processed_uber_trips.csv")
        
        weather_data.to_csv(f'{output_dir}/processed_weather_data.csv', index=False)
        print(f"   ✓ ذخیره شد: {output_dir}/processed_weather_data.csv")
        
        taxi_zones.to_csv(f'{output_dir}/processed_taxi_zones.csv', index=False)
        print(f"   ✓ ذخیره شد: {output_dir}/processed_taxi_zones.csv")
        
        print("\n✅ همه داده‌ها با موفقیت ذخیره شدند.")
        
        # نمایش خلاصه
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*60)
        print("🎉 پایپلاین با موفقیت تکمیل شد!")
        print("="*60)
        print(f"⏱️  زمان اجرا: {duration:.2f} ثانیه")
        print(f"📊 تعداد رکوردهای نهایی:")
        print(f"   - Uber: {len(uber_trips):,}")
        print(f"   - آب و هوا: {len(weather_data):,}")
        print(f"   - مناطق: {len(taxi_zones):,}")
        print(f"📁 فایل‌های خروجی در پوشه '{output_dir}' ذخیره شدند.")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ خطا در اجرای پایپلاین: {str(e)}")
        raise


if __name__ == "__main__":
    main()
