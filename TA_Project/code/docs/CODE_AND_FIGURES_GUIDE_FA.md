# راهنمای کامل کدها و شکل‌های پروژه (نسخه فارسی)

این سند برای توضیح دقیق دو بخش تهیه شده است:
- توضیح فنی کدهای پروژه (ماژول‌به‌ماژول)
- توضیح کامل تمام شکل‌های تولیدشده در پروژه (تک‌به‌تک)

این فایل مکمل گزارش اصلی است و برای ارائه به استاد/داور طراحی شده است.

---

## 1) نقشه کلی کد پروژه

### 1.1 فایل‌های اصلی در `code/scripts/`
- `full_solution_pipeline.py`: ارکستریتور کامل Q1 تا Q20
- `q18_temporal.py`: بک‌تست زمانی و تحلیل افت عملکرد
- `q19_uncertainty.py`: کمی‌سازی عدم‌قطعیت با split conformal
- `q20_fairness_mitigation.py`: آزمایش مداخله عدالت قبل/بعد
- `report_metrics_export.py`: تولید متریک برای LaTeX (`json/tex`)
- `generate_report_assets.py`: ساخت نمودارهای گزارش و آمار تجمیعی
- `generate_synthetic_data.py`: تولید دیتاست مصنوعی هم‌ساختار
- `train_and_explain.py`: خط baseline آموزشی + SHAP ساده

### 1.2 جریان اجرای اصلی
1. بارگذاری داده و اعتبارسنجی ستون هدف
2. حذف/کنترل leakage
3. اجرای سؤال‌های تحلیلی Q1..Q6
4. اجرای بلوک پیشرفته Q15..Q20
5. تولید artifactهای CSV/PNG/JSON
6. تولید `run_summary.json` نسخه 2
7. export متریک‌ها برای تزریق مستقیم به گزارش IEEE

---

## 2) توضیح دقیق کدها

## 2.1 `full_solution_pipeline.py`

### نقش فایل
هسته مرکزی پروژه است و تمام سؤال‌ها را در یک اجرای یکپارچه تولید می‌کند.

### ورودی/خروجی اصلی
- ورودی: `code/data/GlobalTechTalent_50k.csv`
- خروجی:
  - `code/solutions/run_summary.json`
  - `code/solutions/complete_solution_key.md`
  - مجموعه شکل‌ها در `code/figures/`
  - خروجی Q18-Q20 در `code/solutions/`

### کلیدهای فنی مهم
- `RUN_SUMMARY_VERSION = 2`
- پروفایل اجرا: `fast`, `balanced`, `heavy`
- فعال/غیرفعال کردن Q18-Q20 با فلگ CLI

### تابع‌های مهم و کارشان
- `load_dataset`: بارگذاری امن و چک ستون هدف
- `build_features`: حذف leakage و ساخت `X/y`
- `build_preprocessor`: ایجاد `ColumnTransformer` برای عددی/دسته‌ای
- `build_capstone_model`: انتخاب XGBoost یا fallback
- `write_q1_sql`: ساخت پاسخ SQL رسمی
- `leakage_diagnostics`: محاسبه شاخص‌های نشت
- `simulate_optimizers` و `plot_ravine_paths`: Q3
- `run_q4_svm_and_pruning`: sweep برای gamma و alpha
- `run_q5_unsupervised`: PCA + KMeans elbow
- `run_q6_capstone`: مدل نهایی + SHAP + fairness slice
- `run_q15_calibration_threshold`: calibration + threshold policy
- `run_q16_drift_monitoring`: PSI/JS drift
- `run_q17_recourse_analysis`: recourse feasibility
- `run_all`: ارکستراسیون کل pipeline و ساخت summary v2

### نکته مهندسی
این فایل نقش "workflow engine" دارد و منطق تخصصی Q18-Q20 را به ماژول‌های جداگانه می‌سپارد تا تست‌پذیری و نگهداشت بهتر شود.

---

## 2.2 `q18_temporal.py`

### هدف
اعتبارسنجی زمانی غلطان (rolling) و اندازه‌گیری افت عملکرد تحت drift.

### منطق اصلی
- اگر ستون زمانی معتبر وجود داشته باشد از آن استفاده می‌کند.
- اگر نباشد، fallback استاندارد می‌زند (مثلاً `UserID` به‌عنوان proxy).
- foldهای زمانی می‌سازد و در هر fold مدل را train/test می‌کند.
- `AUC/F1/Accuracy/Precision/Recall` و `mean PSI` را ذخیره می‌کند.

### خروجی‌ها
- `code/solutions/q18_temporal_backtest.csv`
- `code/figures/q18_temporal_degradation.png`

### چرا مهم است
Random split ممکن است خوش‌بینانه باشد؛ این ماژول نشان می‌دهد عملکرد در آینده چقدر پایدار می‌ماند.

---

## 2.3 `q19_uncertainty.py`

### هدف
کمی‌سازی عدم‌قطعیت پیش‌بینی با روش split-conformal.

### منطق اصلی
- داده را به train/calibration/test تقسیم می‌کند.
- خطای calibration را اندازه می‌گیرد: `|y - p|`
- برای هر سطح اطمینان (alpha)، `q_hat` را به روش quantile-higher می‌گیرد.
- بازه احتمال `[p-q_hat, p+q_hat]` می‌سازد.
- پوشش تجربی، under-coverage و عرض بازه را گزارش می‌کند.

### خروجی‌ها
- `code/solutions/q19_uncertainty_coverage.csv`
- `code/figures/q19_coverage_vs_alpha.png`

### چرا مهم است
مدل فقط «پیش‌بینی» نمی‌دهد، بلکه «اعتماد» هم می‌دهد تا موارد کم‌اطمینان به بررسی انسانی ارجاع شوند.

---

## 2.4 `q20_fairness_mitigation.py`

### هدف
مقایسه پیش/پس از مداخله عدالت و کنترل افت عملکرد.

### منطق اصلی
- گروه حساس: `Country_Origin`
- سنجه‌های عدالت: `demographic_parity_gap` و `equal_opportunity_gap`
- وزن‌دهی مجدد (`reweighing_with_group_label_tilt`) روی train
- آموزش baseline و mitigated
- policy gate:
  - افت AUC <= 0.03
  - افت F1 <= 0.05
  - بهبود حداقل یکی از gapها

### خروجی‌ها
- `code/solutions/q20_fairness_mitigation_comparison.csv`
- `code/solutions/q20_fairness_groups_baseline.csv`
- `code/solutions/q20_fairness_groups_mitigated.csv`
- `code/figures/q20_fairness_tradeoff.png`

### چرا مهم است
بهبود عدالت باید همراه با کنترل utility باشد؛ این ماژول دقیقاً همین tradeoff را رسمی می‌کند.

---

## 2.5 `report_metrics_export.py`

### هدف
حذف hard-code متریک از گزارش‌های LaTeX.

### کارکرد
- متریک‌های کلیدی را از `run_summary.json` استخراج می‌کند.
- در دو قالب خروجی می‌دهد:
  - `latex_metrics.json` برای ماشین/اسکریپت
  - `latex_metrics.tex` برای `\newcommand` در LaTeX

### چرا مهم است
گزارش PDF همیشه با آخرین اجرای واقعی sync می‌ماند.

---

## 2.6 `generate_report_assets.py`

### هدف
ساخت شکل‌های گزارش مدیریتی از داده خام + خلاصه وضعیت اجرا.

### شکل‌های تولیدی
- `report_target_balance.png`
- `report_missingness_top10.png`
- `report_numeric_correlation.png`
- `report_country_migration_rate.png`

### خروجی آماری
- `code/solutions/report_stats.json`

---

## 2.7 `generate_synthetic_data.py`

### هدف
تولید داده مصنوعی بازتولیدپذیر با روابط آماری واقع‌نما.

### نکات مدل‌سازی داده
- ترکیب اثر خطی/غیرخطی روی propensity مهاجرت
- ساخت leakage عمدی (`Visa_Approval_Date`) برای آموزش مفهوم leakage
- تولید ویژگی‌های عددی/دسته‌ای نزدیک به سناریوی واقعی

---

## 2.8 `train_and_explain.py`

### هدف
اسکریپت آموزشی baseline برای train سریع و SHAP ساده.

### تفاوت با pipeline اصلی
- سبک‌تر و آموزشی‌تر است.
- از RandomForest + SHAP ساده استفاده می‌کند.
- خروجی شکل legacy: `code/figures/shap_force_plot.png`

---

## 3) توضیح کامل شکل‌ها (تک‌به‌تک)

در هر شکل 6 مورد توضیح داده شده:
1. هدف شکل
2. از کدام کد ساخته شده
3. محور/متغیرها
4. چگونه خوانده شود
5. اثر تصمیمی
6. محدودیت

---

## 3.1 `code/figures/report_target_balance.png`
- هدف: سنجش توازن کلاس هدف.
- سازنده: `generate_report_assets.py` -> `plot_target_balance`.
- محور افقی: کلاس‌های `Migration_Status`.
- محور عمودی: تعداد رکورد.
- خوانش: اگر اختلاف کلاس زیاد باشد، Accuracy می‌تواند گمراه‌کننده شود.
- اثر تصمیمی: تمرکز بر `AUC/F1` و threshold policy.
- محدودیت: توازن فعلی تضمین‌کننده توازن آینده نیست.

## 3.2 `code/figures/report_missingness_top10.png`
- هدف: نشان‌دادن ستون‌های با missing بالا.
- سازنده: `plot_missingness`.
- محور افقی: درصد missing.
- محور عمودی: نام ستون.
- خوانش: ستون‌های با missing بالا باید از منظر کیفیت/نشت بررسی شوند.
- اثر تصمیمی: طراحی strategy برای imputation یا حذف.
- محدودیت: missingness می‌تواند MNAR باشد و خودِ آن signal داشته باشد.

## 3.3 `code/figures/report_numeric_correlation.png`
- هدف: مرور اولیه قدرت رابطه ویژگی‌های عددی با هدف.
- سازنده: `plot_correlation`.
- محور: ماتریس هم‌بستگی.
- خوانش: مقادیر نزدیک 1/-1 هم‌بستگی قوی‌تر.
- اثر تصمیمی: اولویت‌بندی ویژگی‌ها برای مدل‌سازی و کنترل کیفیت.
- محدودیت: هم‌بستگی علیت نیست.

## 3.4 `code/figures/report_country_migration_rate.png`
- هدف: مقایسه نرخ مهاجرت بین کشورها.
- سازنده: `plot_country_migration_rate`.
- محور افقی: migration rate.
- محور عمودی: کشور.
- خوانش: تفاوت گروهی می‌تواند نشانگر signal یا bias policy باشد.
- اثر تصمیمی: فعال‌سازی fairness audit.
- محدودیت: ممکن است اثر اندازه نمونه/سیاست بیرونی باشد.

## 3.5 `code/figures/q3_ravine_optimizers.png`
- هدف: مقایسه مسیر همگرایی SGD/Momentum/Adam.
- سازنده: `plot_ravine_paths`.
- محور x/y: پارامترهای `theta_1` و `theta_2`.
- خوانش: مسیرهای همگرایی و loss نهایی هر بهینه‌ساز.
- اثر تصمیمی: انتخاب optimizer شتاب‌دار در توابع بدشرط.
- محدودیت: تابع toy است و همه پیچیدگی‌های دنیای واقعی را ندارد.

## 3.6 `code/figures/q4_svm_gamma_sweep.png`
- هدف: اثر `gamma` روی train/validation accuracy.
- سازنده: `run_q4_svm_and_pruning`.
- محور افقی: `gamma` (log scale).
- محور عمودی: accuracy.
- خوانش: فاصله زیاد train-val نشانه overfit است.
- اثر تصمیمی: انتخاب gamma بهینه بر مبنای validation.
- محدودیت: نتیجه وابسته به split و scaling است.

## 3.7 `code/figures/q4_tree_pruning_curve.png`
- هدف: رابطه `ccp_alpha` با عملکرد.
- سازنده: `run_q4_svm_and_pruning`.
- محور افقی: `ccp_alpha`.
- محور عمودی: accuracy.
- خوانش: نقطه بهینه بین پیچیدگی و تعمیم.
- اثر تصمیمی: تنظیم سطح pruning برای کنترل variance.
- محدودیت: نوسان‌پذیری بین splitها.

## 3.8 `code/figures/q5_kmeans_elbow.png`
- هدف: انتخاب K با Elbow.
- سازنده: `run_q5_unsupervised`.
- محور افقی: تعداد خوشه K.
- محور عمودی: WCSS.
- خوانش: محل تغییر شیب شدیدتر به نقطه elbow نزدیک‌تر است.
- اثر تصمیمی: انتخاب K با توجیه هندسی.
- محدودیت: در منحنی‌های نرم elbow مبهم می‌شود.

## 3.9 `code/figures/q6_shap_force_plot.png`
- هدف: تبیین محلی پیش‌بینی یک کاندید.
- سازنده: `run_q6_capstone`.
- محور/ساختار: سهم ویژگی‌ها در خروجی نهایی مدل.
- خوانش: ویژگی‌های مثبت پیش‌بینی را بالا می‌برند، منفی پایین می‌آورند.
- اثر تصمیمی: بازبینی پرونده بر اساس driverهای اصلی.
- محدودیت: توضیح مدل است، نه رابطه علّی.

## 3.10 `code/figures/q6_shap_summary.png`
- هدف: اهمیت سراسری ویژگی‌های encode شده.
- سازنده: `run_q6_capstone`.
- محور افقی: اهمیت نسبی.
- محور عمودی: ویژگی‌ها.
- خوانش: ویژگی‌های بالاتر، اثر جهانی بیشتری دارند.
- اثر تصمیمی: اولویت کیفیت داده و governance روی top features.
- محدودیت: ناهمگنی زیرگروه‌ها را کامل نشان نمی‌دهد.

## 3.11 `code/figures/q15_calibration_curve.png`
- هدف: تطابق احتمال پیش‌بینی و فراوانی واقعی.
- سازنده: `run_q15_calibration_threshold`.
- محور x: mean predicted probability.
- محور y: observed positive rate.
- خوانش: نزدیکی به قطر=کالیبراسیون بهتر.
- اثر تصمیمی: اعتمادپذیری احتمال در تصمیم‌های risk-aware.
- محدودیت: با drift زمانی ممکن است خراب شود.

## 3.12 `code/figures/q15_threshold_tradeoff.png`
- هدف: tradeoff بین precision/recall/F1 و expected cost.
- سازنده: `run_q15_calibration_threshold`.
- محور x: threshold.
- محور y چپ: معیارهای طبقه‌بندی.
- محور y راست: هزینه مورد انتظار.
- خوانش: آستانه بهینه از دید هدف سازمانی مشخص می‌شود.
- اثر تصمیمی: حذف threshold ثابت 0.5.
- محدودیت: هزینه‌ها context-dependent هستند.

## 3.13 `code/figures/q16_drift_psi_top12.png`
- هدف: رتبه‌بندی drift ویژگی‌ها با PSI.
- سازنده: `run_q16_drift_monitoring`.
- محور افقی: PSI.
- محور عمودی: ویژگی.
- خوانش: بالاتر بودن PSI یعنی ناپایداری بیشتر.
- اثر تصمیمی: تعریف alert و trigger بازآموزی.
- محدودیت: drift رابطه ویژگی-هدف را مستقیم نمی‌سنجد.

## 3.14 `code/figures/q17_recourse_median_deltas.png`
- هدف: اندازه مداخله لازم برای recourse بر حسب ویژگی.
- سازنده: `run_q17_recourse_analysis`.
- محور افقی: ویژگی actionable.
- محور عمودی: median required delta.
- خوانش: delta کمتر => اقدام‌پذیری بهتر.
- اثر تصمیمی: پیشنهاد اقدامات واقع‌بینانه به متقاضی.
- محدودیت: همه مداخلات در دنیای واقعی یکسان‌هزینه نیستند.

## 3.15 `code/figures/q18_temporal_degradation.png`
- هدف: مقایسه تغییر AUC/F1 با drift proxy بین foldهای زمانی.
- سازنده: `run_q18_temporal_backtesting`.
- محور x: شماره fold زمانی.
- محور y چپ: AUC/F1.
- محور y راست: mean PSI.
- خوانش: افت متریک همراه افزایش drift می‌تواند هشدار stability باشد.
- اثر تصمیمی: الزام validation زمانی قبل از deploy.
- محدودیت: در این اجرا fallback زمانی (`UserID`) استفاده شده است.

## 3.16 `code/figures/q19_coverage_vs_alpha.png`
- هدف: ارزیابی کیفیت uncertainty در سطوح confidence مختلف.
- سازنده: `run_q19_uncertainty_quantification`.
- محور x: nominal coverage.
- محور y چپ: empirical coverage.
- محور y راست: average interval width.
- خوانش: پوشش نزدیک خط ideal بهتر است؛ عرض بازه هزینه اطمینان را نشان می‌دهد.
- اثر تصمیمی: تعریف مسیر human review برای low-confidence.
- محدودیت: تضمین conformal تحت shift شدید ممکن است تضعیف شود.

## 3.17 `code/figures/q20_fairness_tradeoff.png`
- هدف: نمایش جابه‌جایی مدل از baseline به mitigated روی صفحه عدالت-کارایی.
- سازنده: `run_q20_fairness_mitigation`.
- محور x: demographic parity gap (کمتر بهتر).
- محور y: ROC-AUC (بیشتر بهتر).
- خوانش: نقطه mitigated باید به سمت gap کمتر و کارایی قابل‌قبول حرکت کند.
- اثر تصمیمی: پذیرش مداخله فقط در صورت عبور از policy gate.
- محدودیت: یک معیار عدالت کافی نیست و تحلیل intersectional هم لازم است.

## 3.18 `code/figures/shap_force_plot.png` (Legacy Baseline)
- هدف: SHAP ساده در اسکریپت baseline آموزشی.
- سازنده: `train_and_explain.py`.
- خوانش: مشابه Q6 اما روی pipeline ساده‌تر.
- اثر تصمیمی: مناسب demo آموزشی، نه مرجع نهایی داوری.
- محدودیت: با خروجی Q6 (capstone اصلی) جایگزین نمی‌شود.

---

## 4) ارتباط شکل‌ها با تصمیم مهندسی

- شکل‌های `report_*`: تشخیص اولیه کیفیت داده و ریسک policy
- شکل‌های Q3-Q6: تصمیم‌های مدل‌سازی اصلی
- شکل‌های Q15-Q17: قابلیت‌اعتماد احتمال، drift، و اقدام‌پذیری
- شکل‌های Q18-Q20: پایداری زمانی، عدم‌قطعیت، و عدالت

پس اگر خروجی مدل خوب باشد اما Q18/Q19/Q20 ضعیف باشند، استقرار واقعی هنوز قابل‌دفاع نیست.

---

## 5) چطور این توضیحات را در ارائه شفاهی استفاده کنیم

برای هر شکل این ترتیب را بگویید:
1. این شکل چه چیزی را اندازه می‌گیرد؟
2. عدد/الگوی مهم شکل چیست؟
3. این الگو چه تصمیمی را تغییر می‌دهد؟
4. محدودیت شکل چیست و چطور پوشش داده می‌شود؟

این الگو باعث می‌شود ارائه شما علمی، حرفه‌ای و تصمیم‌محور باشد.

---

## 6) مسیرهای مرجع

- کدها: `code/scripts/`
- شکل‌ها: `code/figures/`
- خروجی summary: `code/solutions/run_summary.json`
- گزارش جامع فارسی: `code/docs/PROJECT_REPORT_FA.md`
- پاسخ‌نامه کامل فارسی: `code/solutions/SOLUTION_KEY_FA.md`

