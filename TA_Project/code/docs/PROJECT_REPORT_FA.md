# گزارش فنی جامع پروژه (نسخه فارسی)

## عنوان پروژه
تحلیل مهاجرت جهانی استعدادهای فنی با رویکرد داده‌محور  
دانشگاه تهران - دانشکده مهندسی برق و کامپیوتر  
نسخه نهایی کپستون کارشناسی‌ارشد (Q1 تا Q20)

---

## 1) خلاصه مدیریتی (Executive Summary)
این پروژه یک پیاده‌سازی دانشگاهی-صنعتی کامل از چرخه علم داده است که از مرحله مهندسی داده و کنترل نشت شروع می‌شود و تا مدل‌سازی، ارزیابی پیشرفته، تبیین‌پذیری، تحلیل پایداری زمانی، کمی‌سازی عدم‌قطعیت، و مداخله عدالت الگوریتمی ادامه پیدا می‌کند.

خروجی پروژه فقط «یک مدل» نیست؛ بلکه یک بسته کامل قابل داوری است که شامل موارد زیر است:
- کد اجرایی ماژولار برای Q1 تا Q20
- نوت‌بوک‌های آموزشی/حل‌تمرین
- پاسخ‌نامه‌های فارسی و انگلیسی
- خروجی‌های استاندارد `CSV/JSON/PNG`
- گزارش‌های IEEE (فارسی و انگلیسی)
- تست‌های واحد و تست شِما

این طراحی باعث می‌شود پروژه برای سه سناریو قابل استفاده باشد:
- ارزیابی آموزشی در درس علم داده
- تمرین استاندارد مهندسی ML در محیط نیمه‌واقعی
- نمونه مرجع برای مستندسازی تصمیم‌های فنی و اخلاقی

---

## 2) دامنه مسئله و تعریف هدف

### 2.1 مسئله اصلی
هدف، پیش‌بینی `Migration_Status` برای متخصصان فنی است:
- `1`: مهاجرت کاری انجام شده/در وضعیت مثبت مهاجرت
- `0`: مهاجرت کاری انجام نشده

### 2.2 داده ورودی
- مسیر داده: `code/data/GlobalTechTalent_50k.csv`
- حجم: 50,000 رکورد
- نوع داده: ترکیب عددی + دسته‌ای + فیلدهای فرایندی

### 2.3 معیار موفقیت پروژه
موفقیت پروژه تنها با Accuracy سنجیده نمی‌شود. معیارها چندلایه هستند:
- دقت مدل (AUC/F1/Accuracy)
- قابلیت اطمینان احتمال (Calibration)
- پایداری در زمان (Temporal Backtesting)
- کیفیت عدم‌قطعیت (Coverage)
- عدالت بین گروهی (Fairness Gaps)
- بازتولیدپذیری (تست + گزارش + artifact)

---

## 3) معماری کلان پروژه

### 3.1 ساختار اصلی پوشه‌ها
- `code/scripts/`: منطق اصلی پردازش و مدل‌سازی
- `code/tests/`: تست‌های واحد، شِما، artifact
- `code/notebooks/`: نوت‌بوک آموزشی و حل کامل
- `code/solutions/`: خروجی‌های تحلیلی، summary، answer key
- `code/figures/`: نمودارهای خروجی برای تحلیل و گزارش
- `code/latex/`: فایل‌های Assignment/Solution/Report در قالب IEEE
- `code/docs/`: مستندات فنی و آموزشی

### 3.2 نقش فایل‌های کلیدی
- `code/scripts/full_solution_pipeline.py`
  - ارکستریتور اصلی Q1 تا Q20
  - مدیریت profile اجرایی
  - ساخت `run_summary.json` نسخه 2
  - تولید پاسخ‌نامه نهایی و export متریک برای LaTeX

- `code/scripts/q18_temporal.py`
  - بک‌تست زمانی با پنجره غلطان
  - سنجش افت عملکرد و drift-aware diagnostics

- `code/scripts/q19_uncertainty.py`
  - کمی‌سازی عدم‌قطعیت با split conformal interval
  - تحلیل coverage در سطوح اطمینان مختلف

- `code/scripts/q20_fairness_mitigation.py`
  - آزمایش مداخله عدالت قبل/بعد
  - اعمال قید سیاستی روی افت کارایی

- `code/scripts/report_metrics_export.py`
  - تولید `latex_metrics.json` و `latex_metrics.tex`
  - حذف hard-code متریک در گزارش‌های LaTeX

---

## 4) جریان اجرای End-to-End

### 4.1 ترتیب منطقی اجرا
1. خواندن داده و اعتبارسنجی ستون هدف
2. اجرای Q1 (SQL + leakage diagnostics)
3. اجرای Q3 تا Q6 (تحلیل بهینه‌سازی، غیرخطی، بدون‌نظارت، SHAP)
4. اجرای Q15 تا Q17 (کالیبراسیون، drift، recourse)
5. اجرای Q18 تا Q20 (زمان، عدم‌قطعیت، عدالت)
6. تجمیع نتایج در `run_summary.json` (schema v2)
7. ساخت خروجی‌های متریک برای تزریق خودکار در گزارش
8. تولید پاسخ‌نامه نهایی و figureهای گزارش

### 4.2 پروفایل‌های اجرایی
پایپ‌لاین از سه پروفایل پشتیبانی می‌کند:
- `fast`: سریع برای sanity-check و CI سبک
- `balanced`: پیش‌فرض رسمی پروژه (تعادل دقت/زمان)
- `heavy`: تحلیل سنگین‌تر برای پژوهش عمیق

پروفایل پیش‌فرض پروژه: `balanced`

### 4.3 فرمان‌های کلیدی
- اجرای کامل: `make run`
- تست: `make test`
- کامپایل: `make compile`
- گزارش انگلیسی IEEE: `make report`
- گزارش فارسی IEEE: `make report-fa`

---

## 5) تشریح دقیق سؤال‌ها و پیاده‌سازی

## Q1) مهندسی داده، SQL و کنترل نشت

### هدف علمی
دانشجو باید نشان دهد قبل از مدل‌سازی، کیفیت داده و صحت زمانی ویژگی‌ها را می‌فهمد.

### پیاده‌سازی
- کوئری window function در:
  - `code/solutions/q1_moving_average.sql`
- moving average سه‌ساله + رتبه‌بندی کشوری

### تشخیص نشت
فیلد `Visa_Approval_Date` به‌عنوان نشت مستقیم تشخیص داده می‌شود، چون پس‌رخداد (post-outcome) است.

### خروجی و تفسیر
در اجرای فعلی، شاخص‌های leakage نشان می‌دهند مدل با این فیلد می‌تواند پاسخ را تقریباً مستقیم حدس بزند؛ بنابراین این فیلد باید حذف شود.

---

## Q2) استنباط آماری و Elastic Net

### هدف علمی
ارزیابی فهم ریاضی دانشجو در مشتق تابع هزینه regularized.

### نکته فنی
در نقطه صفر برای جمله L1 مشتق کلاسیک نداریم و باید از زیرگرادیان استفاده شود:
- برای `theta_j > 0` مقدار `+1`
- برای `theta_j < 0` مقدار `-1`
- برای `theta_j = 0` بازه `[-1, +1]`

### خروجی آموزشی
پاسخ تشریحی کامل در کلید پاسخ فارسی/انگلیسی آمده است:
- `code/solutions/SOLUTION_KEY_FA.md`
- `code/solutions/complete_solution_key.md`

---

## Q3) تحلیل بهینه‌سازی در هندسه Ravine

### هدف علمی
مقایسه رفتار SGD با Momentum و Adam در تابع بدشرط.

### روش
- شبیه‌سازی مسیر همگرایی برای سه بهینه‌ساز
- رسم trajectory روی کانتور تابع

### Artifact
- `code/figures/q3_ravine_optimizers.png`

### تفسیر
Momentum و Adam نوسان عرضی دره را کم می‌کنند و در امتداد شیب اصلی سریع‌تر به مینیمم نزدیک می‌شوند.

---

## Q4) مدل‌های غیرخطی و کنترل پیچیدگی

### Q4-A: SVM-RBF
- sweep روی `gamma`
- تحلیل bias-variance در تنظیم kernel width

Artifact:
- `code/figures/q4_svm_gamma_sweep.png`

### Q4-B: Decision Tree Pruning
- استفاده از cost-complexity pruning
- تحلیل اثر `ccp_alpha` روی overfit/underfit

Artifact:
- `code/figures/q4_tree_pruning_curve.png`

---

## Q5) PCA و KMeans

### Q5-A: PCA
- تفسیر eigenvalue به‌عنوان واریانس توضیح داده‌شده
- محاسبه نسبت واریانس اجزای اصلی

### Q5-B: KMeans Elbow
- تحلیل رفتار نزولی WCSS با K
- انتخاب K به‌صورت heuristic با بیشترین خمیدگی قابل‌دفاع

Artifact:
- `code/figures/q5_kmeans_elbow.png`

---

## Q6) کپستون مدل + SHAP

### هدف علمی
تبیین تصمیم مدل black-box در سطح محلی و سراسری.

### روش
- اگر `xgboost` نصب باشد: XGBoost
- در غیر این صورت: fallback به RandomForest
- تولید SHAP force plot و summary plot

### خروجی اجرای اخیر (balanced)
- مدل: `XGBoost`
- Accuracy: `0.5835`
- ROC-AUC: `0.5495`
- F1: `0.2475`
- SHAP top feature: `num__Research_Citations`

### Artifacts
- `code/figures/q6_shap_force_plot.png`
- `code/figures/q6_shap_summary.png`
- `code/solutions/q6_fairness_country_rates.csv`

### تفسیر base/output value
- `base_value`: خروجی مورد انتظار مدل روی داده مرجع
- `output_value`: خروجی نهایی برای نمونه مشخص
- جمع SHAPها اختلاف این دو را بازسازی می‌کند

---

## Q15) کالیبراسیون و سیاست آستانه

### هدف
تبدیل امتیاز مدل به احتمال قابل اعتماد برای تصمیم‌گیری.

### روش
- محاسبه Brier Score و ECE
- sweep آستانه برای F1 و هزینه مورد انتظار

### خروجی اجرای اخیر
- ROC-AUC: `0.5414`
- Brier: `0.2436`
- ECE: `0.0327`
- best F1 threshold: `0.25`
- best F1: `0.5853`

### Artifacts
- `code/figures/q15_calibration_curve.png`
- `code/figures/q15_threshold_tradeoff.png`

---

## Q16) پایش Drift

### هدف
بررسی اینکه توزیع داده جدید نسبت به داده مرجع تغییر کرده یا نه.

### روش
- PSI برای ویژگی‌های عددی
- Jensen-Shannon divergence برای توزیع کشوری

### خروجی اجرای اخیر
- split rule: `random half split`
- top drift feature: `Visa_Approval_Date`
- top PSI: `0.00134`
- JS(country): `0.00018`

### Artifacts
- `code/solutions/q16_drift_psi.csv`
- `code/figures/q16_drift_psi_top12.png`

### تفسیر
در این اجرا drift شدید دیده نشده، اما چارچوب مانیتورینگ کامل پیاده‌سازی شده و برای داده‌های واقعی قابل تعمیم است.

---

## Q17) تحلیل Recourse

### هدف
بررسی «قابلیت اقدام» برای نمونه‌های ردشده و برآورد حداقل تغییر لازم.

### روش
- جست‌وجوی تغییرات کوچک روی ویژگی‌های قابل اقدام
- اندازه‌گیری نرخ موفقیت recourse

### خروجی اجرای اخیر
- candidates considered: `120`
- successful recourse: `120`
- recourse success rate: `1.0`
- median delta GitHub: `2.0`
- median delta citations: `50.0`
- median delta experience: `0.5`

### Artifacts
- `code/solutions/q17_recourse_examples.csv`
- `code/figures/q17_recourse_median_deltas.png`

---

## Q18) اعتبارسنجی زمانی و افت عملکرد

### هدف
بررسی پایداری مدل در گذر زمان (نه فقط random split).

### روش
- rolling backtesting
- مقایسه foldهای متوالی
- محاسبه AUC/F1 decay
- گزارش fallback اگر ستون زمانی معتبر وجود نداشته باشد

### خروجی اجرای اخیر
- temporal column used: `UserID`
- split strategy: `fallback_userid_proxy`
- fold count: `4`
- mean AUC: `0.5428`
- mean F1: `0.3744`
- AUC decay absolute: `0.0478`
- fallback used: `True`

### Artifacts
- `code/solutions/q18_temporal_backtest.csv`
- `code/figures/q18_temporal_degradation.png`

### نکته داوری
در گزارش نهایی explicitly ذکر می‌شود که fallback زمانی استفاده شده و تهدید اعتبار زمانی وجود دارد.

---

## Q19) کمی‌سازی عدم‌قطعیت

### هدف
اندازه‌گیری اینکه پیش‌بینی مدل با چه اطمینانی قابل استفاده است.

### روش
- split conformal interval روی احتمال
- گزارش coverage empirical در confidenceهای مختلف

### خروجی اجرای اخیر
- method: `split_conformal_probability_interval`
- train/cal/test: `12000/4000/4000`
- coverage@80: `0.795`
- coverage@90: `0.900`
- coverage@95: `0.9515`
- coverage@98: `0.98075`
- max under-coverage gap: `0.005`

### Artifacts
- `code/solutions/q19_uncertainty_coverage.csv`
- `code/figures/q19_coverage_vs_alpha.png`

### تفسیر
پوشش‌ها نزدیک به سطح هدف هستند و نشان می‌دهند چارچوب عدم‌قطعیت به‌صورت عملی قابل دفاع است.

---

## Q20) مداخله عدالت الگوریتمی

### هدف
مقایسه قبل/بعد mitigation تحت قید سیاستی «عدم افت بیش‌ازحد عملکرد».

### روش
- reweighing با tilt بر مبنای گروه حساس (`Country_Origin`)
- اندازه‌گیری هم‌زمان performance و fairness gaps
- policy gate برای قبولی/رد مداخله

### خروجی اجرای اخیر
- baseline AUC: `0.5495`
- mitigated AUC: `0.5563`
- baseline F1: `0.2475`
- mitigated F1: `0.2621`
- DP gap: `0.1550 -> 0.0978`
- Equal Opportunity gap: `0.1893 -> 0.1666`
- policy pass: `True`

### Artifacts
- `code/solutions/q20_fairness_mitigation_comparison.csv`
- `code/solutions/q20_fairness_groups_baseline.csv`
- `code/solutions/q20_fairness_groups_mitigated.csv`
- `code/figures/q20_fairness_tradeoff.png`

### تفسیر تصمیمی
در این اجرا، هم عدالت بهتر شده و هم معیارهای عملکرد افت نکرده‌اند؛ بنابراین مداخله از نظر policy قابل‌قبول است.

---

## 6) توضیح خروجی‌های کلیدی پروژه

### 6.1 خروجی تجمیعی
- `code/solutions/run_summary.json`
  - نسخه شِما: `2`
  - شامل آبجکت‌های `q18`, `q19`, `q20`
  - شامل metadata اجرای پروفایل و استراتژی split

### 6.2 خروجی متریک برای LaTeX
- `code/solutions/latex_metrics.json`
- `code/solutions/latex_metrics.tex`

این دو فایل باعث می‌شوند گزارش‌های IEEE با اعداد واقعی run تولید شوند و hard-code متریک حذف شود.

### 6.3 پاسخ‌نامه‌ها
- `code/solutions/complete_solution_key.md`
- `code/solutions/extended_solution_key.md`
- `code/solutions/SOLUTION_KEY_FA.md`

### 6.4 شکل‌ها و نمودارها
تمام figureهای استفاده‌شده در گزارش نهایی در `code/figures/` ساخته می‌شوند و از مسیر ثابت و بازتولیدپذیر بارگذاری می‌شوند.

---

## 7) شِمای `run_summary.json` (نسخه 2)

فیلدهای مهم:
- `run_summary_version`
- `runtime_profile`
- `data_split_strategy`
- `metric_export_version`
- `q1` ... `q20`
- `artifacts`

بلوک‌های جدید نسخه 2:
- `q18`: شاخص‌های temporal backtesting
- `q19`: شاخص‌های uncertainty coverage
- `q20`: شاخص‌های fairness mitigation + policy pass

---

## 8) کیفیت کد و تست‌ها

### 8.1 تست‌های موجود
- `code/tests/test_extended_questions.py`
- `code/tests/test_q18_q20.py`
- `code/tests/test_run_summary_schema_v2.py`

### 8.2 اهداف تست
- صحت ساخت خروجی‌های Q18/Q19/Q20
- سازگاری شِمای summary v2
- تولید artifactهای جدید CSV/PNG
- جلوگیری از regression در رابط CLI

---

## 9) نوت‌بوک‌ها و کاربرد آموزشی

### 9.1 نوت‌بوک حل کامل
- `code/notebooks/Solution_Notebook.ipynb`
- پوشش اجرایی Q1 تا Q20
- مناسب برای TA و ارائه Demo رسمی

### 9.2 نوت‌بوک کاربرگ توسعه‌یافته
- `code/notebooks/Extended_Assignment_Workbook.ipynb`
- ساختار آموزشی Q1 تا Q20 + Capstone
- مناسب برای دانشجو جهت توسعه پاسخ‌های مرحله‌ای

---

## 10) گزارش‌های IEEE

### 10.1 گزارش انگلیسی
- فایل منبع: `code/latex/project_report_full.tex`
- خروجی: `code/latex/project_report_full.pdf`

### 10.2 گزارش فارسی
- فایل منبع: `code/latex/project_report_full_fa.tex`
- خروجی: `code/latex/project_report_full_fa.pdf`

### 10.3 ویژگی محتوایی گزارش
برای هر شکل، سه لایه توضیح وجود دارد:
- Interpretation
- Decision Impact
- Limitation/Threat

این طراحی کیفیت داوری را بالا می‌برد چون فقط تصویر ارائه نمی‌شود، بلکه پیام تصمیمی و محدودیت نیز کنار آن ثبت می‌شود.

---

## 11) محدودیت‌ها و تهدیدهای اعتبار

- داده همه علل واقعی مهاجرت (ژئوپولیتیک/خانواده/حقوقی) را پوشش نمی‌دهد.
- برخی تحلیل‌ها به فرض ایستایی وابسته‌اند.
- در نبود ستون زمانی معتبر، Q18 از fallback استفاده می‌کند.
- SHAP تبیین رفتار مدل است، نه علیت دنیای واقعی.
- fairness روی گروه تعریف‌شده ارزیابی شده و نیاز به بررسی intersectional دارد.

---

## 12) نتیجه‌گیری فنی
این پروژه از نظر آموزشی و مهندسی به سطح «کپستون حرفه‌ای» رسیده است:
- از Q1 تا Q20 پیاده‌سازی اجرایی و artifactمحور دارد
- گزارش‌ها در قالب IEEE قابل داوری هستند
- خروجی‌ها بازتولیدپذیر و تست‌پذیر هستند
- تحلیل‌ها فقط دقت‌محور نیستند و fairness/uncertainty/temporal robustness را هم پوشش می‌دهند

به‌صورت عملی، این بسته می‌تواند هم به‌عنوان تحویل نهایی درس و هم به‌عنوان الگوی ساخت پروژه‌های ML مسئولانه در محیط واقعی استفاده شود.

---

## 13) راهنمای اجرای سریع برای داور/استاد

1. فعال‌سازی محیط:
- `source /Users/tahamajs/Documents/uni/venv/bin/activate`

2. اجرای کامل:
- `make run`

3. اجرای تست:
- `make test`

4. تولید گزارش‌ها:
- `make report`
- `make report-fa`

5. بررسی خروجی‌ها:
- `code/solutions/run_summary.json`
- `code/figures/`
- `code/latex/project_report_full.pdf`
- `code/latex/project_report_full_fa.pdf`

---

## 14) فهرست artifactهای کلیدی (برای ارزیابی نهایی)

- `code/solutions/q1_moving_average.sql`
- `code/solutions/complete_solution_key.md`
- `code/solutions/SOLUTION_KEY_FA.md`
- `code/solutions/run_summary.json`
- `code/solutions/latex_metrics.json`
- `code/solutions/latex_metrics.tex`
- `code/solutions/q18_temporal_backtest.csv`
- `code/solutions/q19_uncertainty_coverage.csv`
- `code/solutions/q20_fairness_mitigation_comparison.csv`
- `code/figures/q18_temporal_degradation.png`
- `code/figures/q19_coverage_vs_alpha.png`
- `code/figures/q20_fairness_tradeoff.png`
- `code/latex/project_report_full.pdf`
- `code/latex/project_report_full_fa.pdf`

---

## 15) راهنمای توضیح کد و شکل‌ها

برای توضیح بسیار دقیق کدهای پروژه و تحلیل کامل تک‌تک شکل‌ها، از سند زیر استفاده کنید:

- `code/docs/CODE_AND_FIGURES_GUIDE_FA.md`

این سند شامل:
- شرح ماژول‌به‌ماژول کدها (ورودی/خروجی/منطق/تصمیم فنی)
- توضیح کامل هر تصویر (هدف، روش تولید، نحوه خواندن، اثر تصمیمی، محدودیت)
- راهنمای ارائه شفاهی حرفه‌ای بر اساس خروجی‌ها
