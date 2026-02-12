# پاسخ‌نامه کامل فارسی (راهنمای تصحیح)

این سند پاسخ مرجع برای نسخه فارسی تمرین نهایی توسعه‌یافته است و باید همراه با خروجی‌های اجرایی (`run_summary.json`، شکل‌ها، و کوئری SQL) استفاده شود.

---

## Q1) چرخه عمر علم داده و صورت‌بندی مسئله
پاسخ کامل باید شامل چهار بخش باشد:
1. تعریف تصمیم: مدل قرار است چه تصمیمی را پشتیبانی کند؟
2. معیار موفقیت: حداقل یک معیار رتبه‌بندی، یک معیار کالیبراسیون/آستانه‌ای.
3. ریسک‌ها: نشت داده، drift، تغییرات سیاست، سوگیری تاریخی.
4. استقرار: برنامه پایش، بازآموزی و مسیر بازبینی انسانی.

---

## Q2) عملیات داده و EDA
موارد ضروری:
- حسابرسی `dtype`، مقادیر گمشده، تکراری‌ها، پرت‌ها
- حداقل 6 تا 8 نمودار معنادار با تفسیر عملیاتی
- تابع preprocessing قابل آزمون

نمودارهای حداقلی پیشنهادی:
- توزیع هدف
- Missingness profile
- Correlation heatmap
- Migration rate by country
- Distribution of key numeric features
- Outlier plots

---

## Q3) استنباط آماری
الگوی پاسخ صحیح:
- تعریف `H0` و `H1`
- انتخاب آزمون آماری متناسب با نوع داده
- بررسی مفروضات آزمون
- تفسیر درست p-value و CI

**قاعده:** p-value احتمال درست‌بودن H0 نیست؛ احتمال مشاهده داده تحت فرض H0 است.

---

## Q4) طراحی بصری و روایت
پاسخ قوی:
- KPIهای متصل به تصمیم‌گیری
- توجیه طراحی رنگ/مقیاس/annotation
- نمایش یک مثال گمراه‌کننده و نسخه اصلاح‌شده

---

## Q5) SQL پیشرفته
الگوی مرجع میانگین متحرک:

```sql
WITH citation_velocity AS (
  SELECT UserID, Country_Origin, Year, Research_Citations,
         AVG(Research_Citations) OVER (
           PARTITION BY Country_Origin
           ORDER BY Year
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
         ) AS moving_avg_citations
  FROM Professionals_Data
)
SELECT *, DENSE_RANK() OVER (
  PARTITION BY Country_Origin ORDER BY moving_avg_citations DESC
) AS country_rank
FROM citation_velocity;
```

معیار نمره کامل:
- window frame صحیح
- منطق زمانی صحیح
- کوئری cohort معتبر با CTE

---

## Q6) نشت داده و معماری
تشخیص نشت:
- `Visa_Approval_Date`: نشت مستقیم (post-outcome)
- `Last_Login_Region`: نشت زمانی بالقوه
- `Passport_Renewal_Status`: پراکسی زمانی بالقوه
- `Years_Since_Degree`: در صورت point-in-time قابل قبول

شاخص‌های تشخیصی اجرای پروژه:
- corr(visa_present, target)=1.0
- P(Migration=1 | visa_present)=1.0
- P(Migration=1 | visa_absent)=0.0

معماری قابل قبول:
- Bronze/Silver/Gold
- Feature Store با point-in-time joins
- Offline/Online parity

---

## Q7) Elastic Net و تفسیر آماری
تابع هزینه:

\[
J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda_1\sum_j|\theta_j| + \frac{\lambda_2}{2}\sum_j\theta_j^2
\]

مشتق مختصه‌ای:

\[
\nabla_{\theta_j}J = \frac{1}{m}\sum_i(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)} + \lambda_1\partial|\theta_j| + \lambda_2\theta_j
\]

زیرگرادیان در صفر:
- اگر `theta_j > 0`: مقدار `+1`
- اگر `theta_j < 0`: مقدار `-1`
- اگر `theta_j = 0`: بازه `[-1, 1]`

تفسیر آماری نمونه:
- `p-value=0.003` و CI `[0.18, 0.86]` => رد `H0: beta=0` و اثر مثبت معنادار.

---

## Q8) بهینه‌سازی در ravine
- SGD در محور پرخمیدگی نوسان شدید دارد.
- Momentum نوسان را می‌کاهد و در جهت پایدار سرعت می‌گیرد.
- Adam با تخمین ممان اول/دوم و نرخ یادگیری موثر پارامتری، در داده ناهم‌مقیاس بهتر عمل می‌کند.

رابطه momentum:

\[
v_t = \beta v_{t-1} + \eta g_t, \quad \theta_{t+1}=\theta_t-v_t
\]

---

## Q9) مدل‌های غیرخطی
### SVM-RBF
در بیش‌برازش باید `gamma` کاهش یابد تا مرز تصمیم نرم‌تر شود.

### درخت تصمیم و هرس
رابطه هزینه-پیچیدگی:

\[
R_\alpha(T)=R(T)+\alpha|T|
\]

با افزایش `alpha`:
- اندازه درخت کم می‌شود
- bias افزایش می‌یابد
- variance کاهش می‌یابد

---

## Q10) PCA
اگر مقادیر ویژه `lambda1, lambda2, lambda3` باشند:
- EVR هر مولفه: `lambda_k / sum(lambda)`
- EVR دو مولفه اول: `(lambda1+lambda2) / sum(lambda)`

تفسیر: مقدار ویژه، واریانس توضیح‌داده‌شده توسط مولفه متناظر است.

---

## Q11) KMeans و Elbow
- WCSS با افزایش K کاهش می‌یابد.
- کاهش WCSS دارای بازده نزولی است.
- نقطه elbow جایی است که افزایش K سود معنادار جدیدی ایجاد نمی‌کند.

DBSCAN به‌عنوان مکمل برای خوشه‌های غیرکروی و نقاط نویزی مفید است.

---

## Q12) شبکه‌های عصبی
پاسخ کامل باید شامل:
- معماری، loss، optimizer، برنامه آموزش
- کنترل بیش‌برازش (dropout/early stopping/regularization)
- مقایسه با baseline کلاسیک

---

## Q13) LLM Agent
جریان مرجع:
`plan -> retrieve -> reason -> verify`

معیار ارزیابی:
- faithfulness
- hallucination rate
- safety/compliance

الزام: تعیین fallback و محدودیت دسترسی ابزارها.

---

## Q14) عدالت و حاکمیت
حداقل خروجی:
- جدول متریک زیرگروهی (کشور/تحصیلات/...)
- تحلیل خطر proxy bias
- سیاست human-in-the-loop
- مسیر اعتراض و بازبینی

---

## Q15) کالیبراسیون و آستانه
- منحنی کالیبراسیون و متریک‌های احتمالی (Brier/ECE)
- دو آستانه: بیشینه‌سازی F1 و کمینه‌سازی هزینه نامتقارن خطا
- توصیه نهایی باید بر اساس هزینه/کاربرد باشد، نه پیش‌فرض ۰٫۵

---

## Q16) درفت
- تقسیم مرجع/جاری (ترجیحاً زمانی)
- PSI برای ویژگی‌های عددی و رتبه‌بندی
- یک شاخص درفت برای دسته‌ای‌ها (مثل JS divergence)
- سیاست هشدار/بحرانی و محرک بازآموزی

---

## Q17) ریکورس
- تعیین ویژگی‌های قابل اقدام
- کمینه تغییر لازم برای عبور از آستانه برای نمونه‌های نزدیک مرز
- نرخ موفقیت ریکورس و میانه مداخله برای هر ویژگی
- بحث امکان‌پذیری و اخلاق مداخله‌ها

---

## Q18) اعتبارسنجی زمانی و افت عملکرد
- اجرای اعتبارسنجی غلطان زمانی (یا fallback مستند در نبود زمان معتبر)
- گزارش متریک‌های هر fold (حداقل `AUC` و `F1`)
- گزارش افت نسبت به fold اول و تحلیل ارتباط با شاخص درفت

خروجی‌های مورد انتظار:
- `code/solutions/q18_temporal_backtest.csv`
- `code/figures/q18_temporal_degradation.png`

---

## Q19) کمی‌سازی عدم‌قطعیت
- پیاده‌سازی روش `conformal` یا روش معتبر معادل برای بازه/امتیاز اطمینان
- گزارش پوشش تجربی در چند سطح اطمینان
- تحلیل پهنای بازه و ریسک کم‌پوششی

خروجی‌های مورد انتظار:
- `code/solutions/q19_uncertainty_coverage.csv`
- `code/figures/q19_coverage_vs_alpha.png`

---

## Q20) مداخله عدالت الگوریتمی
- محاسبه خط پایه متریک‌های عدالت زیرگروهی
- اجرای مداخله (مثل `reweighing`) و مقایسه قبل/بعد
- گزارش مصالحه عدالت-عملکرد و نتیجه قید سیاستی

خروجی‌های مورد انتظار:
- `code/solutions/q20_fairness_mitigation_comparison.csv`
- `code/figures/q20_fairness_tradeoff.png`

---

## بلوک Bonus J
- \textbf{DAG علّی}: گراف، مجموعه تعدیل، محدودیت‌های شناسایی.
- \textbf{Conformal/عدم‌قطعیت}: بازه یا امتیاز اطمینان با پوشش تجربی.
- \textbf{اعتبارسنجی زمانی}: مقایسه اسپلـیت زمانی با تصادفی و تحلیل افت عملکرد.
- \textbf{سروینگ آنلاین/استریمینگ}: طرح ویژگی‌های تازه، SLA، نگهبان OOD/درفت و مسیر rollback.

---

## Capstone - SHAP
تفاوت کلیدی:
- `base_value`: مقدار پایه مدل
- `output_value`: خروجی نمونه خاص
- مجموع SHAP values = `output_value - base_value`

اگر کاندید با citations بالا «عدم مهاجرت» بگیرد، باید سهم منفی سایر ویژگی‌ها (و تعامل‌ها) توضیح داده شود.

---

## مرجع خروجی‌های پروژه
- `code/solutions/q1_moving_average.sql`
- `code/solutions/run_summary.json`
- `code/solutions/report_stats.json`
- `code/solutions/q6_fairness_country_rates.csv`
- `code/figures/q3_ravine_optimizers.png`
- `code/figures/q4_svm_gamma_sweep.png`
- `code/figures/q4_tree_pruning_curve.png`
- `code/figures/q5_kmeans_elbow.png`
- `code/figures/q6_shap_force_plot.png`
- `code/figures/q6_shap_summary.png`
- `code/solutions/q18_temporal_backtest.csv`
- `code/solutions/q19_uncertainty_coverage.csv`
- `code/solutions/q20_fairness_mitigation_comparison.csv`
- `code/figures/q18_temporal_degradation.png`
- `code/figures/q19_coverage_vs_alpha.png`
- `code/figures/q20_fairness_tradeoff.png`
