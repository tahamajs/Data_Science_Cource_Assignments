import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic dataset generator for GlobalTechTalent_50k
# Generates 50k rows with plausible correlations between features and migration status.

def generate_dataset(n_rows: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    countries = [
        "USA", "Canada", "Germany", "UK", "France", "Iran", "India", "China", "Brazil", "Australia",
        "South Africa", "Sweden", "Netherlands", "Singapore", "UAE"
    ]
    education_levels = ["Bachelors", "Masters", "PhD", "Bootcamp"]
    fields = ["Software", "Data Science", "Security", "Hardware", "AI Research"]

    country_origin = rng.choice(countries, n_rows)
    education_level = rng.choice(education_levels, n_rows, p=[0.4, 0.35, 0.2, 0.05])
    field = rng.choice(fields, n_rows)

    github_activity = rng.normal(loc=55, scale=18, size=n_rows).clip(0, 100)
    research_citations = rng.lognormal(mean=5.3, sigma=0.7, size=n_rows).clip(0, 8000)
    industry_experience = rng.normal(loc=7, scale=4, size=n_rows).clip(0, 35)
    publications = rng.poisson(lam=6, size=n_rows)
    remote_work = rng.choice([0, 1], size=n_rows, p=[0.55, 0.45])
    age = rng.normal(loc=32, scale=6, size=n_rows).clip(20, 60)

    # Linear-ish latent propensity for migration
    base = (
        0.015 * github_activity
        + 0.0006 * research_citations
        + 0.08 * remote_work
        + 0.06 * (education_level == "PhD")
        + 0.04 * (education_level == "Masters")
        - 0.03 * (country_origin == "USA")  # less likely to migrate out
        - 0.02 * (country_origin == "Canada")
        + 0.05 * (country_origin == "India")
        + 0.04 * (country_origin == "Iran")
        + 0.01 * publications
        - 0.005 * np.abs(industry_experience - 8)  # peak intent mid-career
    )

    # Non-linear saturation
    logits = base - 1.5 + rng.normal(0, 0.5, n_rows)
    prob = 1 / (1 + np.exp(-logits))
    migration_status = rng.binomial(1, prob)

    visa_approval_date = np.where(migration_status == 1, rng.integers(2015, 2025, n_rows), np.nan)
    years_since_degree = (age - rng.normal(loc=23, scale=2, size=n_rows)).clip(0, 40)
    last_login_region = country_origin  # weak proxy, placeholder
    passport_renewal_status = rng.choice(["Current", "Expired", "In Process"], size=n_rows, p=[0.6, 0.2, 0.2])

    df = pd.DataFrame(
        {
            "UserID": np.arange(1, n_rows + 1),
            "Country_Origin": country_origin,
            "Education_Level": education_level,
            "Field": field,
            "GitHub_Activity": github_activity.round(2),
            "Research_Citations": research_citations.round(0),
            "Industry_Experience": industry_experience.round(1),
            "Publications": publications,
            "Remote_Work": remote_work,
            "Age": age.round(1),
            "Migration_Status": migration_status,
            "Visa_Approval_Date": visa_approval_date,
            "Years_Since_Degree": years_since_degree.round(1),
            "Last_Login_Region": last_login_region,
            "Passport_Renewal_Status": passport_renewal_status,
        }
    )

    return df


def main():
    output = Path(__file__).resolve().parent.parent / "data" / "GlobalTechTalent_50k.csv"
    df = generate_dataset()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")


if __name__ == "__main__":
    main()
