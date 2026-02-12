# Data Dictionary and Leakage Notes

Dataset: `data/GlobalTechTalent_50k.csv`

## Core Columns

- `UserID`: unique row identifier
- `Country_Origin`: origin country category
- `Education_Level`: categorical education level
- `Field`: professional domain
- `GitHub_Activity`: activity score (0-100)
- `Research_Citations`: citation count
- `Industry_Experience`: years of experience
- `Publications`: publication count
- `Remote_Work`: binary indicator (0/1)
- `Age`: age in years
- `Migration_Status`: target label (1=Migration, 0=No Migration)

## Candidate Auxiliary Columns

- `Visa_Approval_Date`: visa approval timestamp proxy
- `Years_Since_Degree`: years elapsed since graduation
- `Last_Login_Region`: last observed login region
- `Passport_Renewal_Status`: categorical passport/renewal state

## Leakage Classification

- `Visa_Approval_Date`: **Direct leakage** (post-outcome information)
- `Last_Login_Region`: **Potential temporal leakage** if captured after migration
- `Passport_Renewal_Status`: **Potential temporal leakage** depending on timestamp policy
- `Years_Since_Degree`: generally safe if degree date is known at inference time

## Modeling Rule Used in This Project

`Visa_Approval_Date` is always dropped from model features before training.
