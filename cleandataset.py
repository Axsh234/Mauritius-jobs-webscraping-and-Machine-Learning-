import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================
# 1. Load dataset
# ==========================
df = pd.read_csv("myjob.csv")
print("Original shape:", df.shape)

# Normalize column names (lowercase, strip spaces, replace spaces with "_")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Normalized columns:", df.columns.tolist())

# ==========================
# 2. Drop unnecessary column
# ==========================
if "link" in df.columns:
    df = df.drop(columns=["link"])
    print("Dropped column: link")

# ==========================
# 3. Drop duplicates
# ==========================
df = df.drop_duplicates()

# ==========================
# 4. Handle missing values
# ==========================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("")
    else:
        df[col] = df[col].fillna(df[col].median())

# ==========================
# 5. Salary cleaning
# ==========================
def _convert_to_number(val):
    if not val:
        return None
    val = val.strip().lower()
    if val.endswith("k"):
        return float(val[:-1]) * 1000
    elif val.endswith("m"):
        return float(val[:-1]) * 1_000_000
    val = re.sub(r"[^\d\.]", "", val)
    if val == "":
        return None
    return float(val)

def parse_salary(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower().replace(",", "").strip()

    # remove currency/extra words
    s = re.sub(r"(rs\.?|mur|usd|per month|per year|annum|yearly)", "", s)

    # handle non-numeric cases
    if any(x in s for x in ["negotiable", "not disclosed", "n/a", "confidential"]):
        return np.nan

    # detect ranges
    if "-" in s:
        parts = [p.strip() for p in s.split("-")]
        low = _convert_to_number(parts[0])
        high = _convert_to_number(parts[1]) if len(parts) > 1 else None
        if low is not None and high is not None:
            return (low + high) / 2
        else:
            return low if low is not None else np.nan

    return _convert_to_number(s)

# apply salary cleaning only if salary column exists
if "salary" in df.columns:
    df["salary"] = df["salary"].apply(parse_salary)
    df = df[df["salary"].notna()]  # drop rows without salary

    # remove extreme outliers
    low, high = df["salary"].quantile([0.01, 0.99])
    df = df[(df["salary"] >= low) & (df["salary"] <= high)]
    print("✅ Salary cleaned")
else:
    print("⚠️ No salary column found, skipping salary cleaning")

# ==========================
# 6. Clean text columns
# ==========================
def clean_text(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z\s]", " ", s)  # keep only letters/spaces
    s = re.sub(r"\s+", " ", s)       # collapse spaces
    return s

for col in ["title", "location"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

# ==========================
# 7. Parse dates & create features
# ==========================
if "date_posted" in df.columns and "closing_date" in df.columns:
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce", dayfirst=True)
    df["closing_date"] = pd.to_datetime(df["closing_date"], errors="coerce", dayfirst=True)

    df["days_open"] = (df["closing_date"] - df["date_posted"]).dt.days
    df["posted_month"] = df["date_posted"].dt.month
    df["posted_year"] = df["date_posted"].dt.year
    df["closing_month"] = df["closing_date"].dt.month
    df["closing_year"] = df["closing_date"].dt.year
    print("✅ Date features created")
else:
    print("⚠️ Missing date columns, skipping date features")

# ==========================
# 8. TF-IDF sanity check (on job titles)
# ==========================
if "title" in df.columns:
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    try:
        X_titles = tfidf.fit_transform(df["title"].astype(str))
        print("✅ TF-IDF created successfully. Shape:", X_titles.shape)
    except ValueError as e:
        print("⚠️ TF-IDF failed:", e)

# ==========================
# 9. Save cleaned dataset
# ==========================
df.to_csv("cleanedmyjob.csv", index=False)
print("Cleaned dataset saved as cleanedmyjob.csv")

# ==========================
# 10. Summary
# ==========================
print("Final shape:", df.shape)
print("Columns:", df.columns.tolist())
