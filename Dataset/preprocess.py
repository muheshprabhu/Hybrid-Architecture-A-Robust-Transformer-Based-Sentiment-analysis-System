import pandas as pd
import re
import unicodedata
import os


# ==========================================
# 1. DEFINE DATASET PATH
# ==========================================

folder_path = r"D:\MINI PROJECT VII\Dataset\IMDB"

# Check files available in the folder
print("Files in folder:")
print(os.listdir(folder_path))


# ==========================================
# 2. LOAD THE IMDB DATASET
# ==========================================

# CHANGE THIS NAME if your actual CSV filename is different
file_path = os.path.join(folder_path, "IMDB Dataset.csv")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("\nColumns:")
print(df.columns)

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 3. TEXT PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    # Handle missing values
    if pd.isna(text):
        return ""

    # Convert text to string
    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Replace multiple spaces, tabs and newlines
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


# ==========================================
# 4. APPLY PREPROCESSING
# ==========================================

print("\nStarting preprocessing...")

df["preprocessed_review"] = df["review"].apply(preprocess_text)

print("Preprocessing completed!")


# ==========================================
# 5. SHOW BEFORE AND AFTER
# ==========================================

print("\n" + "=" * 60)
print("ORIGINAL REVIEW:")
print("=" * 60)

print(df["review"].iloc[0])

print("\n" + "=" * 60)
print("PREPROCESSED REVIEW:")
print("=" * 60)

print(df["preprocessed_review"].iloc[0])


# ==========================================
# 6. CHECK FOR EMPTY REVIEWS
# ==========================================

empty_reviews = (df["preprocessed_review"] == "").sum()

print("\nNumber of empty reviews:", empty_reviews)


# ==========================================
# 7. SAVE PREPROCESSED DATASET
# ==========================================

output_path = os.path.join(
    folder_path,
    "IMDB_Preprocessed.csv"
)

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Saved file location:")
print(output_path)

print("\nFinal dataset shape:")
print(df.shape)