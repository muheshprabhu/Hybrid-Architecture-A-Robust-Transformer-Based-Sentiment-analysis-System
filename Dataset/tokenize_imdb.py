import pandas as pd
import os
from transformers import AutoTokenizer


# ==========================================
# 1. DEFINE PATHS
# ==========================================

folder_path = r"D:\MINI PROJECT VII\Dataset\IMDB"

input_path = os.path.join(
    folder_path,
    "IMDB_Preprocessed.csv"
)

output_path = os.path.join(
    folder_path,
    "IMDB_Tokenized.csv"
)


# ==========================================
# 2. LOAD PREPROCESSED DATASET
# ==========================================

print("Loading preprocessed dataset...")

df = pd.read_csv(input_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns)


# ==========================================
# 3. LOAD DEBERTA TOKENIZER
# ==========================================

print("\nLoading DeBERTa tokenizer...")

MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully!")


# ==========================================
# 4. TOKENIZATION SETTINGS
# ==========================================

MAX_LENGTH = 256


# ==========================================
# 5. TOKENIZE THE PREPROCESSED REVIEWS
# ==========================================

print("\nStarting tokenization...")

texts = df["preprocessed_review"].fillna("").tolist()

encodings = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=MAX_LENGTH,
    return_attention_mask=True
)

print("Tokenization completed!")


# ==========================================
# 6. ADD TOKENIZED OUTPUT TO DATAFRAME
# ==========================================

df["input_ids"] = encodings["input_ids"]

df["attention_mask"] = encodings["attention_mask"]


# ==========================================
# 7. CONVERT SENTIMENT LABELS
# ==========================================

label_mapping = {
    "negative": 0,
    "positive": 1
}

df["label"] = df["sentiment"].map(label_mapping)

print("\nLabel distribution:")
print(df["label"].value_counts())


# ==========================================
# 8. DISPLAY EXAMPLE
# ==========================================

print("\n" + "=" * 60)

print("PREPROCESSED TEXT:")
print(df["preprocessed_review"].iloc[0])

print("\nINPUT IDS:")
print(df["input_ids"].iloc[0])

print("\nATTENTION MASK:")
print(df["attention_mask"].iloc[0])

print("\nLABEL:")
print(df["label"].iloc[0])


# ==========================================
# 9. SAVE TOKENIZED DATASET
# ==========================================

print("\nSaving tokenized dataset...")

df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("TOKENIZATION COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("Saved at:")
print(output_path)

print("\nFinal dataset shape:")
print(df.shape)