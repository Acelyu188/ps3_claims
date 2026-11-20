
from ps3.data._load_transform import load_transform
from ps3.data._sample_split import create_sample_split


df = load_transform()
print("Data shape:", df.shape)


df = create_sample_split(
    df,
    id_column="IDpol",
    training_frac=0.8
)


print("\nTrain/Test split:")
print(df["sample"].value_counts())


print("\nFirst 5 rows:")
print(df[["IDpol", "Exposure", "ClaimAmount", "sample"]].head())


assert df["sample"].isin(["train", "test"]).all()

print("\nData loading and splitting successful!")
