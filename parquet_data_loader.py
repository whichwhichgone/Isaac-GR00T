import pandas as pd

parquet_path = "/liujinxin/liyifan/Isaac-GR00T/dataset/2026-05-19_clean_desk_place_sofa_g1_fast_test/data/chunk-000/episode_000000.parquet"

df = pd.read_parquet(parquet_path)

print(df)
print(df.columns)
print(df.dtypes)