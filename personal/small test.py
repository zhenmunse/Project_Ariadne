import pandas as pd
import os

# 1 read csv,look first 5 rows and statistic

print("TASK 1")

df_concepts = pd.read_csv('data\\ecs32a_concepts_required_full_v1.csv')
df_order = pd.read_csv('data\\ecs32a_teaching_order_required_full_v1.csv')

print("1.1 concept preview first 5 rows:")
print(df_concepts.head(5))
print("-"*50)

print("1.2 basic statistics values")
print(df_concepts.describe())
print("-"*50)

print("TASK 2")
print("handle missing values")

inital_rows = len(df_order)

df_order_cleaned = df_order.dropna(subset=['concept_name'])
final_rows = len(df_order_cleaned)
deleted_rows = inital_rows - final_rows

print(f"rows before cleaned:{inital_rows}")
print(f"rows after cleaned:{final_rows}")
print(f"deleted rows:{deleted_rows}")
print(f"successful delete {deleted_rows} in 'concept name'which has missing values")

print("\n"+"="*50+"\n")

print("TASK 3")
print("3 data deduplicatiion")

df_concepts_unique = df_concepts.drop_duplicates(subset=['node_id','concept_name'],keep='last')
print(f"number of rows before deduplication:{len(df_concepts)}")
print(f"number of rows after deduplication:{len(df_concepts_unique)}")
print(f"number of rows deleted during deduplication{len(df_concepts) - len(df_concepts_unique)}")
print("\n" + "="*50 + "\n")

print(f"TASK 4")
print(f"4 aggerate and mean bloom")

week_difficulty = df_concepts_unique.groupby('week_introduced')['bloom_level'].mean().reset_index()
print(f"mean difficulty(bloom) of node per week:")
print(week_difficulty)
print("\n"+"="*50+"\n")

print("TASK 5")
print("5 inner join")

df_merged = pd.merge(
    df_concepts_unique,
    df_order_cleaned,
    on=['node_id','concept_name', 'teaching_order', 'source_block'],
    how='inner'
)
print("preview of 5 merged data")
print(df_merged.head(5))
print(f"total data lines after merge:{len(df_merged)}")
