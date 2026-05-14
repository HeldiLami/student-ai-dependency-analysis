import pandas as pd
import numpy as np



RAW_PATH = 'data/raw/ai_dependency_career_anxiety_students.csv'
CLEAN_PATH = 'data/clean/ai_students_clean.csv'

def load_dataset():

  df = pd.read_csv(RAW_PATH)
  print(f'Kemi ngarkuar {len(df)} rreshta dhe {df.shape[1]} kolona.')
  print(df.dtypes())
  return df

df_raw = load_dataset()

print('--------------------------------------------------------------------------------------')

def column_standardization(df):
  df = df.copy()
  raw_columns = df.columns.tolist()
  df.columns = (df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True))

  columns_changed = [(old_column, new_column) for old_column, new_column  in zip (raw_columns, df.columns) if old_column != new_column]

  if columns_changed:
    print(f"U rregulluan {len(columns_changed)} kolona")
  return df

df_clean = column_standardization(df_raw)
print(df_clean.columns.tolist())

print('--------------------------------------------------------------------------------------')


def optimization_fix_type():
  df = df.copy()
  categorical_columns = ['gender', 'degree_type', 'stream', 'college_tier',
                         'urban_or_rural', 'primary_ai_tools_used', 'uses_ai_for_assignments']
  #arsyeja pse e bejme kete eshte qe ne dataset kto te dhena mos te ruhen si string, por si kategori,
  #menyre eficente qe mos te perdorim shume memorie kur kemi vetemdisa kategori per nje kolone
  for column in categorical_columns:
    df[column] = df[column].astype('category')

  if 'seeks_career_counseling' in df.columns:
    df['seeks_career_counseling'] = df['seeks_career_counseling'].astype('boolean')

  df = df.convert_dtypes()
  
  return df

