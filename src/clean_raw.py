import pandas as pd
import numpy as np



RAW_PATH = 'data/raw/ai_dependency_career_anxiety_students.csv'
CLEAN_PATH = 'data/clean/ai_students_clean.csv'

def load_dataset():

  df = pd.read_csv(RAW_PATH)
  print(f'Kemi ngarkuar {len(df)} rreshta dhe {df.shape[1]} kolona.')
  print(df.dtypes)
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

df_standardized = column_standardization(df_raw)
print(df_standardized.columns.tolist())

print('--------------------------------------------------------------------------------------')


def optimization_fix_type(df):
  df = df.copy()
  categorical_columns = ['gender', 'degree_type', 'stream', 'college_tier',
                         'urban_or_rural', 'primary_ai_tools_used', 'uses_ai_for_assignments']
  #arsyeja pse e bejme kete eshte qe ne dataset kto te dhena mos te ruhen si string, por si kategori,
  #menyre eficente qe mos te perdorim shume memorie kur kemi vetemdisa kategori per nje kolone
  for column in categorical_columns:
    df[column] = df[column].astype('category')

  if 'seeks_career_counseling' in df.columns:
    df['seeks_career_counseling'] = df['seeks_career_counseling'].astype('boolean')

  df = df.convert_dtypes() #makes not nullable columns take null values
  
  return df

print('--------------------------------------------------------------------------------------')


optimization = optimization_fix_type(df_standardized)
print(optimization.dtypes)

def null_handling(df):
  df = df.copy()

  row_num = len(df)
  null_values = df.isnull().sum()
  null_columns = null_values[null_values>0]
  print(f"Numri i vlerave null para rregullimi: {df.isnull().sum().sum()}")
  print(f"Kolonat qe kane null:\n{null_columns.index.tolist()}")

  for column, num in null_columns.items():
    percentage = (num/row_num) * 100
    print(f"Kolona: {column}: {num} mungesa [{percentage:.2f}%]")

  print('-------------------------------------------------------------------------')

  if 'primary_ai_tools_used' in df.columns:
    df['primary_ai_tools_used'] = (df['primary_ai_tools_used'].cat.add_categories('Unknown').fillna('Unknown'))
    # per faktin qe e shtuam kolonen primary_ai_tools_used si kategori, duhet te shtojme opsionin
    # Unknown si kategori gjithashtu. Vlerat qe mungojne i shenojme me unknown.
    # per mendimin tim plotesimi i vlerave boshe me unknown eshte menyra me e mire sepse nuk ka ndonje trend,
    # ose diku ku mund te bazohemi per plotesimin e asaj kategorie. Gjithashtu eshte nje numer shum i madh perqindjeje
    # mungese, keshtuqe plotesimi me nje vlere random do kishte shume ndikim.

  hour_columns = ['sleep_hours', 'social_media_hrs_per_day', 'self_learning_hours_per_week']
  for column in hour_columns:
    if column in df.columns:
      median = round(df[column].median(), 2)
      df[column] = df[column].fillna(median) 
      print(f"Kolona [{column}] u plotesua me vleren: {median}")

  #per plotesimin e ketyra kolonave kisha dy metoda ne mendje. njera prej tyre ishte mesatarja dhe tjetra ishte mesorja.
  #pavaresisht se nuk kam llogaritur akoma vlerat outlying (ato qe mund te jene shum larg vlerave normale), mendoj qe ne rast
  # se do kishim dissa vlera te oreve goxha te shperndara, do te ndikonin per keq ne plotesimin e ketyre vlerave null.
  # keshtu qe mendoj qe median eshte menyra me e sakte.

  if 'seeks_career_counseling' in df.columns:
    df['seeks_career_counseling'] = df['seeks_career_counseling'].fillna(False)
  
  # sipas llogarive qe kam bere me poshte vetem 28.16% e studenteve kerkojne career counseling
  # keshtu qe duke marre parasysh qe shumica e studenteve nuk kerkojne, vlera me te cilen do plotesoj
  # vlerat null te kesaj kolone jane 0 (False)
  # po e komentoj rreshtin me poshte qe mos te ngarkoj afishimin e te dhenave.

  # print(f'{(df['seeks_career_counseling'].sum()) / len(df) *100}% e studenteve konsulohen')

  print(f"Numri i vlerave null pas rregullimi: {df.isnull().sum().sum()}")
  return df.convert_dtypes()

df_1 =null_handling(optimization)
print(df_1)

def remove_duplicates(df):
  df = df.copy()
  row_num_before = len(df)
  df = df.drop_duplicates(keep= 'first')
  print(row_num_before)



print(remove_duplicates(df_1))