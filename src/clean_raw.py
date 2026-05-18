import pandas as pd
import numpy as np



RAW_PATH = 'data/raw/ai_dependency_career_anxiety_students.csv'
CLEAN_PATH = 'data/clean/ai_students_clean.csv'

def load_dataset():

  df = pd.read_csv(RAW_PATH)
  print(f'Kemi ngarkuar {len(df)} rreshta dhe {df.shape[1]} kolona.')
  print(df.dtypes)
  return df


def column_standardization(df):
  df = df.copy()
  raw_columns = df.columns.tolist()
  df.columns = (df.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True))

  columns_changed = [(old_column, new_column) for old_column, new_column  in zip (raw_columns, df.columns) if old_column != new_column]

  if columns_changed:
    print(f"U rregulluan {len(columns_changed)} kolona")
  return df


def optimization_fix_type(df):
  df = df.copy()
  categorical_columns = ['gender', 'degree_type', 'stream', 'college_tier',
                         'urban_or_rural', 'primary_ai_tools_used', 'uses_ai_for_assignments']
  #arsyeja pse e bejme kete eshte qe ne dataset kto te dhena mos te ruhen si string, por si kategori,
  #menyre eficente qe mos te perdorim shume memorie kur kemi vetemdisa mundesi kategorish per nje kolone
  for column in categorical_columns:
    df[column] = df[column].astype('category')

  if 'seeks_career_counseling' in df.columns:
    df['seeks_career_counseling'] = df['seeks_career_counseling'].astype('boolean')

  df = df.convert_dtypes() #ben qe kolonat te cilat nuk marin dot vleren null, ta marin ate
  
  return df



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

#----------------------------------------------------------------------------------------------------

def remove_duplicates(df):
  df = df.copy()
  row_num_before = len(df)

  df = df.drop_duplicates(keep= 'first')

  if 'student_id' in df.columns:
    df = df.drop_duplicates(subset = 'student_id', keep = 'first')
    # kontrollojme edhe nese id e nje studenti perseritet dy here qe ta fshijme dhe ate rast

  duplicates_dropped = row_num_before - len(df)
  if duplicates_dropped:
    print(f"Numri i duplikatave te fshira: {duplicates_dropped}")
  else:
    print("Dataseti nuk ka duplikata")
    
  return df



#----------------------------------------------------------------------------------------------------
def outliers_check(df):
  df = df.copy()

  columns = [
          'daily_ai_tool_usage_hrs',
          'self_learning_hours_per_week',
          'daily_study_hours',
          'social_media_hrs_per_day',
          'sleep_hours',
          'weekly_job_application_count',
      ]
  
  for column in columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3-q1
    qLow = q1 - 1.5 * iqr
    qHigh = q3 + 1.5 * iqr

    outliers_count = ((df[column] < qLow)| (df[column] > qHigh)).sum()
    if outliers_count > 0:
      df[column] = df[column].astype(float).clip(lower = qLow, upper = qHigh)
      print(f"{outliers_count} vlera outliers u kufizuan ne kolonen: {column}")

    else:
      print(f" Kolona [{column}] nuk kishte outliers")
  
  return df

#----------------------------------------------------------------------------------------------------


def text_standardization(df):
  df = df.copy()

  text_columns = [
        'gender', 'degree_type', 'stream', 'college_tier',
        'urban_or_rural', 'uses_ai_for_assignments', 'primary_ai_tools_used'
    ]
  
  for column in text_columns:
    if hasattr(df[column], 'cat'):
      df[column] =df[column].astype(str).str.strip().str.title().astype('category')
      # kthen kolonen nga kategori ne string , heq hapsirat par mbrapa, ben shkronjen e pare kapitale, e kthen perseri ne kategori.
    else:
      df[column] = df[column].str.strip().str.title()
  
  manual_fix = {
      'Chatgpt':        'ChatGPT',
      'Github Copilot': 'GitHub Copilot',
  }
  if 'primary_ai_tools_used' in df.columns:
      df['primary_ai_tools_used'] = (
          df['primary_ai_tools_used']
          .astype(str)
          .replace(manual_fix)
          .astype('category')
      )

  return df
  

def column_mapping(df):
  if 'seeks_career_counseling' in df.columns:
        df['seeks_career_counseling'] = (
            df['seeks_career_counseling']
            .astype(str)
            .map({'True': 'Yes', 'False': 'No', '1.0': 'Yes', '0.0': 'No',
                  '1': 'Yes', '0': 'No'})
            .astype('category')
        )
  return df


def clean_dataset():

    df = load_dataset()
    print('--------------------------------------------------------------------------------------')

    df = column_standardization(df)
    print('--------------------------------------------------------------------------------------')

    df = optimization_fix_type(df)
    print('--------------------------------------------------------------------------------------')

    df = null_handling(df)
    print('--------------------------------------------------------------------------------------')

    df = remove_duplicates(df)
    print('--------------------------------------------------------------------------------------')

    df = outliers_check(df)
    print('--------------------------------------------------------------------------------------')

    df = text_standardization(df)
    print('--------------------------------------------------------------------------------------')

    df = column_mapping(df)

    df.to_csv(CLEAN_PATH, index=False)

    print(f"Madhesia perfundimtare e datasetit: {len(df):,} || Kolona: {df.shape[1]} || Mungesa totale (NaN): {df.isnull().sum().sum()}")
    print(f"Skedari  perfundimtar u ruajt tek: {CLEAN_PATH}")
    
    return df

df_final = clean_dataset()
