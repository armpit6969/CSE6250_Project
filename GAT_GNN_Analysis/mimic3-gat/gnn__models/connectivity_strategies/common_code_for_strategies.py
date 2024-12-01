import math 
import pickle 
import os 
import pandas as pd 
from tqdm import tqdm
from gnn__models.config.config import merge_features, orig_rules, rules, rules_normal, all_mappings 


def create_category_files(FOLDER_STRUCTURE, spstr=''):
    fname = f'dict_dfs_cat.pk{spstr}'
    if not os.path.exists(fname):
        print(f'Creating: fname: {fname}')
        
        # Parse all files in the folder path recursively
        def parse_folder(path):
            count = 0
            for root, dirs, files in os.walk(path):
                for file in files:
                    count += 1 
                    if count % 10000 == 0:
                        print(count)
                    if count > 60000:
                        raise Exception('too many something is wrong ')
                    if 'episode' in file:
                        yield os.path.join(root, file)

        filenames = []
        dict_dfs = {}

        # Ensure the folder structure exists and is not empty
        assert os.path.exists(FOLDER_STRUCTURE), f'folder: {FOLDER_STRUCTURE} not existing'
        assert len(os.listdir(FOLDER_STRUCTURE)) > 0, f'folder: {FOLDER_STRUCTURE} not containing files'

        # Iterate over parsed files and load them into a dictionary
        for i, file in enumerate(parse_folder(FOLDER_STRUCTURE)):
            filenames.append(file)
            df = pd.read_csv(file)
            dict_dfs[file] = df 
            if spstr != '' and i > 100:
                break

        print(' Conversion of categorical values')

        # Invert the mapping dictionary for value replacement
        def inverse_mapping(mapping):
            inverse = {}
            for key in mapping:
                for value in mapping[key]:
                    inverse[value] = key
            return inverse

        all_mappings_inv = {col: inverse_mapping(all_mappings[col]) for col in all_mappings.keys()}

        print('convert_values_of_categorical_columns')
        
        # Convert categorical columns to numeric values
        def convert_values_of_categorical_columns(df):
            df.replace(all_mappings_inv, inplace=True)
            categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
            for col in categorical_columns:
                # Extract the first element if the column contains lists
                df[col] = df[col].str.split(" ").apply(
                    lambda x: x[0] if isinstance(x, list) else x)
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        dict_dfs_processed = {}
        # Process each DataFrame by converting categorical columns
        for k, df in dict_dfs.items():
            res = convert_values_of_categorical_columns(df.copy())
            if res is None:
                print(k)
                break
            dict_dfs_processed[k] = res

        print(' 3m create categories per column & df')
        dict_dfs_cat = {}
        # Create categories for each column in each DataFrame
        for k, df in tqdm(dict_dfs_processed.items()):
            df = df.copy()
            df.drop('Hours', axis=1, inplace=True)

            # Calculate BMI if Height and Weight are present
            if (df['Height'] is not None) and (df['Weight'] is not None):   
                df['Height'].fillna(method='ffill', inplace=True)
                df['Height'].fillna(method='bfill', inplace=True)
                
                df['BMI'] = (df['Weight'] / ((df['Height'] / 100) ** 2))
            else:
                df['BMI'] = math.nan
            # Drop unnecessary columns
            df.drop(['Height', 'Weight', 'Fraction inspired oxygen'], axis=1, inplace=True)
            
            # Apply rules to categorize columns
            for col in rules.keys():
                arr1 = df[col].apply(lambda x: [rule(x) for rule in rules[col]])
                arr = arr1.apply(lambda x: x.index(True) if True in x else None)
                df[col] = arr

            # Store unique categories for each column
            dict_dfs_cat[k] = df.apply(lambda x: x.unique())
        # Save the categorized DataFrames to a pickle file
        pickle.dump(dict_dfs_cat, open('dict_dfs_cat.pk', 'wb'))

    print(f'Phase 2')
    fname = f'alldfs_categories{spstr}.csv'

    if not os.path.exists(fname):
        # Load the categorized DataFrames from the pickle file
        dict_dfs_cat = pickle.load(open('dict_dfs_cat.pk', 'rb'))
        print('### remove nans and Nones')
        dict_dfs_cat_uniform = {}
        # Remove NaN and None values from the categories
        for k, series in tqdm(dict_dfs_cat.items()):
            series = series.apply(lambda x: sorted(x))
            series = series.apply(lambda x: [i for i in x if i is not None])
            try:
                if len(series) > 1:
                    series = series.apply(lambda x: [i for i in x if not math.isnan(i)])
            except:
                import pdb; pdb.set_trace() 
            dict_dfs_cat_uniform[k] = series

        # Create a DataFrame from the uniform categories and save to CSV
        alldfs = pd.DataFrame.from_dict(dict_dfs_cat_uniform, orient='index',
                                       columns=list(dict_dfs_cat_uniform.values())[0].index.tolist())
        alldfs.to_csv(fname)