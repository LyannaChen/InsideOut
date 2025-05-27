import pandas as pd
import argparse

def evaluate_cep(file_path):
    df = pd.read_csv(file_path)

    # # Filter entries in full generated scripts for mitigation evaluation
    # df['tmp'] = None
    # df2 = pd.read_csv('/local/elaine1wan/culture_interview/retrieval/retrieved/chatgpt/generated_scripts_reddit_top5.csv')
    # df2['tmp'] = None
    # for i in range(len(df2)):
    #     df2.loc[i,'tmp'] = '_'.join([df2.loc[i, 'country'], df2.loc[i, 'name'], str(df2.loc[i, 'age']), df2.loc[i, 'gender'],df2.loc[i, 'occupation']])
    # for i in range(len(df)):
    #     df.loc[i,'tmp'] = '_'.join([df.loc[i, 'country'], df.loc[i, 'name'],str(df.loc[i, 'age']), df.loc[i, 'gender'],df.loc[i, 'occupation']])
    # # print(len(df))
    # df = df[df['tmp'].isin(list(df2['tmp'].unique()))]
    # df.reset_index(drop=True, inplace=True)
    # # print(len(df))
    # # exit()

    df['answer'] = df['answer'].str.strip().str.lower()

    country_counts = df.groupby('country')['answer'].value_counts().unstack(fill_value=0)

    country_counts['yes_percentage'] = (country_counts['yes'] / (country_counts['yes'] + country_counts['no'])) * 100

    result = country_counts[['yes_percentage']]
    print(result)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True, help="input file path")
    args = parser.parse_args()

    evaluate_cep(args.input_file)