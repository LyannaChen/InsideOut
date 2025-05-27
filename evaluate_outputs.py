import pandas as pd
from tqdm import tqdm 
from transformers import pipeline
import torch
import argparse
import os
import sys
from generation_util import * 
from huggingface_hub import login
hf_access_token = YOUR_HUGGINGFACE_TOKEN
login(token = hf_access_token)
## !!! find ways to parse model 

evaluation_prompt = [
    "Does the interviewer use language that suggests they are not part of the interviewee's culture? Answer only in yes or no."
]

def load_model(model_name):
    if model_name == "llama":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype = torch.bfloat16,
            device_map="auto"
        )
    elif model_name == "mistral":
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        pipe = pipeline(
            "text-generation",
            model= model_id,
            torch_dtype="auto", 
            device_map="auto"
        )
    elif model_name == "qwen":
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        pipe = pipeline(
            "text-generation",
            model = model_id,
            torch_dtype="auto", 
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    print(f"Loading model: {model_id}")
    return pipe

def generate_output(pipe):
    def generate(prompt, max_new_tokens=1024):
        messages = [{"role": "user", "content": prompt}]
        outputs = pipe(
            messages, 
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.5,
            temperature=0.1,
            top_p=0.75,
            num_beams=2
        )
        return outputs[0]["generated_text"][-1]["content"]
    return generate

def evaluate_csv(args, file_path, output_path, output_file_name, evaluation_prompt, generator):
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

    output_file = os.path.join(output_path, output_file_name) # "evaluation_results2.csv")
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        existing_evaluation = set(
            tuple(row) for row in existing_df[["scripts", "question"]].values.tolist()
        )
        print(f"Existing evaluation loaded. Number of existing evalution: {len(existing_evaluation)}")
    else:
        existing_df = pd.DataFrame(columns = ['country', 'name', 'gender', 'age', 'occupation', 'prompts', 'scripts', 'question', 'answer'])
        existing_evaluation = set()

    print(f"Evaluating {len(df)} scripts. Outputing {len(df) * len(questions)} evaluations.")

    with open(output_file, 'a', encoding='utf-8', newline='') as f:
        write_header = f.tell() == 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            country = row["country"]
            name = row["name"]
            age = row["age"]
            gender = row["gender"]
            occupation = row["occupation"]
            generated = row["generated"]
            try:
                generated = generated.split('Interview Script:')[1].replace('<return>', '\n')
            except (KeyError,IndexError):
                try:
                    generated = generated.split('Interview Transcript:')[1].replace('<return>', '\n')
                except (KeyError,IndexError):
                    generated = row["generated"]

            for question in questions:
                if (generated, question) in existing_evaluation:
                    continue  # Skip already evaluated

                prompt = f"{question}\n Script: {generated}\n"
                try:
                    answer = generator(prompt)
                except Exception as e:
                    print(f"Error during generation: {e}. Skipping and continuing...")
                    continue

                new_row = {
                    "country": country,
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "occupation": occupation,
                    "scripts": generated,
                    "question": question,
                    "answer": answer
                }

                pd.DataFrame([new_row]).to_csv(f, mode='a', header=write_header, index=False)
                write_header = False 
                existing_evaluation.add((generated, question))

    print(f"Evaluation completed. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="chatgpt", help="evaluating model")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="input file path")
    parser.add_argument("-of", "--output_file_name", type=str, default="evaluation_results.csv")
    parser.add_argument("-o", "--output_path", type=str, default="./evaluation_results/chatgpt", help="output file path")
    args = parser.parse_args()

    if args.model == "chatgpt":
        generator = generate_chatgpt_original
    elif args.model == "o4-mini":
        generator = lambda utt: generate_chatgpt_original(utt, model="o4-mini")
    else:
        pipe = load_model(args.model)
        generator = generate_output(pipe)


    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: Input path '{input_file}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(input_file) or not input_file.lower().endswith(".csv"):
        print(f"Error: Input path '{input_file}' is not a valid CSV file.")
        sys.exit(1)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    evaluate_csv(args, input_file, output_path, args.output_file_name, questions, generator)

if __name__ == "__main__":
    main()