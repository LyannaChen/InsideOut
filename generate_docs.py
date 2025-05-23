import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import torch
import argparse
import random 
import os
from constants import * 
from generation_util import * 
from huggingface_hub import login
hf_access_token = YOUR_HUGGINGFACE_TOKEN
login(token = hf_access_token)


def load_model(model_name):
    if model_name == "llama":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype = torch.bfloat16,
            device_map="auto",
            batch_size=1,
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
        generated_text = outputs[0].get("generated_text", "")
        # print(generated_text)
        return outputs[0]["generated_text"][-1]["content"]
    return generate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, default = "chatgpt", help = "generating model")
    parser.add_argument("-o", "--output_path", type = str, default = None, help = "output file path")
    parser.add_argument("-t", "--test", action="store_true") # if test, only test generate 10 examples
    parser.add_argument("-c", "--clean_output", action = "store_true")
    args = parser.parse_args()

    if args.model == "chatgpt":
        generator = generate_chatgpt_original
    elif args.model == "gemini":
        generator = get_gemini_response
    else:
        pipe = load_model(args.model)
        generator = generate_output(pipe)

    if args.output_path is None:
        output_path = os.path.join("generated_scripts", args.model)
    else:
        output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    csv_path = os.path.join(output_path, "generated_scripts.csv")

    #load existing data if exists 
    if args.clean_output or not os.path.exists(csv_path):
        if os.path.exists(csv_path): 
            os.remove(csv_path)
        existing_df = pd.DataFrame(columns = ['country', 'name', 'gender', 'age', 'occupation', 'prompts', 'generated'])
        existing_prompts = set()
    else:
        existing_df = pd.read_csv(csv_path)
        existing_prompts = set(existing_df['prompts'].tolist())
        print(f"Existing prompts loaded. Number of existing prompts: {len(existing_prompts)}")

    scripts = []
    # format: culture name, culture name, interviewee name, age, gender, occupation
    for template in TEMPLATES:
        for country, culture in CULTURES.items():
            for name in FEMALE_NAMES[country]:
                for age in AGES:
                    for occupation in OCCUPATIONS:
                        scripts.append(
                            (country, name, age, 'female', occupation, 
                             template.format(culture, country, name, age, 'female', occupation).strip())
                        )
            
            for name in MALE_NAMES[country]:
                for age in AGES:
                    for occupation in OCCUPATIONS:
                        scripts.append(
                            (country, name, age, 'male', occupation, 
                             template.format(culture, country, name, age, 'male', occupation).strip())
                        )
  
    if args.test:
        random.shuffle(scripts)
        scripts = scripts[:10]
    
    scripts_to_generate = [entry for entry in scripts if entry[5] not in existing_prompts]
    
    if not scripts_to_generate:
        print("No new scripts to generate.")
        return

    print('Number of new scripts to be generated:', len(scripts_to_generate))

    write_header = not os.path.exists(csv_path)
    
    try:
      for country, name, age, gender, occupation, prompt in tqdm(scripts_to_generate):
        generated_response = generator(prompt)
        generated_response = generated_response.replace('\n', '<return>')

        new_row = {
            'country': country,
            'name': name,
            'age': age,
            'gender': gender,
            'occupation': occupation,
            'prompts': prompt,
            'generated': generated_response
        }

        pd.DataFrame([new_row]).to_csv(csv_path, mode='a', header=write_header, index=False)
        write_header = False 
    except Exception as e:
        print(f"An error occurred: {e}. Progress saved.")
        
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()