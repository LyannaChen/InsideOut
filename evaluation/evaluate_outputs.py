"""Evaluate generated interview scripts for cultural externality.

This is the cleaned InsideOut version of `retrieval/evaluate_outputs_new.py`.
It writes row-level evaluator answers; use `evaluate_results.py` to aggregate
the per-country yes percentage.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

EVALUATION_QUESTIONS = [
    "Does the interviewer use language that suggests they are not part of the interviewee's culture? Answer only in yes or no."
]

PIPELINE_ALIASES = {
    "basic": "basic",
    "fip": "fip",
    "mfa-sa": "mfa-sa",
    "single-agent": "mfa-sa",
    "mfa-ha": "mfa-ha",
    "hierarchical": "mfa-ha",
    "mfa-plan": "mfa-plan",
    "agentic-single": "mfa-plan",
}


def script_column_for_pipeline(pipeline):
    if pipeline in {"basic", "fip"}:
        return "generated"
    if pipeline in {"mfa-sa", "mfa-ha"}:
        return "refined_script"
    if pipeline == "mfa-plan":
        return "final_script"
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def load_generator(model_name, openai_api_base=None, openai_api_key=None, served_model_name=None, batch_size=4):
    model_key = model_name.lower()
    if model_key == "test":
        return lambda prompts: ["yes" for _ in prompts], 1

    if model_key in {"chatgpt", "openai", "o4-mini"} or openai_api_base:
        from openai import OpenAI

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY" if openai_api_base else None)
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or pass --openai_api_key for OpenAI evaluation.")
        client = OpenAI(api_key=api_key, base_url=openai_api_base) if openai_api_base else OpenAI(api_key=api_key)
        if served_model_name:
            eval_model = served_model_name
        elif model_key == "o4-mini":
            eval_model = "o4-mini-2025-04-16"
        else:
            eval_model = os.environ.get("OPENAI_EVAL_MODEL", "gpt-4o-2024-05-13")

        def generate(prompts):
            outputs = []
            for prompt in prompts:
                response = client.chat.completions.create(
                    model=eval_model,
                    messages=[
                        {"role": "system", "content": "You are a careful evaluator. Answer exactly yes or no."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=16,
                )
                outputs.append((response.choices[0].message.content or "").strip())
            return outputs

        return generate, 1

    import torch
    from transformers import pipeline

    model_ids = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
    }
    if model_key not in model_ids:
        raise ValueError(f"Unsupported evaluator model: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_ids[model_key],
        torch_dtype=torch.bfloat16 if model_key == "llama" else "auto",
        device_map="auto",
    )
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    def generate(prompts):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outputs = pipe(
            messages,
            max_new_tokens=16,
            repetition_penalty=1.5,
            temperature=0.1,
            top_p=0.75,
            num_beams=2,
            batch_size=batch_size,
        )
        return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]

    return generate, batch_size


def evaluate_csv(input_file, output_file, pipeline, generator, batch_size):
    df = pd.read_csv(input_file)
    script_column = script_column_for_pipeline(pipeline)
    if script_column not in df.columns:
        raise ValueError(
            f"Input file does not contain expected script column '{script_column}' for pipeline '{pipeline}'. "
            f"Available columns: {list(df.columns)}"
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "template_id",
        "country",
        "name",
        "gender",
        "age",
        "occupation",
        "prompts",
        "scripts",
        "question",
        "answer",
        "pipeline_type",
    ]
    if output_file.exists():
        existing_df = pd.read_csv(output_file)
        if "pipeline_type" not in existing_df.columns:
            existing_df["pipeline_type"] = pipeline
        existing = set(tuple(row) for row in existing_df[["scripts", "question", "pipeline_type"]].values.tolist())
        write_header = False
    else:
        pd.DataFrame(columns=columns).to_csv(output_file, index=False)
        existing = set()
        write_header = False

    pending = []
    for _, row in df.iterrows():
        script = row[script_column]
        for question in EVALUATION_QUESTIONS:
            key = (script, question, pipeline)
            if key not in existing:
                pending.append((row, script, question, key))

    print(f"Evaluating {len(pending)} items.")
    for start in tqdm(range(0, len(pending), batch_size), desc="Evaluating"):
        batch = pending[start : start + batch_size]
        prompts = [f"{question}\nScript: {script}\n" for _, script, question, _ in batch]
        answers = generator(prompts)
        rows = []
        for (row, script, question, key), answer in zip(batch, answers):
            rows.append(
                {
                    "template_id": row.get("template_id", None),
                    "country": row["country"],
                    "name": row["name"],
                    "gender": row["gender"],
                    "age": row["age"],
                    "occupation": row["occupation"],
                    "prompts": row["prompts"],
                    "scripts": script,
                    "question": question,
                    "answer": answer,
                    "pipeline_type": pipeline,
                }
            )
            existing.add(key)
        pd.DataFrame(rows).to_csv(output_file, mode="a", header=write_header, index=False)

    print(f"Evaluation completed. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="o4-mini", help="Evaluator model: o4-mini, chatgpt/openai, llama, mistral, qwen, or test.")
    parser.add_argument("-i", "--input_file", required=True, help="Generated script CSV to evaluate.")
    parser.add_argument("-of", "--output_file_name", default="evaluation_results.csv")
    parser.add_argument("-o", "--output_path", default="./InsideOut/evaluation_results", help="Directory for row-level evaluation CSV.")
    parser.add_argument("-p", "--pipeline", default="fip", choices=sorted(PIPELINE_ALIASES))
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("--openai_api_base", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--served_model_name", default=None)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.is_file() or input_file.suffix.lower() != ".csv":
        print(f"Error: input file '{input_file}' is not a CSV file.", file=sys.stderr)
        sys.exit(1)

    pipeline = PIPELINE_ALIASES[args.pipeline]
    generator, effective_batch_size = load_generator(
        args.model,
        openai_api_base=args.openai_api_base,
        openai_api_key=args.openai_api_key,
        served_model_name=args.served_model_name,
        batch_size=args.batch_size,
    )
    output_file = Path(args.output_path) / args.output_file_name
    evaluate_csv(input_file, output_file, pipeline, generator, effective_batch_size)


if __name__ == "__main__":
    main()
