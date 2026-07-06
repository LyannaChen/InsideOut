"""Generate interview scripts for the InsideOut reproduction package.

Pipeline names:
- basic: original interview prompt only, matching generate_docs.py.
- fip: fairness pillars included in the initial generation prompt.
- mfa-sa: old single-agent one-shot self-refinement pipeline.
- mfa-ha: old hierarchical generator/critic/refiner pipeline.
- mfa-plan: new iterative single-agent review/rewrite pipeline.
"""

import argparse
import ast
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from InsideOut.data.prompts import (
    AGENTIC_ACTION_FORMAT,
    AGENTIC_REVIEW_PROMPT,
    FAIR_INTERVIEW_PILLARS,
    HIERARCHICAL_CRITIQUE_PROMPT,
    HIERARCHICAL_PLANNING_PROMPT,
    HIERARCHICAL_REFINEMENT_PROMPT,
    REFINEMENT_PROMPT,
    iter_prompt_records,
)

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

ACTION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_action",
            "description": "Submit the decision for rewrite or end with the critique, change plan, and updated script.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["rewrite", "end"]},
                    "reason": {"type": "string"},
                    "critique": {"type": "string"},
                    "change_plan": {"type": "string"},
                    "script": {"type": "string"},
                },
                "required": ["action", "reason", "critique", "change_plan", "script"],
                "additionalProperties": False,
            },
        },
    }
]

TOOL_INSTRUCTION = "You must call the submit_action tool. Do not respond with plain text.\n"


def normalize_newlines(text):
    return (text or "").replace("\n", "<return>")


def denormalize_newlines(text):
    return (text or "").replace("<return>", "\n")


def make_cache_key(template_id, country, name, gender, age, occupation):
    return "|".join(map(str, [template_id, country, name, gender, age, occupation]))


def load_selection_keys(path):
    if not path:
        return None
    df = pd.read_csv(path)
    required = ["template_id", "country", "name", "gender", "age", "occupation"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Selection file is missing columns: {missing}")
    return {
        make_cache_key(row.template_id, row.country, row.name, row.gender, row.age, row.occupation)
        for row in df.itertuples(index=False)
    }


def load_initial_cache(path):
    cache = {"by_prompt": {}, "by_key": {}}
    if not path or not os.path.exists(path):
        return cache
    df = pd.read_csv(path)
    script_col = "generated" if "generated" in df.columns else "final_script" if "final_script" in df.columns else None
    if not script_col or "prompts" not in df.columns:
        raise ValueError("Initial-cache CSV must include 'prompts' and either 'generated' or 'final_script'.")
    cache["by_prompt"] = dict(zip(df["prompts"], df[script_col].astype(str)))
    required = ["template_id", "country", "name", "gender", "age", "occupation"]
    if all(col in df.columns for col in required):
        for row in df.itertuples(index=False):
            cache["by_key"][make_cache_key(row.template_id, row.country, row.name, row.gender, row.age, row.occupation)] = str(getattr(row, script_col))
    return cache


def load_generator(model_name, batch_size, openai_api_base=None, openai_api_key=None, served_model_name=None):
    model_key = model_name.lower()
    if model_key in {"chatgpt", "openai"}:
        from openai import OpenAI

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or pass --openai_api_key for ChatGPT/OpenAI generation.")
        client = OpenAI(api_key=api_key)
        served_name = served_model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-2024-05-13")

        def generate(prompts, max_tokens=4096):
            outputs = []
            for prompt in prompts:
                response = client.chat.completions.create(
                    model=served_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )
                outputs.append((response.choices[0].message.content or "").strip())
            return outputs

        def generate_with_tools(prompts, tools, tool_model=None, max_tokens=4096):
            outputs = []
            for prompt in prompts:
                response = client.chat.completions.create(
                    model=tool_model or served_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    tools=tools,
                    tool_choice="required",
                    max_tokens=max_tokens,
                )
                message = response.choices[0].message
                tool_calls = []
                for call in message.tool_calls or []:
                    args = call.function.arguments
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        parsed = {"raw": args}
                    tool_calls.append({"name": call.function.name, "arguments": parsed})
                outputs.append({"content": message.content or "", "tool_calls": tool_calls})
            return outputs

        return generate, generate_with_tools, 1

    if openai_api_base:
        from openai import OpenAI

        client = OpenAI(
            api_key=openai_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=openai_api_base,
        )
        served_name = served_model_name or model_name

        def generate(prompts, max_tokens=4096):
            outputs = []
            for prompt in prompts:
                response = client.chat.completions.create(
                    model=served_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                outputs.append((response.choices[0].message.content or "").strip())
            return outputs

        def generate_with_tools(prompts, tools, tool_model=None, max_tokens=4096):
            outputs = []
            for prompt in prompts:
                response = client.chat.completions.create(
                    model=tool_model or served_name,
                    messages=[{"role": "user", "content": prompt}],
                    tools=tools,
                    tool_choice="required",
                    max_tokens=max_tokens,
                )
                message = response.choices[0].message
                tool_calls = []
                for call in message.tool_calls or []:
                    args = call.function.arguments
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        parsed = {"raw": args}
                    tool_calls.append({"name": call.function.name, "arguments": parsed})
                outputs.append({"content": message.content or "", "tool_calls": tool_calls})
            return outputs

        return generate, generate_with_tools, batch_size

    import torch
    from transformers import pipeline

    model_ids = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
    }
    if model_key not in model_ids:
        raise ValueError(f"Unsupported model: {model_name}")
    pipe = pipeline(
        "text-generation",
        model=model_ids[model_key],
        torch_dtype=torch.bfloat16 if model_key == "llama" else "auto",
        device_map="auto",
    )
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    def generate(prompts, max_tokens=4096):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outputs = pipe(
            messages,
            max_new_tokens=max_tokens,
            repetition_penalty=1.5,
            top_p=0.75,
            num_beams=2,
            batch_size=batch_size,
        )
        return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]

    return generate, None, batch_size


def parse_action_response(text):
    action_match = re.search(r"(?im)^Action:\s*([^\r\n]+)", text or "")
    reason_match = re.search(r"(?im)^Reason:\s*([^\r\n]+)", text or "")
    critique_match = re.search(r"(?im)^Critique:\s*([^\r\n]+(?:\n(?!Change Plan:|Script:).+)*)", text or "")
    change_plan_match = re.search(r"(?im)^Change Plan:\s*([^\r\n]+(?:\n(?!Script:).+)*)", text or "")
    script = text.split("Script:", 1)[1].strip() if text and "Script:" in text else (text or "").strip()
    return (
        action_match.group(1).strip().lower() if action_match else "",
        reason_match.group(1).strip() if reason_match else "",
        critique_match.group(1).strip() if critique_match else "",
        change_plan_match.group(1).strip() if change_plan_match else "",
        script,
    )


def parse_action_from_tool_calls(tool_calls):
    for call in tool_calls or []:
        if call.get("name") != "submit_action":
            continue
        args = call.get("arguments", {}) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = ast.literal_eval(args)
        return (
            str(args.get("action", "")).strip().lower(),
            str(args.get("reason", "")).strip(),
            str(args.get("critique", "")).strip(),
            str(args.get("change_plan", "")).strip(),
            str(args.get("script", "")).strip(),
        )
    return "", "", "", "", ""


def has_meaningful_change(old_script, new_script, min_delta_chars=40, max_similarity=0.995):
    old_norm = re.sub(r"\s+", " ", denormalize_newlines(old_script)).strip()
    new_norm = re.sub(r"\s+", " ", denormalize_newlines(new_script)).strip()
    if old_norm == new_norm:
        return False
    return SequenceMatcher(None, old_norm, new_norm).ratio() < max_similarity or abs(len(old_norm) - len(new_norm)) >= min_delta_chars


def has_unresolved_issues(critique, change_plan):
    combined = f"{critique or ''} {change_plan or ''}".strip().lower()
    if not combined or combined in {"none none", "none"}:
        return False
    clear_markers = ["no issues remain", "no changes are necessary", "no changes needed", "adheres to all"]
    if any(marker in combined for marker in clear_markers):
        return False
    issue_markers = ["issue", "concern", "overgeneral", "assum", "generic", "should", "improve", "revise", "rewrite", "clarify"]
    return any(marker in combined for marker in issue_markers)


def run_mfa_plan(prompt, initial_script, generator, tool_generator, max_rounds, use_tool_calls, tool_model):
    trace = []
    current_script = denormalize_newlines(initial_script)
    final_action = "max_rounds"
    followup = ""
    for round_idx in range(1, max_rounds + 1):
        review_prompt = AGENTIC_REVIEW_PROMPT.format(
            prompt=prompt,
            pillars=FAIR_INTERVIEW_PILLARS,
            script=current_script,
            action_format=AGENTIC_ACTION_FORMAT,
        )
        if followup:
            review_prompt += "\nAdditional instruction for this round:\n" + followup + "\n"
        if use_tool_calls:
            if tool_generator is None:
                raise RuntimeError("Tool calls require ChatGPT/OpenAI-compatible generation.")
            response = tool_generator([TOOL_INSTRUCTION + review_prompt], ACTION_TOOL, tool_model=tool_model)[0]
            action, reason, critique, change_plan, revised_script = parse_action_from_tool_calls(response.get("tool_calls"))
            if not action:
                action, reason, critique, change_plan, revised_script = parse_action_response(response.get("content", ""))
        else:
            response = generator([review_prompt])[0]
            action, reason, critique, change_plan, revised_script = parse_action_response(response)

        if action == "rewrite":
            if revised_script and has_meaningful_change(current_script, revised_script):
                current_script = revised_script
                followup = ""
            else:
                followup = "Your previous response chose rewrite but did not materially change the script. Either end with no remaining issues or revise substantively."
        elif action == "end" and round_idx < max_rounds and has_unresolved_issues(critique, change_plan):
            action = "rewrite"
            followup = "You chose end while still describing unresolved issues. Revise the script now, or choose end only if no substantive fairness issues remain."

        trace.append(
            {
                "round": round_idx,
                "action": action or "rewrite",
                "reason": reason,
                "critique": critique,
                "change_plan": change_plan,
                "script": normalize_newlines(current_script),
            }
        )
        if action == "end":
            final_action = "end"
            break
    return normalize_newlines(current_script), final_action, trace, len(trace)


def output_columns(pipeline):
    base = ["template_id", "country", "name", "gender", "age", "occupation", "prompts"]
    if pipeline in {"basic", "fip"}:
        return base + ["generated"]
    if pipeline in {"mfa-sa", "mfa-ha"}:
        return base + ["initial_script", "critique", "refined_script"]
    return base + ["initial_script", "final_script", "final_action", "rounds_used"]


def write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="chatgpt", help="Model: chatgpt/openai, llama, mistral, qwen, or an OpenAI-compatible served model")
    parser.add_argument("-o", "--output_path", default=None)
    parser.add_argument("-f", "--output_file_name", default=None)
    parser.add_argument("-p", "--pipeline", default="fip", choices=sorted(PIPELINE_ALIASES), help="Use basic, fip, mfa-sa, mfa-ha, or mfa-plan.")
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-c", "--clean_output", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("--selection_file", default=None, help="Optional CSV limiting generation to selected metadata rows.")
    parser.add_argument("--initial_cache", default=None, help="Optional CSV of baseline scripts to reuse.")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--use_tool_calls", action="store_true")
    parser.add_argument("--tool_model", default=None)
    parser.add_argument("--openai_api_base", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--served_model_name", default=None)
    args = parser.parse_args()

    pipeline = PIPELINE_ALIASES[args.pipeline]
    prompt_mode = "basic" if pipeline == "basic" else "fip"
    output_path = Path(args.output_path or Path("outputs") / pipeline / args.model)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_name = args.output_file_name or f"generated_scripts_{pipeline}_{args.model}.csv"
    csv_path = output_path / output_file_name
    trace_path = output_path / f"{Path(output_file_name).stem}_trace.jsonl"

    generator, tool_generator, effective_batch_size = load_generator(
        args.model,
        args.batch_size,
        openai_api_base=args.openai_api_base,
        openai_api_key=args.openai_api_key,
        served_model_name=args.served_model_name,
    )
    args.batch_size = effective_batch_size

    if args.clean_output:
        csv_path.unlink(missing_ok=True)
        trace_path.unlink(missing_ok=True)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_keys = {
            make_cache_key(row.template_id, row.country, row.name, row.gender, row.age, row.occupation)
            for row in existing_df.itertuples(index=False)
        }
        write_header = False
    else:
        existing_keys = set()
        pd.DataFrame(columns=output_columns(pipeline)).to_csv(csv_path, index=False)
        write_header = False

    selection_keys = load_selection_keys(args.selection_file)
    cache = load_initial_cache(args.initial_cache)
    records = []
    for record in iter_prompt_records(prompt_mode):
        template_id, country, name, gender, age, occupation, prompt = record
        key = make_cache_key(template_id, country, name, gender, age, occupation)
        if selection_keys is not None and key not in selection_keys:
            continue
        if key in existing_keys:
            continue
        records.append(record)
    if args.test:
        records = records[:10]
    if not records:
        print("No new scripts to generate.")
        return

    try:
        for start in tqdm(range(0, len(records), args.batch_size), desc=f"Generating {pipeline}"):
            batch = records[start : start + args.batch_size]
            initial_scripts = []
            prompts_to_generate = []
            prompt_positions = []
            for pos, row in enumerate(batch):
                template_id, country, name, gender, age, occupation, prompt = row
                key = make_cache_key(template_id, country, name, gender, age, occupation)
                cached = cache["by_prompt"].get(prompt) or cache["by_key"].get(key)
                initial_scripts.append(cached)
                if cached is None:
                    prompts_to_generate.append(prompt)
                    prompt_positions.append(pos)

            if prompts_to_generate:
                if pipeline == "mfa-ha":
                    initial_prompts = [HIERARCHICAL_PLANNING_PROMPT.format(prompt, FAIR_INTERVIEW_PILLARS) for prompt in prompts_to_generate]
                else:
                    initial_prompts = prompts_to_generate
                generated = generator(initial_prompts)
                for pos, script in zip(prompt_positions, generated):
                    initial_scripts[pos] = normalize_newlines(script)

            if pipeline in {"basic", "fip"}:
                rows = []
                for row, script in zip(batch, initial_scripts):
                    template_id, country, name, gender, age, occupation, prompt = row
                    rows.append(
                        {
                            "template_id": template_id,
                            "country": country,
                            "name": name,
                            "gender": gender,
                            "age": age,
                            "occupation": occupation,
                            "prompts": prompt,
                            "generated": script,
                        }
                    )
                pd.DataFrame(rows).to_csv(csv_path, mode="a", header=write_header, index=False)
                continue

            if pipeline == "mfa-sa":
                refinement_prompts = [
                    REFINEMENT_PROMPT.format(FAIR_INTERVIEW_PILLARS, denormalize_newlines(script))
                    for script in initial_scripts
                ]
                refined_scripts = [normalize_newlines(output) for output in generator(refinement_prompts)]
                rows = []
                for row, initial, refined in zip(batch, initial_scripts, refined_scripts):
                    template_id, country, name, gender, age, occupation, prompt = row
                    rows.append(
                        {
                            "template_id": template_id,
                            "country": country,
                            "name": name,
                            "gender": gender,
                            "age": age,
                            "occupation": occupation,
                            "prompts": prompt,
                            "initial_script": initial,
                            "critique": None,
                            "refined_script": refined,
                        }
                    )
                pd.DataFrame(rows).to_csv(csv_path, mode="a", header=write_header, index=False)
                continue

            if pipeline == "mfa-ha":
                critique_prompts = [
                    HIERARCHICAL_CRITIQUE_PROMPT.format(FAIR_INTERVIEW_PILLARS, denormalize_newlines(script))
                    for script in initial_scripts
                ]
                critiques = [normalize_newlines(output) for output in generator(critique_prompts)]
                refinement_prompts = [
                    HIERARCHICAL_REFINEMENT_PROMPT.format(prompt, FAIR_INTERVIEW_PILLARS, denormalize_newlines(critique))
                    for (*_, prompt), critique in zip(batch, critiques)
                ]
                refined_scripts = [normalize_newlines(output) for output in generator(refinement_prompts)]
                rows = []
                for row, initial, critique, refined in zip(batch, initial_scripts, critiques, refined_scripts):
                    template_id, country, name, gender, age, occupation, prompt = row
                    rows.append(
                        {
                            "template_id": template_id,
                            "country": country,
                            "name": name,
                            "gender": gender,
                            "age": age,
                            "occupation": occupation,
                            "prompts": prompt,
                            "initial_script": initial,
                            "critique": critique,
                            "refined_script": refined,
                        }
                    )
                pd.DataFrame(rows).to_csv(csv_path, mode="a", header=write_header, index=False)
                continue

            for row, initial in zip(batch, initial_scripts):
                template_id, country, name, gender, age, occupation, prompt = row
                final_script, final_action, trace, rounds_used = run_mfa_plan(
                    prompt,
                    initial,
                    generator,
                    tool_generator,
                    args.max_rounds,
                    args.use_tool_calls,
                    args.tool_model,
                )
                out_row = {
                    "template_id": template_id,
                    "country": country,
                    "name": name,
                    "gender": gender,
                    "age": age,
                    "occupation": occupation,
                    "prompts": prompt,
                    "initial_script": initial,
                    "final_script": final_script,
                    "final_action": final_action,
                    "rounds_used": rounds_used,
                }
                pd.DataFrame([out_row]).to_csv(csv_path, mode="a", header=write_header, index=False)
                write_jsonl(trace_path, {**out_row, "agentic_trace": trace})
    except Exception as exc:
        print(f"An error occurred: {exc}. Progress saved to {csv_path}.", file=sys.stderr)
        raise

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
