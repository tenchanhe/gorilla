# -*- coding: utf-8 -*-
"""
Optimized Async Script to calculate confidence scores.
Uses asyncio for parallel processing to maximize GPU throughput.
"""
import json
import re
import argparse
import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from tqdm.asyncio import tqdm
from bfcl_eval.constants.default_prompts import GROUNDTRUTH_EXTRACTION_PROMPT
from bfcl_eval.utils import load_file
import ollama

# Configuration (default values, can be overridden by args)
MODEL = "qwen2.5:32b"
OLLAMA_URL = "http://localhost:11434"
CONTEXT_LENGTH = 32768
MAX_TOKENS = 4096
MAX_CONCURRENT_REQUESTS = 30
ANSWER_PATH = "bfcl_eval/data/possible_answer/BFCL_v4_memory.json"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async confidence score calculator.")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input directory containing JSONL files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory for JSON files.")
    parser.add_argument("--functions-file", type=str, required=True, help="Path to functions JSON.")
    parser.add_argument("--log-file", type=str, default="generate_confidence.log")
    parser.add_argument("--console-level", type=str, default="INFO")
    return parser.parse_args()

def setup_logging(log_file: str, console_level: str) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(fmt)
    root.handlers = [fh, ch]

def parse_chatml_blocks(text: str) -> List[Dict[str, str]]:
    pattern = r"<\|im_start\|>(.*?)\n(.*?<\|im_end\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"role": r.strip(), "content": c.strip()} for r, c in matches]

def parse_core_memory_block(text: str) -> dict:
    marker = "Here is the content of your Core Memory from previous interactions:"
    start = text.find(marker)
    if start == -1:
        raise ValueError("Core Memory marker not found")
    brace_start = text.find("{", start)
    if brace_start == -1:
        raise ValueError("Opening '{' not found")
    brace_count = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                brace_end = i
                break
    else:
        raise ValueError("Matching '}' not found")
    json_str = text[brace_start : brace_end + 1]
    return json.loads(json_str)

async def get_completion_async(client: ollama.AsyncClient, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
    """Async call to Ollama API for completion."""
    try:
        logging.debug(f"[LLM INPUT] Messages (truncated to 500 chars):\n{str(messages)[:500]}{'...' if len(str(messages)) > 500 else ''}")
        response = await client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": 0.1,
                "context_length": CONTEXT_LENGTH,
            },
        )
        output = response["message"]["content"]
        logging.debug(f"[LLM OUTPUT] Response (truncated to 500 chars):\n{output[:500]}{'...' if len(output) > 500 else ''}")
        return output
    except Exception as e:
        logging.error(f"Error calling Ollama API: {e}")
        return None

async def process_groundtruth_line(
    line: str,
    sem: asyncio.Semaphore,
    client: ollama.AsyncClient,
    model: str,
    functions: List[Dict[str, Any]],
    answers: List[Dict],
) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    
    scores = []
    doc_id = data.get("id", "").replace("_kv", "")
    answer_entry = next((ans for ans in answers if ans["id"] == doc_id), None)
    
    if not answer_entry:
        raise ValueError(f"\33[31mNo answer entry found for id: {data.get('id', '')}\33[0m")
    
    async with sem:
        for inference_logs in data.get("inference_log", []):
            for key, value in inference_logs.items():
                if key != "step_0":
                    continue
                for step_log in value:
                    if step_log.get("role") != "inference_input":
                        continue
                    content = step_log.get("content", {}).get("formatted_prompt", "")
                    if not content:
                        raise ValueError(f"\33[31mNo content found in formatted_prompt, id: {data.get('id', '')}\33[0m")
                    blocks = parse_chatml_blocks(content)
                    core_memory = parse_core_memory_block(blocks[0]["content"])
                    if not blocks:
                        raise ValueError(f"\33[31mFailed to parse ChatML blocks, id: {data.get('id', '')}\33[0m")
                    user_prompt = blocks[-1]["content"]
                    system_prompt = GROUNDTRUTH_EXTRACTION_PROMPT.format(
                        user_query=user_prompt,
                        answer=answer_entry["ground_truth"],
                        source=answer_entry["source"],
                        core_memory=json.dumps(core_memory, ensure_ascii=False, indent=2),
                    )
                    prompt_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    resp = await get_completion_async(client, model, prompt_messages)
                    if resp:
                        scores.append(resp.strip())
    if scores:
        return {"id": data.get("id", ""), "scores": scores}
    return None

async def main_async():
    args = parse_arguments()
    if os.path.exists(args.output_dir) and any(os.scandir(args.output_dir)):
        logging.warning(f"Output directory {args.output_dir} exists and is not empty. Skipping.")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_file, args.console_level)
    try:
        functions = load_file(args.functions_file)
    except Exception as e:
        logging.error(f"Failed to load functions: {e}")
        return
    client = ollama.AsyncClient(host=OLLAMA_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    logging.info(f"Reading input directory: {args.input_dir}")
    result_file = args.input_dir + "BFCL_v4_memory_kv_result.json"
    output_result = args.output_dir + "ground_truth.json"
    answers = load_file(ANSWER_PATH)
    with open(result_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        task = asyncio.create_task(
            process_groundtruth_line(line, sem, client, MODEL, functions, answers)
        )
        tasks.append(task)
    logging.info(f"Total tasks created: {len(tasks)}. Starting execution...")
    with open(output_result, "w", encoding="utf-8") as f_out:
        for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            result = await completed_task
            if result:
                json.dump(result, f_out, ensure_ascii=False)
                f_out.write("\n")
                f_out.flush()
    logging.info("Processing complete.")

if __name__ == "__main__":
    asyncio.run(main_async())
