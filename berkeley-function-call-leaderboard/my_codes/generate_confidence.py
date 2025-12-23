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

from bfcl_eval.constants.default_prompts import CONFIDENCE_SCORE_TOPK, CONFIDENCE_SCORE_result
from bfcl_eval.utils import load_file
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configuration
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
API_BASE_URL = os.getenv("LOCAL_SERVER_ENDPOINT", "http://localhost:5678/v1")
API_KEY = os.getenv("LOCAL_SERVER_API_KEY", "EMPTY")
CONTEXT_LENGTH = 32768
MAX_TOKENS = 4096
MAX_CONCURRENT_REQUESTS = 30  # TODO: Adjust this based on your GPU memory (VRAM)

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

def setup_clients() -> Tuple[AsyncOpenAI, PreTrainedTokenizer]:
    """Initializes AsyncOpenAI client and Tokenizer."""
    logging.info(f"Initializing AsyncOpenAI client: {API_BASE_URL}")
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    logging.info(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    return client, tokenizer

def parse_chatml_blocks(text: str) -> List[Dict[str, str]]:
    pattern = r"<\|im_start\|>(.*?)\n(.*?<\|im_end\|>)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"role": r.strip(), "content": c.strip()} for r, c in matches]

def parse_core_memory_block(text: str) -> dict:
    marker = "Here is the content of your Core Memory from previous interactions:"
    start = text.find(marker)
    if start == -1:
        raise ValueError("Core Memory marker not found")

    # Find first '{' after marker
    brace_start = text.find("{", start)
    if brace_start == -1:
        raise ValueError("Opening '{' not found")

    # Brace matching to find the correct closing '}'
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

    json_str = text[brace_start: brace_end + 1]

    # Parse JSON
    return json.loads(json_str)

def format_prompt_for_model(messages: List[Dict[str, str]]) -> str:
    formatted = ""
    for m in messages:
        formatted += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted

async def get_completion_async(client: AsyncOpenAI, prompt: str, max_tokens: int) -> Optional[str]:
    """Async version of completion call."""
    try:
        logging.debug(f"[LLM INPUT] Prompt:\n{prompt}")
        response = await client.completions.create(
            model=MODEL,
            temperature=0.1,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        output = response.choices[0].text
        logging.debug(f"[LLM OUTPUT] Response:\n{output}")
        return output
    except (APIConnectionError, RateLimitError) as e:
        logging.warning(f"Network/Rate limit error (retrying might be needed): {e}")
        return None
    except Exception as e:
        logging.error(f"Error calling API: {e}")
        return None

async def process_prereq_line(
    line: str,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    tokenizer: PreTrainedTokenizer,
    functions: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Process a single line from the log file.
    Uses Semaphore to limit concurrency.
    """
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    scores = []
    
    async with sem:
        for inference_logs in data.get('inference_log', []):
            for key, value in inference_logs.items():
                if key != 'step_0': continue # Only process step_0 for confidence score
                
                for step_log in value:
                    if step_log.get('role') != "inference_input": continue
                    
                    content = step_log.get('content', {}).get('formatted_prompt', '')
                    if not content:
                        raise ValueError(f"\33[31mNo content found in formatted_prompt, id: {data.get('id', '')}\33[0m")
                    
                    blocks = parse_chatml_blocks(content)
                    if not blocks:
                        raise ValueError(f"\33[31mFailed to parse ChatML blocks, id: {data.get('id', '')}\33[0m")

                    user_prompt = blocks[-1]['content']
                    formatted_functions = json.dumps(functions, indent=2, ensure_ascii=False)
                    system_prompt = CONFIDENCE_SCORE_TOPK.format(top_k=3, functions=formatted_functions)
                    
                    prompt_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    
                    formatted_prompt = format_prompt_for_model(prompt_messages)
                    input_token_count = len(tokenizer.tokenize(formatted_prompt))
                    available_tokens = CONTEXT_LENGTH - input_token_count - 2
                    
                    if available_tokens <= 0:
                        continue
                        
                    tokens_to_generate = min(MAX_TOKENS, available_tokens)
                    
                    # Await the async API call
                    resp = await get_completion_async(client, formatted_prompt, tokens_to_generate)
                    if resp:
                        scores.append(resp.strip())

    if scores:
        return {"id": data.get("id", ""), "scores": scores}
    return None

async def process_result_line(
    line: str,
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    tokenizer: PreTrainedTokenizer,
    functions: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None

    scores = []
    
    async with sem:
        for inference_logs in data.get('inference_log', []):
            for key, value in inference_logs.items():
                if key != 'step_0': continue # Only process step_0 for confidence score
                
                for step_log in value:
                    if step_log.get('role') != "inference_input": continue
                    
                    content = step_log.get('content', {}).get('formatted_prompt', '')
                    if not content:
                        raise ValueError(f"\33[31mNo content found in formatted_prompt, id: {data.get('id', '')}\33[0m")
                    
                    blocks = parse_chatml_blocks(content)
                    core_memory = parse_core_memory_block(blocks[0]['content'])
                    if not blocks:
                        raise ValueError(f"\33[31mFailed to parse ChatML blocks, id: {data.get('id', '')}\33[0m")

                    user_prompt = blocks[-1]['content']
                    formatted_functions = json.dumps(functions, indent=2, ensure_ascii=False)
                    system_prompt = CONFIDENCE_SCORE_result.format(functions=formatted_functions, core_memory=json.dumps(core_memory, ensure_ascii=False, indent=2))
                    
                    prompt_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    
                    formatted_prompt = format_prompt_for_model(prompt_messages)
                    input_token_count = len(tokenizer.tokenize(formatted_prompt))
                    available_tokens = CONTEXT_LENGTH - input_token_count - 2
                    
                    if available_tokens <= 0:
                        continue
                        
                    tokens_to_generate = min(MAX_TOKENS, available_tokens)
                    
                    # Await the async API call
                    resp = await get_completion_async(client, formatted_prompt, tokens_to_generate)
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

    client, tokenizer = setup_clients()
    try:
        functions = load_file(args.functions_file)
    except Exception as e:
        logging.error(f"Failed to load functions: {e}")
        return

    # Semaphore limits how many tasks run AT THE SAME TIME
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks = []
    logging.info(f"Reading input directory: {args.input_dir}")
    prereq_file = args.input_dir + "BFCL_v4_memory_kv_prereq_result.json"
    result_file = args.input_dir + "BFCL_v4_memory_kv_result.json"
    output_preq = args.output_dir + "prereq_confidence.json"
    output_result = args.output_dir + "result_confidence.json"
    
    # # Create tasks (Prepare the work)
    # with open(prereq_file, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
        
    # for line in lines:
    #     task = asyncio.create_task(
    #         process_prereq_line(line, sem, client, tokenizer, functions)
    #     )
    #     tasks.append(task)
    
    # logging.info(f"Total tasks created: {len(tasks)}. Starting execution...")

    # with open(output_preq, 'w', encoding='utf-8') as f_out:
    #     for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
    #         result = await completed_task
    #         if result:
    #             json.dump(result, f_out, ensure_ascii=False)
    #             f_out.write("\n")
    #             f_out.flush() # Ensure it's written to disk

    # Create tasks (Prepare the work)
    with open(result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        task = asyncio.create_task(
            process_result_line(line, sem, client, tokenizer, functions)
        )
        tasks.append(task)
    
    logging.info(f"Total tasks created: {len(tasks)}. Starting execution...")

    with open(output_result, 'w', encoding='utf-8') as f_out:
        for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            result = await completed_task
            if result:
                json.dump(result, f_out, ensure_ascii=False)
                f_out.write("\n")
                f_out.flush() # Ensure it's written to disk
                
    logging.info("Processing complete.")

if __name__ == '__main__':
    asyncio.run(main_async())