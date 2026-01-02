import json
import os
import argparse
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Configuration
# MODEL = "Qwen/Qwen3-4B-Instruct-2507"
load_dotenv()  # Load environment variables from .env file

API_BASE_URL = os.getenv("OLLAMA_BASE_URL")
API_KEY = os.getenv("OLLAMA_API_KEY")
DEFAULT_MAX_CONCURRENT = 20

INPUT_FILE = "bfcl_eval/data/BFCL_v4_memory.json"
OUTPUT_DIR = 'result/naive_llm/{}/'

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Use LLM to answer questions from a JSONL file.")
    parser.add_argument("--model", type=str, help="The model to use for generating answers.")
    return parser.parse_args()

def _parse_query_response_prompting(api_response) -> dict:
        model_response = api_response.choices[0].message.content

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

async def get_completion_async(client: AsyncOpenAI, question: str, model: str) -> str:
    """Calls the LLM to get an answer for a given question."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": f"For your final answer to the user, you must respond in this format: {{'answer': A short and precise answer to the question, 'context': A brief explanation of how you arrived at this answer or why it is correct}}. If you do not know the answer, respond with {{'answer': 'I do not know', 'context': 'I do not know'}}. If you think the question cannot be properly answered, response with {{'answer': 'I cannot answer this question', 'context': A short reason explaining why this question cannot be answered}}. Answer the following question: {question}"}
            ],
            temperature=0.1,
            extra_body={
                "num_ctx": 40960
            }
        )
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"Error: {e}"

async def main():
    """Main function to process the file and generate answers."""
    args = parse_arguments()

    output_dir = OUTPUT_DIR.format(args.model.replace('/', '_'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    output_filename = "BFCL_v4_memory_kv_result.json"
    output_filepath = os.path.join(output_dir, output_filename)
    # Read all lines first
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    print(f"Total lines to process: {len(lines)}")

    async def process_line(line):
        data = json.loads(line)
        question = data["question"][0][0]["content"]
        response = await get_completion_async(client, question, args.model)
        answer = _parse_query_response_prompting(response)
        return {"id": data["id"].replace('memory_', 'memory_kv_'), "result": [[answer["model_responses"]]]}

    # Process in batches to limit concurrent requests and memory usage
    results = []
    for i in range(0, len(lines), DEFAULT_MAX_CONCURRENT):
        batch = lines[i:i+DEFAULT_MAX_CONCURRENT]
        tasks = [asyncio.create_task(process_line(line)) for line in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"Processed {min(i + DEFAULT_MAX_CONCURRENT, len(lines))} / {len(lines)} lines")

    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        for result in results:
            f_out.write(json.dumps(result) + '\n')

    print(f"Processing complete. Results saved to {output_filepath}")

if __name__ == "__main__":
    asyncio.run(main())
