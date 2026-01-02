import json
import os
import bm25s
import asyncio
import argparse
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Global Variables ---
# These will be initialized in the main function based on arguments.
client: AsyncOpenAI

load_dotenv()  # Load environment variables from .env file

API_BASE_URL = os.getenv("OLLAMA_BASE_URL")
API_KEY = os.getenv("OLLAMA_API_KEY")
# Path definitions
QUESTIONS_FILE = 'bfcl_eval/data/BFCL_v4_memory.json'
CONTEXT_DIR = 'bfcl_eval/data/memory_prereq_conversation/'
OUTPUT_DIR = 'result/simple_rag/{}/'

# --- Helper Functions ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run RAG with a selectable LLM provider.")
    parser.add_argument(
        "--model",
        type=str,
        help="The name of the model to use."
    )
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

def initialize_llm_client(model: str):
    """Initializes the AsyncOpenAI client based on the selected provider."""
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
    return client

def load_questions(file_path):
    """Loads questions from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_context_data(scenario):
    """Loads and extracts content from a scenario-specific JSONL file."""
    file_path = os.path.join(CONTEXT_DIR, f'memory_{scenario}.json')
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Extract text content from the nested structure
            text_content = " ".join([
                turn['content']
                for conv in item.get('question', [])
                for turn in conv
            ])
            if text_content:
                documents.append(text_content)
    return documents

async def get_llm_answer(client, model, question, context):
    """Generates an answer using the OpenAI LLM, including retrieved context."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": f"Based on the following context, please answer the question. For your final answer to the user, you must respond in this format: {{'answer': A short and precise answer to the question, 'context': A brief explanation of how you arrived at this answer or why it is correct}}. If you do not know the answer, respond with {{'answer': 'I do not know', 'context': 'I do not know'}}. If you think the question cannot be properly answered, response with {{'answer': 'I cannot answer this question', 'context': A short reason explaining why this question cannot be answered}}. Answer the following question: {question}"}
            ],
            temperature=0.1,
            extra_body={
                "num_ctx": 40960
            }
        )
        return response
    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return "Error generating answer."

# --- Main Logic ---

async def main():
    """Main function to run the RAG process."""
    args = parse_arguments()
    
    client = initialize_llm_client(args.model)

    output_dir = OUTPUT_DIR.format(args.model.replace('/', '_'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    questions_data = load_questions(QUESTIONS_FILE)
    
    # Group questions by scenario to process them in batches
    scenarios = {}
    for q in questions_data:
        scenario = q.get('scenario')
        if scenario:
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(q)

    all_results = []
    for scenario, questions in scenarios.items():
        print(f"Processing scenario: {scenario}...")
        
        # Load and index documents for the current scenario
        documents = load_context_data(scenario)
        if not documents:
            print(f"No documents found for scenario: {scenario}. Skipping.")
            continue
            
        # Tokenize the documents for BM25
        corpus_tokens = [doc.split() for doc in documents]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        for question_item in questions:
            # Extract the user's question content
            user_question = ""
            if question_item.get('question') and question_item['question'][0]:
                user_question = question_item['question'][0][0].get('content', '')

            if not user_question:
                continue

            # Retrieve the most relevant document
            query_tokens = user_question.split()
            results_bm25, _ = retriever.retrieve([query_tokens], corpus=documents, k=1)
            
            # The result from bm25s is a list of lists, we take the first one
            retrieved_context = results_bm25[0][0] if results_bm25 and results_bm25[0] else ""

            # Generate answer with LLM
            response = await get_llm_answer(client, args.model, user_question, retrieved_context)
            answer = _parse_query_response_prompting(response)
            
            # Format the output to match the specified structure
            output_item = {
                'id': question_item['id'].replace('memory_', 'memory_kv_'),
                'result': [[json.dumps(answer)]]
            }
            all_results.append(output_item)

    # Write all results to a single file
    output_filename = os.path.join(output_dir, 'BFCL_v4_memory_kv_result.json')
    with open(output_filename, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res) + '\n')
    print(f"All results saved to {output_filename}")


if __name__ == "__main__":
    asyncio.run(main())
