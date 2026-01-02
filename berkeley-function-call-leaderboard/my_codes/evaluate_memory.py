
import argparse
import json
import re
import os
from pathlib import Path

from bfcl_eval.utils import load_file, write_list_of_dicts_to_file
from bfcl_eval.eval_checker.eval_runner_helper import get_directory_structure_by_category
from bfcl_eval.constants.eval_config import PROJECT_ROOT, SCORE_PATH
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response
from bfcl_eval.model_handler.utils import default_decode_execute_prompting


def standardize_string(input_string: str):
    """
    Standardizes the string by removing specific punctuation, converting to lowercase,
    and replacing single quotes with double quotes.
    """
    regex_string = r"[\,\.\/\-\_\*\^\(\)]"  # 移除 , . / - _ * ^ ( )
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def agentic_checker(model_response: str, possible_answer_list: list[str]) -> dict:
    """
    Check if one of the possible answers is contained in the model response, ignoring case, whitespace and ",./-_*^" punctuation.
    """
    standardized_possible_answer_list = [
        standardize_string(possible_answer) for possible_answer in possible_answer_list
    ]
    # Sometimes the model response is a list of one string
    if type(model_response) is list:
        model_response = model_response[0]
    if type(model_response) is not str:
        model_response = str(model_response)

    standardized_model_response = standardize_string(model_response)

    for possible_answer in standardized_possible_answer_list:
        if re.search(rf"\b{re.escape(possible_answer)}\b", standardized_model_response):
            return {"valid": True, "error": []}

    return {
        "valid": False,
        "error_message": f"None of the expected answers were found in the model response.",
        "error_type": "agentic:answer_not_found",
        "details": {
            "model_response": model_response,
            "possible_answers": possible_answer_list,
            "standardized_model_response": standardized_model_response,
            "standardized_possible_answers": standardized_possible_answer_list,
        },
    }


def evaluate_memory_entry(model_result_entry: dict, ground_truth_entry: dict, prompt_entry: dict):
    """Evaluates a single memory (agentic) entry."""
    test_id = model_result_entry["id"]
    model_result_list = model_result_entry["result"]
    possible_answer_item = ground_truth_entry["ground_truth"]
    model_name = model_result_entry.get("model_name", "unknown_model")
    test_category = ground_truth_entry.get("category", "memory")

    if not isinstance(model_result_list, list) or len(model_result_list) != 1:
        return {
            "id": test_id, "valid": False, "error": {"error_type": "agentic:inference_error", "error_message": "Model did not output a list of one conversation history."},
            "prompt": prompt_entry, "model_result": model_result_list, "possible_answer": possible_answer_item,
        }

    last_unsuccessful_decoding_message = None
    for model_turn in model_result_list[0]:
        try:
            decoded_result: list[str] = default_decode_execute_prompting(str(model_turn), has_tool_call_tag=False)
            if is_empty_execute_response(decoded_result):
                last_unsuccessful_decoding_message = model_turn
                continue
        except Exception:
            last_unsuccessful_decoding_message = model_turn
            continue
    
    if not last_unsuccessful_decoding_message:
        return {
            "id": test_id, "valid": False, "error": {"error_type": "agentic:no_last_message", "error_message": "Cannot find the last chat message that is not a function call."},
            "prompt": prompt_entry, "model_result": model_result_list, "possible_answer": possible_answer_item,
        }

    accuracy_checker_result = agentic_checker(last_unsuccessful_decoding_message, possible_answer_item)

    if not accuracy_checker_result["valid"]:
        return {
            "id": test_id, "model_name": model_name, "test_category": test_category, "valid": False,
            "error": accuracy_checker_result, "prompt": prompt_entry.get("question"),
            "model_result_raw": model_result_list, "last_non_fc_message": last_unsuccessful_decoding_message,
            "possible_answer": possible_answer_item,
        }

    return {
        "id": test_id,
        "model_name": model_name,
        "test_category": test_category,
        "valid": True,
        "prompt": prompt_entry.get("question"),
        "model_result_raw": model_result_list,
        "last_non_fc_message": last_unsuccessful_decoding_message,
        "possible_answer": possible_answer_item,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate memory-type test cases.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the model's result JSON file.")

    args = parser.parse_args()

    # Deduce test category from result file name if not specified as memory
    try:
        # e.g. BFCL_v4_memory_kv_result.json -> memory_kv
        test_category_from_file = Path(args.result_file).stem.split('BFCL_v4_')[1].replace('_result', '')
        if 'memory' in test_category_from_file:
                args.test_category = test_category_from_file
    except IndexError:
        print("Could not deduce test category from filename. Please use --test_category to specify.")
        return

    model_results = load_file(args.result_file, sort_by_id=True)
    
    # The prompt and ground truth for all memory tests are in the same two files.
    data_dir = PROJECT_ROOT / "bfcl_eval" / "data"
    ground_truth_file = data_dir / "possible_answer" / f"{VERSION_PREFIX}_memory.json"
    prompt_file = data_dir / f"{VERSION_PREFIX}_memory.json" # Prompts are in the base memory file

    if not ground_truth_file.exists() or not prompt_file.exists():
        print(f"Error: Could not find the general memory ground truth or prompt file.")
        print(f"Looked for: {ground_truth_file} and {prompt_file}")
        return

    # Load all ground truths and prompts and create a quick lookup dictionary
    all_ground_truths = load_file(ground_truth_file, sort_by_id=False) # sort is not needed
    all_prompts = load_file(prompt_file, sort_by_id=False)
    ground_truth_dict = {gt['id']: gt for gt in all_ground_truths}
    prompt_dict = {p['id']: p for p in all_prompts}
        
    correct_count = 0
    failed_results = []
    passed_results = []
    processed_count = 0

    for model_result in model_results:
        original_id = model_result["id"]
        # The result file IDs have a sub-category like '_kv_' that is not in the ground truth file IDs.
        # We need to remove it to match the ground truth.
        # e.g., "memory_kv_0-customer-0" -> "memory_0-customer-0"
        test_id = original_id.replace('_kv_', '_').replace('_vector_', '_').replace('_rec_sum_', '_')
        
        if test_id not in ground_truth_dict or test_id not in prompt_dict:
            print(f"Warning: Skipping ID '{original_id}' (transformed to '{test_id}') as it was not found in the ground truth/prompt file.")
            continue

        processed_count += 1
        ground_truth_entry = ground_truth_dict[test_id]
        prompt_entry = prompt_dict[test_id]

        eval_result = evaluate_memory_entry(model_result, ground_truth_entry, prompt_entry)

        if eval_result["valid"]:
            correct_count += 1
            passed_results.append(eval_result)
        else:
            # Add model_name for clarity in output
            failed_results.append(eval_result)

    total_count = processed_count
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\nEvaluation Complete:")
    print(f"  Correct: {correct_count}")
    print(f"  Total:   {total_count}")
    print(f"  Accuracy: {accuracy:.2%}")

    # Save results to a file for review
    if processed_count > 0:
        
        results = {
            "summary": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": total_count,
            },
            "passed_results": passed_results,
            "failed_results": failed_results,
        }

        output_path = args.result_file.replace("_result.json", "_evaluation.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved detailed report to: {output_path}")

if __name__ == "__main__":
    # Add version prefix for data files
    VERSION_PREFIX = "BFCL_v4"
    main()
