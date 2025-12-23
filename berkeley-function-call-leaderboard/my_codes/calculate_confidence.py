import json
import re
import argparse
import os
from collections import Counter
from typing import Dict, List, Optional, Any

def parse_tool_call(s: str) -> Optional[Dict[str, Any]]:
    pattern = r'\[(\w+)\((.*?)\)\]'
    results = []
    # re.finditer 會回傳所有匹配的 iterator
    for match in re.finditer(pattern, s):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        args: Dict[str, str] = {}
        if args_str:
            # 這是您原本的參數解析邏輯，針對 key="value" 格式
            arg_pattern = r'(\w+)\s*=\s*"([^"]*)"'
            for key, value in re.findall(arg_pattern, args_str):
                args[key] = value
        
        results.append({"tool": tool_name, "args": args})
    return results

def process_result_data(data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    real_tool = []
    for turn in data:
        turn_tools = []
        for call in turn:
            call = parse_tool_call(call)
            if call:
                turn_tools.extend(call)
        real_tool.append(turn_tools)
    return real_tool

def process_confi_data(scores: List[str]) -> List[Dict[str, Any]]:
    real_tool = []
    for score in scores:
        turn_data = json.loads(score)
        real_tool.append(turn_data)
    return real_tool

def check_step(result_data: List[Dict[str, Any]], confi_data: List[Dict[str, Any]]) -> None:
    for result in result_data:
        for confi in confi_data:
            if result.get("id") == confi.get("id"):
                result_turns = result.get("result", [])
                confi_turns = confi.get("scores", [])
                if len(result_turns) != len(confi_turns):
                    print(f"Mismatch in step counts for ID {result.get('id')}: result has {len(result_turns)} steps, confidence score has {len(confi_turns)} steps.")
                    return False
    return True

def calculate_statistics(result_data, confi_data):
    stats = {
        "matched_tools": 0,
        "unmatched_tools": 0,
        "total_tools": 0,
        "not_matched": []
    }
    for result in result_data:
        for confi in confi_data:
            if result.get("id") == confi.get("id"):
                result_tools = process_result_data(result['result'])
                confi_tools = process_confi_data(confi['scores'])
                for idx, turn_tools in enumerate(result_tools):
                    for tool_call in turn_tools:
                        tool_name = tool_call['tool']
                        stats["total_tools"] += 1
                        if any(tool_name == v['tool'] for v in confi_tools[idx].values()):
                            stats["matched_tools"] += 1
                        else:
                            stats["unmatched_tools"] += 1
                            stats["not_matched"].append({
                                "id": result.get("id"),
                                "turn": idx,
                                "result_tool": tool_name,
                                "confi_tools": confi_tools[idx]
                            })
    return stats


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []
    
    data = []
    with open(file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line in {file_path}: {line.strip()}")
                continue
    return data

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate tool call statistics from model inference logs.")
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to the input directory with inference results.",
    )
    parser.add_argument(
        "--confi-dir",
        type=str,
        required=True,
        help="Path to the input directory with confidence scores.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Optional: Path to save the output statistics as a JSON file.",
    )
    return parser.parse_args()

def main():
    """
    Main function to read files, calculate statistics, and print/save the results.
    """
    args = parse_arguments()
    prereq_file = args.result_dir + "BFCL_v4_memory_kv_prereq_result.json"
    result_file = args.result_dir + "BFCL_v4_memory_kv_result.json"

    prereq_data = read_jsonl(prereq_file)
    result_data = read_jsonl(result_file)
    prereq_confi = read_jsonl(args.confi_dir + "prereq_confidence.json")
    result_confi = read_jsonl(args.confi_dir + "result_confidence.json")
    if check_step(prereq_data, prereq_confi):
        print("\33[92mStep counts match between result and confidence score files.\33[0m")
    else:
        raise ValueError("\33[91mMismatch in step counts between result and confidence score files.\33[0m")

    prereq_stat = calculate_statistics(prereq_data, prereq_confi)
    print(json.dumps(prereq_stat, indent=2))
    result_stat = calculate_statistics(result_data, result_confi)
    print(json.dumps(result_stat, indent=2))

    final_stats = {
        "prereq_stats": dict(prereq_stat),
        "result_stats": dict(result_stat)
    }

    # Output results
    output_json = json.dumps(final_stats, indent=2)

    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as outfile:
                outfile.write(output_json)
            print(f"Statistics successfully saved to {args.output_file}")
        except IOError as e:
            print(f"Error writing to output file {args.output_file}: {e}")

if __name__ == '__main__':
    main()
