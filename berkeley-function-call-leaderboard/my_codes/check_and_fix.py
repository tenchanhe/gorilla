
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

SCENARIO = ['customer', 'healthcare', 'finance', 'student', 'notetaker']
PREREQ_COUNT = {
    'customer': 10,
    'healthcare': 5,
    'finance': 7,
    'student': 10,
    'notetaker': 5
}

def load_json(path: Path):
    return_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            return_data.append(data)
    return return_data

def count_scenarios(obj):
        scenario_counts = Counter()
        for entry in obj:
            for scenario in SCENARIO:
                if scenario in entry.get("id", ""):
                    scenario_counts[scenario] += 1
        return scenario_counts

def check_entries(result_obj, answer_obj):
    print(f"Result Count = {len(result_obj)}")
    print(f"Answer Count = {len(answer_obj)}")
    result_counts = count_scenarios(result_obj)
    answer_counts = count_scenarios(answer_obj)
    print("Result scenario counts:", dict(result_counts))
    print("Answer scenario counts:", dict(answer_counts))
    for result in result_obj:
        if isinstance(result["result"], str):
            return False
    if len(result_obj) == len(answer_obj) and result_counts == answer_counts:
        return True
    return False

def fix_result(result_obj):
    for result in result_obj:
        if isinstance(result["result"], str):
            if result["result"].startswith("Error during inference:"):
                result_obj.remove(result)
    return result_obj

def check_prereq(prereq_obj):
    flag = True
    print(f"Result Count = {len(prereq_obj)}")
    print(f"Answer Count = {sum(PREREQ_COUNT.values())}")
    result_counts = count_scenarios(prereq_obj)
    print("Result scenario counts:", dict(result_counts))
    print("Answer scenario counts:", PREREQ_COUNT)
    for entry in prereq_obj:
        if isinstance(entry["result"], str):
            print(entry['id'])
            flag = False
    if len(prereq_obj) == sum(PREREQ_COUNT.values()) and all(result_counts[scn] == PREREQ_COUNT[scn] for scn in SCENARIO):
        flag = True
    return flag


def main():
    parser = argparse.ArgumentParser(description="比對 result.json 與 answer.json 的條目數量是否相同")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--answer", type=Path)
    parser.add_argument("--fix", default=False, action='store_true')
    args = parser.parse_args()

    result_file = Path(args.result) / "BFCL_v4_memory_kv_result.json"
    prereq_file = Path(args.result) / "BFCL_v4_memory_kv_prereq_result.json"
    result_obj = load_json(result_file)
    answer_obj = load_json(args.answer)
    prereq_obj = load_json(prereq_file)
    
    print("=== Checking Prereq Entries ===")
    if check_prereq(prereq_obj):
        print("✅ Prereq Entries same. No need to fix.\n")
    else:
        print("\33[31mPrereq Error entries\33[0m\n")
        if args.fix:
            fixed_prereq = fix_result(prereq_obj)
            check_prereq(fixed_prereq)
            with open(prereq_file, 'w', encoding='utf-8') as f:
                for entry in fixed_prereq:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            print("If you want to fix it, please add --fix argument.")
    
    print("=== Checking Result Entries ===")
    if check_entries(result_obj, answer_obj):
        print("✅ Result Entries same. No need to fix.\n")
    else:
        print("\33[31mResult Error entries\33[0m\n")
        if args.fix:
            fixed_result = fix_result(result_obj)
            check_entries(fixed_result, answer_obj)
            with open(result_file, 'w', encoding='utf-8') as f:
                for entry in fixed_result:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        else:
            print("If you want to fix it, please add --fix argument.")
    
    


if __name__ == "__main__":
    main()
