import json
import numpy as np

def analyze_confidence():
    ground_truth_file = 'score_confidence/groundT/ground_truth.json'
    result_confidence_file = 'score_confidence/groundT/result_confidence.json'

    ground_truth_data = {}
    with open(ground_truth_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            score_data = json.loads(data['scores'][0])
            ground_truth_data[data['id']] = score_data['need_tool']

    result_confidence_data = {}
    with open(result_confidence_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            score_data = json.loads(data['scores'][0])
            result_confidence_data[data['id']] = score_data

    correct_predictions = 0
    total_predictions = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    confidence_levels = []

    for id, result_data in result_confidence_data.items():
        if id in ground_truth_data:
            total_predictions += 1
            ground_truth_need_tool = ground_truth_data[id]
            result_need_tool = result_data['need_tool']

            if ground_truth_need_tool == result_need_tool:
                correct_predictions += 1
            
            # 4 quadrants
            if result_need_tool and ground_truth_need_tool:
                tp += 1
            elif result_need_tool and not ground_truth_need_tool:
                fp += 1
            elif not result_need_tool and not ground_truth_need_tool:
                tn += 1
            elif not result_need_tool and ground_truth_need_tool:
                fn += 1

            confidence = 0
            # 取tools裡的最大confidence（如果LLM覺得要一個以上）
            if 'tools' in result_data and result_data['tools']:
                confidences = [tool['confidence'] for tool in result_data['tools']]
                if confidences:
                    confidence = max(confidences)

            confidence_levels.append({
                'id': id,
                'ground_truth': ground_truth_need_tool,
                'predicted': result_need_tool,
                'confidence': confidence
            })
        else:
            raise ValueError(f"ID {id} not found in ground truth data.")

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Accuracy of need_tool prediction: {accuracy:.2f}%")
    
    # 打印TP, FP, TN, FN
    print("\nFour Quadrants:")
    print(f"TP (True Positives): {tp}")
    print(f"FP (False Positives): {fp}")
    print(f"TN (True Negatives): {tn}")
    print(f"FN (False Negatives): {fn}")

    # Analysis of confidence vs. ground truth
    bins = np.arange(0, 101, 10)
    bin_labels = [f'{i}-{i+10}' for i in bins[:-1]]
    
    # Tool needed
    tool_needed_counts = {label: {'total': 0, 'correct': 0} for label in bin_labels}
    for item in confidence_levels:
        if item['ground_truth']:
            bin_index = min(int(item['confidence'] / 10), 9)
            label = bin_labels[bin_index]
            tool_needed_counts[label]['total'] += 1
            if item['predicted']:
                tool_needed_counts[label]['correct'] += 1

    print("\nConfidence analysis when tool is needed (Ground Truth = True):")
    for label, counts in tool_needed_counts.items():
        if counts['total'] > 0:
            percentage = (counts['correct'] / counts['total']) * 100
            print(f"Confidence {label}: {counts['correct']}/{counts['total']} ({percentage:.2f}%) predicted True")
        else:
            print(f"Confidence {label}: 0/0 (0.00%)")
            
    # Tool not needed
    tool_not_needed_counts = {label: {'total': 0, 'correct_non_call': 0} for label in bin_labels}
    for item in confidence_levels:
        if not item['ground_truth']:
            bin_index = min(int(item['confidence'] / 10), 9)
            label = bin_labels[bin_index]
            tool_not_needed_counts[label]['total'] += 1
            if not item['predicted']:
                tool_not_needed_counts[label]['correct_non_call'] += 1

    print("\nConfidence analysis when tool is NOT needed (Ground Truth = False):")
    for label, counts in tool_not_needed_counts.items():
        if counts['total'] > 0:
            percentage = (counts['correct_non_call'] / counts['total']) * 100
            print(f"Confidence {label}: {counts['correct_non_call']}/{counts['total']} ({percentage:.2f}%) predicted False")
        else:
            print(f"Confidence {label}: 0/0 (0.00%)")
    
    generate_confidence_histogram(confidence_levels)

def generate_confidence_histogram(confidence_levels):
    print("\nConfidence Score Histogram:")
    
    confidences = [item['confidence'] for item in confidence_levels]
    
    # Use bins=10 and range=(0,100) for scores from 0 to 100.
    hist, bin_edges = np.histogram(confidences, bins=10, range=(0, 100))
    
    # 打印histogram
    for i in range(len(hist)):
        bin_range = f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
        bar = '█' * int(hist[i] / len(confidences) * 50) if len(confidences) > 0 else '' # Scale bar for display
        count = hist[i]
        print(f"{bin_range}: {bar} ({count})")

if __name__ == '__main__':
    analyze_confidence()
