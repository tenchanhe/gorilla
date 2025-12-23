import json
import os
from collections import defaultdict

# --- Metrics Calculation Functions ---

def calculate_consistency(original_answer, candidate_responses):
    """
    Calculates the consistency score based on[cite: 141].
    Consistency = (1/M) * Sum(Indicator(Yi == Y_tilde))
    """
    m = len(candidate_responses)
    if m == 0:
        return 0.0

    agreement_count = 0
    for response in candidate_responses:
        # response is a list of Top-K guesses. response[0] is the top guess.
        if response and response[0]['tool'] == original_answer:
            agreement_count += 1

    return agreement_count / m

def calculate_avg_conf(original_answer, candidate_responses):
    """
    Calculates the average confidence score based on.
    Avg-Conf = Sum(Indicator(Yi == Y_tilde) * Ci) / Sum(Ci)
    """
    numerator = 0
    denominator = 0

    for response in candidate_responses:
        if not response:
            continue
        
        # The paper uses the confidence of the top guess for Avg-Conf weighting
        top_guess = response[0]
        tool_answer = top_guess['tool']
        
        # Ensure confidence is a number (handle string '85%' or similar if necessary)
        try:
            confidence = float(top_guess['confidence'])
        except (ValueError, TypeError):
            confidence = 0.0

        denominator += confidence
        if tool_answer == original_answer:
            numerator += confidence

    if denominator == 0:
        return 0.0

    return numerator / denominator

def calculate_pair_rank(candidate_responses):
    """
    Calculates a Pair-Rank score. 
    The paper proposes an optimization problem to find a distribution P[cite: 167].
    Here we use a Weighted Borda Count as a robust approximation for ranking aggregation
    without requiring a gradient descent solver.
    
    Logic: 
    - A guess at Rank 1 gets more points than Rank 2.
    - Points are weighted by the model's verbalized confidence.
    """
    scores = defaultdict(float)
    
    for response in candidate_responses:
        if not response:
            continue
            
        # response is [G1, G2, G3...]
        # K is the number of guesses provided in this response
        k = len(response)
        
        for rank, guess in enumerate(response):
            tool = guess['tool']
            try:
                conf = float(guess['confidence'])
            except:
                conf = 0.0
                
            # Weighting scheme:
            # 1. Rank weight: (K - rank) ensures higher guesses get more points.
            # 2. Confidence weight: scales the vote by the model's certainty.
            weight = (k - rank) * conf
            scores[tool] += weight
            
    # Normalize scores to form a probability distribution (sum to 1)
    total_score = sum(scores.values())
    if total_score == 0:
        return {}
        
    return {tool: score / total_score for tool, score in scores.items()}

# --- Data Processing Logic ---

def parse_scores_string(scores_str_list):
    """
    Parses the list of JSON strings found in the 'scores' field.
    Returns a list of Top-K guesses (G1, G2, G3).
    """
    if not scores_str_list:
        return []

    # Assuming scores_str_list contains one string that needs parsing, 
    # or multiple strings. We take the first valid parse.
    guesses = []
    
    # In some datasets, 'scores' is a list of strings. We iterate to find G1..G3.
    for score_str in scores_str_list:
        try:
            score_dict = json.loads(score_str)
        except json.JSONDecodeError:
            continue
            
        # Extract G1, G2, G3...
        # We assume the structure is like {'G1': {...}, 'G2': {...}}
        # We collect them in order.
        current_guesses = []
        for i in range(1, 10): # Support up to G9, though typically Top-3
            key = f"G{i}"
            if key in score_dict:
                current_guesses.append(score_dict[key])
        
        if current_guesses:
            guesses = current_guesses
            break # Found valid guesses
            
    return guesses

def main():
    base_dir = "score_confidence"
    target_files = ["prereq_confidence.json", "result_confidence.json"]
    confi_folders = ["confi1", "confi2", "confi3"]
    
    # Output structure: output[filename][item_id] = metrics
    output = {fname: {} for fname in target_files}

    for filename in target_files:
        print(f"Processing {filename} across {confi_folders}...")
        
        # 1. Data Gathering Phase
        # Map item_id -> list of responses from confi1, confi2, confi3
        aggregated_data = defaultdict(list)
        
        for folder in confi_folders:
            filepath = os.path.join(base_dir, folder, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: {filepath} not found.")
                continue
                
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        item_id = item['id']
                        raw_scores = item.get('scores', [])
                        
                        # Parse the Top-K guesses for this specific sample
                        guesses = parse_scores_string(raw_scores)
                        
                        # Even if guesses is empty, we append it to maintain the slot for this folder
                        # (metrics handle empty lists)
                        aggregated_data[item_id].append(guesses)
                        
                    except json.JSONDecodeError:
                        raise ValueError(f"Malformed JSON line in {filepath}: {line.strip()}")

        # 2. Metrics Calculation Phase
        results = {}
        for item_id, responses in aggregated_data.items():
            # Filter out completely empty processing errors, 
            # but keep empty lists if the file existed but had no scores (as 0 confidence)
            if not responses:
                continue

            # Define Original Answer (Y_tilde)
            # Strategy: Use the Top-1 guess from the first available response (confi1)
            # This aligns with evaluating "Given an answer Y, how confident are we?"
            original_answer = None
            for resp in responses:
                if resp:
                    original_answer = resp[0]['tool']
                    break
            
            if original_answer is None:
                continue

            breakpoint()
            # Calculate Metrics using the aggregated list of responses
            # responses shape: [[G1, G2..], [G1, G2..], [G1, G2..]]
            consistency = calculate_consistency(original_answer, responses)
            avg_conf = calculate_avg_conf(original_answer, responses)
            pair_rank = calculate_pair_rank(responses)

            results[item_id] = {
                "original_answer": original_answer,
                "consistency": consistency,
                "avg_conf": avg_conf,
                "pair_rank": pair_rank,
                "sample_count": len(responses)
            }
            
        output[filename] = results

    # Save or Print Results
    # Saving to a file is usually better for large outputs
    output_path = "aggregated_confidence_metrics.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"Done. Metrics saved to {output_path}")

if __name__ == "__main__":
    main()