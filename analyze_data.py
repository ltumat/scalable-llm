import json

def analyze_conversations(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    starts_with_gpt = 0
    consecutive_gpt = 0
    total = len(lines)
    
    for line in lines:
        data = json.loads(line)
        convs = data['conversations']
        
        if not convs:
            continue
            
        if convs[0]['from'] == 'gpt':
            starts_with_gpt += 1
            
        for i in range(len(convs) - 1):
            if convs[i]['from'] == 'gpt' and convs[i+1]['from'] == 'gpt':
                consecutive_gpt += 1
                break # Count each file at most once for this
                
    print(f"Total conversations: {total}")
    print(f"Starts with GPT: {starts_with_gpt}")
    print(f"Has consecutive GPT turns: {consecutive_gpt}")

analyze_conversations('/Users/jonaslorenz/Desktop/Code_KTH/ScalableML/scalable-llm/lebron_james/lebron_interviews_cleaned.jsonl')
