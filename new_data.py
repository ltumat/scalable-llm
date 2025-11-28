import requests
from bs4 import BeautifulSoup
import json
import re
import time
import sys

def get_soup(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def get_interview_links(player_url):
    soup = get_soup(player_url)
    if not soup:
        return []
    
    links = []
    # Look for links to show_interview.php
    # The links might be relative.
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'show_interview.php' in href:
            if not href.startswith('http'):
                href = 'https://www.asapsports.com/' + href.lstrip('/')
            links.append(href)
            
    return list(set(links))

def clean_text(text):
    # Replace non-breaking spaces and other common artifacts
    text = text.replace('\u00a0', ' ').replace('\u00c2', '')
    # Replace non-breaking hyphens with standard hyphens
    text = text.replace('\u2011', '-')
    return text.strip()

def parse_interview(url):
    soup = get_soup(url)
    if not soup:
        return None

    # Extract text from the body or main content
    # We can try to find the main container. 
    # Based on the fetch_webpage output, the text is just there.
    # We will get all text and process it line by line.
    
    # Remove scripts and styles
    for script in soup(["script", "style"]):
        script.decompose()
        
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    conversations = []
    current_role = None
    current_text = []
    
    # We need to find where the interview starts.
    # Usually after "Q." appears.
    
    start_parsing = False
    
    for line in lines:
        # Check for footer
        if "FastScripts Transcript by ASAP Sports" in line:
            break
            
        # Check for speaker markers
        is_question = line.startswith("Q.")
        is_lebron = line.upper().startswith("LEBRON JAMES:") or line.upper().startswith("LBJ:")
        
        if is_question:
            start_parsing = True
            # Save previous turn
            if current_role and current_text:
                conversations.append({"from": current_role, "value": clean_text(" ".join(current_text))})
            
            current_role = "human"
            # Remove "Q."
            content = line[2:].strip()
            current_text = [content]
            
        elif is_lebron:
            start_parsing = True
            # Save previous turn
            if current_role and current_text:
                conversations.append({"from": current_role, "value": clean_text(" ".join(current_text))})
            
            current_role = "gpt"
            # Remove name
            parts = line.split(':', 1)
            if len(parts) > 1:
                content = parts[1].strip()
                current_text = [content]
            else:
                current_text = []
                
        elif start_parsing:
            # If we have started parsing and it's not a new speaker, append to current text
            # But we need to be careful about other speakers or metadata.
            # If another speaker appears (e.g. "COACH SPOELSTRA:"), we should probably stop or switch role.
            # For now, let's assume if it's not Q or LeBron, it's continuation OR another speaker.
            # If it looks like a speaker (UPPERCASE NAME:), we might want to handle it.
            
            # Simple heuristic: if line contains ":" and starts with uppercase words, it might be a speaker.
            # But let's stick to the user request: "retrieved for all of the lebron interviews".
            # If another speaker talks, maybe we should ignore it or treat it as context?
            # The user wants Lebron data.
            # If we treat other speakers as "human" (context for Lebron), it might be confusing if it's not a question.
            # Let's just append to current text for now, unless it clearly looks like another speaker.
            
            if ":" in line and len(line.split(':')[0]) < 30 and line.split(':')[0].isupper():
                # Likely another speaker
                # If we were in a turn, save it
                if current_role and current_text:
                    conversations.append({"from": current_role, "value": clean_text(" ".join(current_text))})
                
                # If it's not Lebron, we can either skip or treat as human/system.
                # Let's treat as human (context) so Lebron responds to it?
                # Or maybe just skip until next Q or Lebron.
                # Let's skip other speakers to be safe and only keep Q -> Lebron interactions if possible.
                # But often Q is followed by Lebron.
                current_role = None # Stop capturing
                current_text = []
            else:
                if current_role:
                    current_text.append(line)

    # Save last turn
    if current_role and current_text:
        conversations.append({"from": current_role, "value": clean_text(" ".join(current_text))})
        
    # Post-processing: Merge consecutive turns from the same speaker
    merged_conversations = []
    for turn in conversations:
        if merged_conversations and merged_conversations[-1]['from'] == turn['from']:
            merged_conversations[-1]['value'] += " " + turn['value']
        else:
            merged_conversations.append(turn)
    conversations = merged_conversations

    # Post-processing: Ensure conversation starts with human
    if conversations and conversations[0]['from'] == 'gpt':
        conversations.insert(0, {"from": "human", "value": "Could you share your thoughts on this?"})
    
    return conversations

def main():
    base_url = "https://www.asapsports.com/show_player.php?id=13888"
    print(f"Fetching interview links from {base_url}...")
    links = get_interview_links(base_url)
    print(f"Found {len(links)} interviews.")
    
    all_data = []
    
    # Limit for testing? No, user said "all".
    # But we should be careful about rate limits.
    
    for i, link in enumerate(links):
        print(f"Processing {i+1}/{len(links)}: {link}")
        convs = parse_interview(link)
        if convs:
            # We want to group the whole interview as one conversation?
            # Or split into Q&A pairs?
            # FineTome-100k usually has multi-turn conversations.
            # So one interview = one conversation entry.
            
            # Check if we have at least one Lebron response
            has_lebron = any(t['from'] == 'gpt' for t in convs)
            if has_lebron:
                all_data.append({"conversations": convs})
        
        # time.sleep(0.1) 

    output_file = "lebron_james/lebron_interviews_cleaned.jsonl"
    print(f"Saving {len(all_data)} conversations to {output_file}...")
    
    with open(output_file, "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
