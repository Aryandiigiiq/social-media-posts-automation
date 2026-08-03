#!/usr/bin/env python3
"""
post.py - Post Selection & History Management Utility for Social Media Posts Automation

Usage:
  Interactive Mode:
    python post.py

  Command Line Mode:
    python post.py --file posts_20260801_001907.txt --select 1,3,7
    python post.py --latest --select 2
"""

import os
import sys
import re
import argparse
from datetime import datetime

HISTORY_FILE = "history.txt"

def find_latest_posts_file() -> str:
    """Finds the most recent posts_YYYYMMDD_HHMMSS.txt file in the workspace."""
    files = [f for f in os.listdir(".") if f.startswith("posts_") and f.endswith(".txt")]
    if not files:
        return ""
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def parse_posts_file(filepath: str) -> list:
    """Parses a timestamped posts file into individual post dictionaries."""
    if not os.path.exists(filepath):
        print(f"[!] Error: File '{filepath}' not found.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by POST separator blocks
    post_blocks = re.split(r'={70,}\nPOST\s+(\d+)\s+OF\s+\d+\s+\[(.*?)\]\n={70,}', content)
    
    posts = []
    # Block 0 is header summary
    for i in range(1, len(post_blocks), 3):
        post_num = post_blocks[i].strip()
        platform = post_blocks[i+1].strip()
        raw_body = post_blocks[i+2].strip()
        
        # Split report from post text
        if "----------------------------------------------------------------------" in raw_body:
            parts = raw_body.split("----------------------------------------------------------------------", 1)
            report_section = parts[0].strip()
            post_text = parts[1].strip()
        else:
            report_section = ""
            post_text = raw_body

        posts.append({
            "post_num": int(post_num),
            "platform": platform,
            "report": report_section,
            "post_text": post_text,
            "raw_block": raw_body,
            "source_file": filepath
        })
        
    return posts

def append_selected_to_history(selected_posts: list):
    """Appends selected post objects to history.txt."""
    if not selected_posts:
        print("[!] No posts selected to append.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_lines = ["\n"]

    for post in selected_posts:
        source_file = post.get("source_file", "Manual Selection")
        post_num = post.get("post_num", "N/A")
        platform = post.get("platform", "UNKNOWN").upper()
        report = post.get("report", "")
        post_text = post.get("post_text", "").strip()

        history_block = f"""======================================================================
POST SELECTED ON {timestamp} [FROM {source_file} - POST #{post_num} - {platform}]
======================================================================
{report}
----------------------------------------------------------------------

{post_text}
"""
        history_lines.append(history_block)

    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(history_lines))

    print(f"[+] Successfully appended {len(selected_posts)} post(s) to '{HISTORY_FILE}'.")

def interactive_mode():
    """Runs interactive selection prompt."""
    print("======================================================================")
    print("SOCIAL MEDIA POST SELECTION & HISTORY MANAGER")
    print("======================================================================")
    
    latest = find_latest_posts_file()
    if not latest:
        print("[!] No generated posts_*.txt files found in the current directory.")
        return

    print(f"[*] Found latest generated file: {latest}")
    use_latest = input(f"Do you want to use '{latest}'? (Y/n/filename): ").strip()
    
    if use_latest.lower() in ["", "y", "yes"]:
        target_file = latest
    elif os.path.exists(use_latest):
        target_file = use_latest
    else:
        target_file = latest

    posts = parse_posts_file(target_file)
    if not posts:
        print(f"[!] No valid posts found inside '{target_file}'.")
        return

    print(f"\n[+] Loaded {len(posts)} posts from '{target_file}':\n")
    for p in posts:
        first_line = p["post_text"].split("\n")[0] if p["post_text"] else "No text"
        print(f"  [{p['post_num']}] {p['platform']}")
        print(f"      Hook: \"{first_line[:75]}...\"\n")

    user_input = input("Enter post numbers selected for posting (e.g. '1, 3, 7' or 'all'): ").strip()
    if not user_input:
        print("[!] No input provided. Exiting.")
        return

    if user_input.lower() == "all":
        selected_indices = [p["post_num"] for p in posts]
    else:
        try:
            selected_indices = [int(x.strip()) for x in user_input.replace(",", " ").split() if x.strip().isdigit()]
        except Exception:
            print("[!] Invalid selection input.")
            return

    selected_posts = [p for p in posts if p["post_num"] in selected_indices]
    if not selected_posts:
        print("[!] None of the selected numbers matched posts in the file.")
        return

    append_selected_to_history(selected_posts)

def main():
    parser = argparse.ArgumentParser(description="Post selection and history appender utility.")
    parser.add_argument("--file", type=str, help="Path to posts_YYYYMMDD_HHMMSS.txt file.")
    parser.add_argument("--latest", action="store_true", help="Use the most recent posts_*.txt file.")
    parser.add_argument("--select", type=str, help="Comma-separated post numbers to select (e.g., '1,3,7').")
    
    args = parser.parse_args()

    if args.file or args.latest or args.select:
        target_file = find_latest_posts_file() if (args.latest or not args.file) else args.file
        posts = parse_posts_file(target_file)
        if not posts:
            return
            
        if not args.select:
            print("[!] --select argument required in CLI mode (e.g., --select 1,3,7).")
            return
            
        if args.select.lower() == "all":
            selected_indices = [p["post_num"] for p in posts]
        else:
            selected_indices = [int(x.strip()) for x in args.select.replace(",", " ").split() if x.strip().isdigit()]
            
        selected_posts = [p for p in posts if p["post_num"] in selected_indices]
        append_selected_to_history(selected_posts)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
