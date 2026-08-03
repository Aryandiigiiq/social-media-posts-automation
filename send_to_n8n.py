#!/usr/bin/env python3
"""
send_to_n8n.py - Standalone N8N Webhook Payload Dispatch, SQLite Permanent Logger & History Sync Utility

Reads the clean X post text from output/x_post.txt and metadata from output/x_post_metadata.json.
When the post is successfully dispatched to n8n, it transitions the post from temporary state to permanent state:
1. Inserts a permanent record into the lightweight SQLite database (dispatched_posts.db).
2. Appends the dispatched post content to history.txt.

Usage:
    python send_to_n8n.py
    python send_to_n8n.py --file output/x_post.txt
    python send_to_n8n.py --text "Custom tweet content to send to n8n"
"""

import os
import sys
import json
import sqlite3
import argparse
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

N8N_API_KEY = os.getenv("N8N_API_KEY", "")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")
DEFAULT_X_FILE = os.path.join("output", "x_post.txt")
DEFAULT_META_FILE = os.path.join("output", "x_post_metadata.json")
HISTORY_FILE = "history.txt"
DB_FILE = "dispatched_posts.db"

def init_sqlite_db():
    """Initializes the lightweight SQLite database and dispatched_x_posts table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dispatched_x_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dispatched_at TEXT NOT NULL,
                post_text TEXT NOT NULL,
                style_blend TEXT NOT NULL,
                topics_used TEXT,
                sources_used TEXT,
                hook_type TEXT,
                content_structure TEXT,
                closure_type TEXT,
                detailed_critique TEXT,
                platform TEXT DEFAULT 'X'
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Error initializing SQLite database '{DB_FILE}': {e}")

def save_to_sqlite(post_text: str, meta: dict = None):
    """Permanently logs the dispatched X post into the lightweight SQLite database."""
    try:
        init_sqlite_db()
        meta = meta or {}
        
        timestamp_str = datetime.now(timezone.utc).isoformat()
        style_blend = meta.get("style_blend", "User Selected Writing Style")
        topics_used = ", ".join(meta.get("topics_used", [])) if isinstance(meta.get("topics_used"), list) else str(meta.get("topics_used", "Custom Topics"))
        sources_used = ", ".join(meta.get("sources_used", [])) if isinstance(meta.get("sources_used"), list) else str(meta.get("sources_used", "Web Search"))
        hook_type = meta.get("hook_type", "Surprising Metric / Insight")
        content_structure = meta.get("content_structure", "3-Act Narrative Arc")
        closure_type = meta.get("closure_type", "Reflective Discussion CTA")
        detailed_critique = meta.get("detailed_critique", "Passed all quality gates.")
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO dispatched_x_posts (
                dispatched_at, post_text, style_blend, topics_used, sources_used,
                hook_type, content_structure, closure_type, detailed_critique, platform
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp_str, post_text.strip(), style_blend, topics_used, sources_used,
            hook_type, content_structure, closure_type, detailed_critique, 'X'
        ))
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        print(f"[+] Permanently saved dispatched X post (ID #{last_id}) to SQLite database '{DB_FILE}'!")
    except Exception as e:
        print(f"[!] Error saving to SQLite database '{DB_FILE}': {e}")

def append_to_history(post_text: str, platform: str = "X"):
    """Appends dispatched post content directly to history.txt."""
    try:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_block = f"""

======================================================================
POST DISPATCHED TO N8N ON {timestamp_str} [{platform.upper()}]
======================================================================
{post_text.strip()}
"""
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry_block)
        print(f"[+] Automatically appended dispatched {platform} post to '{HISTORY_FILE}'!")
    except Exception as e:
        print(f"[!] Warning updating history.txt: {e}")

def send_x_post_to_n8n(post_text: str, webhook_url: str = None, api_key: str = None, meta: dict = None, sync_history: bool = True) -> bool:
    """Dispatches clean X post text payload to n8n webhook endpoint, persists to SQLite DB, and updates history.txt."""
    url = webhook_url or N8N_WEBHOOK_URL
    key = api_key or N8N_API_KEY
    
    if not url or url.startswith("https://your-n8n-instance.com"):
        print("[!] Error: N8N_WEBHOOK_URL is not configured in .env file.")
        print("[!] Please set N8N_WEBHOOK_URL=https://your-n8n-domain/webhook/... in .env")
        return False
        
    if not post_text or not post_text.strip():
        print("[!] Error: Post text content is empty. Nothing to send.")
        return False

    clean_text = post_text.strip()
    if len(clean_text) > 270:
        print(f"[!] WARNING: Post length ({len(clean_text)} chars) exceeds 270 char limit! Trimming for n8n/X safety...")
        clean_text = clean_text[:267].rsplit(' ', 1)[0] + "..."

    payload_data = {
        "api_key": key,
        "platform": "X",
        "text": clean_text,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    json_bytes = json.dumps(payload_data).encode("utf-8")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "X-N8N-API-KEY": key,
        "User-Agent": "DIGIiq-Content-Engine/1.0"
    }
    
    print(f"[->] Dispatching payload to n8n webhook: {url}")
    try:
        req = urllib.request.Request(url, data=json_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            status_code = resp.status
            response_body = resp.read().decode("utf-8")
            print(f"[+] SUCCESS: Payload delivered to n8n! (HTTP {status_code})")
            if response_body:
                print(f"[+] n8n Response: {response_body[:200]}")
            
            # Transition from temporary to permanent state upon successful dispatch
            save_to_sqlite(clean_text, meta)
            
            if sync_history:
                append_to_history(clean_text, platform="X")
                
            return True
    except Exception as e:
        print(f"[!] FAILED to send payload to n8n: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Dispatch X post content to n8n workflow webhook, persist to SQLite DB, and update history.txt.")
    parser.add_argument("--file", "-f", default=DEFAULT_X_FILE, help="Path to text file containing X post (default: output/x_post.txt)")
    parser.add_argument("--meta", "-m", default=DEFAULT_META_FILE, help="Path to metadata JSON file (default: output/x_post_metadata.json)")
    parser.add_argument("--text", "-t", help="Direct text string to send instead of reading file")
    parser.add_argument("--url", help="Override N8N_WEBHOOK_URL")
    parser.add_argument("--key", help="Override N8N_API_KEY")
    parser.add_argument("--no-history", action="store_true", help="Disable auto-appending dispatched post to history.txt & SQLite")
    
    args = parser.parse_args()
    
    meta_data = {}
    if os.path.exists(args.meta):
        try:
            with open(args.meta, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        except Exception as e:
            print(f"[!] Note loading metadata file '{args.meta}': {e}")

    if args.text:
        text_to_send = args.text
        print(f"[*] Loaded post text from CLI argument ({len(text_to_send)} chars).")
    else:
        file_path = args.file
        if not os.path.exists(file_path) and os.path.exists("x_post.txt"):
            file_path = "x_post.txt"
            
        if not os.path.exists(file_path):
            print(f"[!] Error: File '{file_path}' not found!")
            print(f"[!] Run 'python main.py' first to generate '{file_path}'.")
            sys.exit(1)
            
        with open(file_path, "r", encoding="utf-8") as f:
            text_to_send = f.read().strip()
            
        print(f"[*] Loaded X post content from '{file_path}' ({len(text_to_send)} chars).")
        
    print("-" * 60)
    print(text_to_send)
    print("-" * 60)
    
    success = send_x_post_to_n8n(text_to_send, webhook_url=args.url, api_key=args.key, meta=meta_data, sync_history=not args.no_history)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
