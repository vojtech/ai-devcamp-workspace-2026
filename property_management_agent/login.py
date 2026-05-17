"""
Run this script once to authenticate with Gmail and save token.json.
It requests the scopes required by the Gmail MCP server.

Usage:
    python property_management_agent/login.py
"""
import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(AGENT_DIR, "credentials-web.json")
TOKEN_PATH = os.path.join(AGENT_DIR, "token.json")

print("Starting Google OAuth flow...")
print(f"Using credentials: {CREDENTIALS_PATH}")

flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
creds = flow.run_local_server(port=8080, open_browser=True)

with open(TOKEN_PATH, "w") as f:
    f.write(creds.to_json())

print(f"\nSuccess! token.json saved to: {TOKEN_PATH}")
print("You can now run: adk web property_management_agent")
