"""
One-off Google OAuth bootstrap. The interactive agent triggers its own
sign-in flow on demand, so you usually don't need to run this — but it's
useful for pre-warming token.json before the first agent session.

Usage:
    python3.11 property_management_agent/login.py

Imports the canonical SCOPES list from _auth.py so this script can never
drift out of sync with what the agent actually needs.
"""
import os

from google_auth_oauthlib.flow import InstalledAppFlow

from property_management_agent._auth import (
    CREDENTIALS_PATH,
    SCOPES,
    TOKEN_PATH,
    AUTH_PORT,
)


def main() -> None:
    print("Starting Google OAuth flow...")
    print(f"Using credentials: {CREDENTIALS_PATH}")
    print("Requesting scopes:")
    for s in SCOPES:
        print(f"  - {s}")

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=AUTH_PORT, open_browser=True)

    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    print(f"\nSuccess! token.json saved to: {TOKEN_PATH}")
    print("You can now run: adk web")


if __name__ == "__main__":
    main()
