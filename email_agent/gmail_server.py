# gmail_server.py
import os
import sys
import threading
from mcp.server.fastmcp import FastMCP
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import logging

# Critical: Force logs to stderr so standard output is reserved for MCP communication
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
AUTH_PORT = 8765  # Local port for the OAuth callback (must not clash with adk web port)

# This finds the exact folder where gmail_server.py lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute paths to credential files
CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, 'credentials-web.json')
TOKEN_PATH = os.path.join(SCRIPT_DIR, 'token.json')

logging.info(f"SCRIPT_DIR: {SCRIPT_DIR}")
logging.info(f"CREDENTIALS_PATH: {CREDENTIALS_PATH}")
logging.info(f"TOKEN_PATH: {TOKEN_PATH}")

# --- Auth state (shared between the background thread and the tool) ---
_auth_thread: threading.Thread | None = None
_auth_done = threading.Event()
_auth_error: str | None = None


def _run_auth_flow() -> None:
    """Runs the OAuth flow in a background thread, opens the browser, saves token.json."""
    global _auth_error
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=AUTH_PORT, open_browser=True)
        with open(TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
        logging.info("Gmail auth completed — token.json saved.")
    except Exception as e:
        _auth_error = str(e)
        logging.error(f"Gmail auth failed: {e}")
    finally:
        _auth_done.set()


def _get_valid_credentials() -> Credentials | None:
    """Returns valid credentials from token.json, refreshing if expired. Returns None if not present."""
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_PATH, 'w') as f:
                f.write(creds.to_json())
            return creds
    except Exception as e:
        logging.error(f"Error loading/refreshing credentials: {e}")
    return None


# Initialize FastMCP Server
mcp = FastMCP("GmailServer")


@mcp.tool()
def get_unread_emails(max_results: int = 5) -> str:
    """Fetches the subjects and senders of recent unread emails from Gmail.
    Will automatically prompt the user to sign in via browser if not yet authenticated."""
    global _auth_thread, _auth_done, _auth_error

    creds = _get_valid_credentials()

    if creds is None:
        # --- No valid token — need to authenticate ---

        # Auth already failed previously: reset and let user retry
        if _auth_done.is_set() and _auth_error:
            error = _auth_error
            _auth_thread = None
            _auth_done.clear()
            _auth_error = None
            return (
                f"Gmail authentication failed: {error}. "
                "Please ask me to fetch emails again to retry."
            )

        # Auth is still in progress
        if _auth_thread is not None and _auth_thread.is_alive():
            return (
                "Gmail sign-in is in progress. "
                "Please complete the sign-in in the browser window that opened, "
                "then ask me to fetch your emails again."
            )

        # Start a fresh auth flow — opens the browser automatically
        _auth_done.clear()
        _auth_error = None
        _auth_thread = threading.Thread(target=_run_auth_flow, daemon=True)
        _auth_thread.start()
        return (
            "Gmail authentication required. A browser window has been opened for you to sign in. "
            "Please grant the requested permissions, then ask me to fetch your emails again."
        )

    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(
            userId='me', labelIds=['UNREAD'], maxResults=max_results
        ).execute()
        messages = results.get('messages', [])

        if not messages:
            return "No unread emails found."

        output = []
        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
            output.append(f"From: {sender} | Subject: {subject}")

        return "\n".join(output)

    except Exception as e:
        return f"Error fetching emails: {str(e)}"


if __name__ == "__main__":
    # Start listening for ADK requests on standard input/output
    mcp.run(transport="stdio")