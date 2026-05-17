"""
Shared OAuth machinery for the property_management_agent package.

Both the Gmail tools (in agent.py) and the Drive tools (in drive_agent/)
need a valid Google OAuth token, so the credential loading + interactive
browser-based sign-in flow lives here as a single source of truth.

Key behaviours:
  - SCOPES is the UNION of every scope any sub-agent in this package needs
    (currently Gmail readonly + Drive readonly). When a new sub-agent needs
    a new scope, add it here.
  - If the on-disk token.json was issued for a STRICT SUBSET of SCOPES
    (e.g. user authenticated before Drive was added), we treat the token as
    invalid and force a re-auth so the user grants the extra scopes.
  - All actual API calls (`_ensure_authenticated`) return a (token, message)
    tuple — when token is None, message is a user-facing string the agent
    relays verbatim ("browser opened, please sign in").
"""
import logging
import os
import sys
import threading
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(PACKAGE_DIR, "token.json")
CREDENTIALS_PATH = os.path.join(PACKAGE_DIR, "credentials-web.json")
AUTH_PORT = 8080  # OAuth callback — must match a redirect URI registered
                  # in your Google Cloud OAuth client.

# Union of every scope any sub-agent in this package needs.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

# Shared auth state (one auth flow at a time, across all threads)
_auth_thread: Optional[threading.Thread] = None
_auth_done = threading.Event()
_auth_error: Optional[str] = None


def _run_auth_flow() -> None:
    """Runs the OAuth flow on a background thread, opens the browser,
    writes token.json. Sets _auth_done / _auth_error when complete."""
    global _auth_error
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=AUTH_PORT, open_browser=True)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
        logger.info("Google auth completed — token.json saved.")
    except Exception as e:
        _auth_error = str(e)
        logger.error(f"Google auth failed: {e}")
    finally:
        _auth_done.set()


def _has_required_scopes(creds: Credentials) -> bool:
    """True if the loaded token covers every scope in SCOPES."""
    granted = set(creds.scopes or [])
    return all(s in granted for s in SCOPES)


def _get_valid_credentials() -> Optional[Credentials]:
    """Load token.json, refresh if expired, validate scopes. Returns None
    when re-auth is needed (caller should trigger _ensure_authenticated)."""
    if not os.path.exists(TOKEN_PATH):
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if not _has_required_scopes(creds):
            logger.warning(
                "Existing token.json is missing required scopes; re-auth needed."
            )
            return None
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            logger.info("Refreshing Google OAuth token...")
            creds.refresh(Request())
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
            return creds
    except Exception as e:
        logger.warning(f"Error loading credentials: {e}")
    return None


def ensure_authenticated() -> tuple[Optional[str], Optional[str]]:
    """Return (access_token, user_message).

    - On success: (token, None)
    - On any failure mode (no creds file, expired, scope mismatch, in-progress
      sign-in, prior failure): (None, user_facing_message_to_relay)
    """
    global _auth_thread, _auth_error

    creds = _get_valid_credentials()
    if creds is not None:
        return creds.token, None

    if not os.path.exists(CREDENTIALS_PATH):
        return None, (
            f"Google authentication required but credentials-web.json is missing.\n"
            f"Place your OAuth client JSON at: {CREDENTIALS_PATH}"
        )

    if _auth_done.is_set() and _auth_error:
        err = _auth_error
        _auth_thread = None
        _auth_done.clear()
        _auth_error = None
        return None, (
            f"Google authentication failed: {err}\n"
            "Please ask me again to retry — a new browser window will open."
        )

    if _auth_thread is not None and _auth_thread.is_alive():
        return None, (
            "Google sign-in is in progress. Please complete the sign-in in the "
            "browser window that opened, then ask me again to continue."
        )

    _auth_done.clear()
    _auth_error = None
    _auth_thread = threading.Thread(target=_run_auth_flow, daemon=True)
    _auth_thread.start()
    return None, (
        "Google authentication required. A browser window has just been "
        "opened — please sign in and grant the requested permissions "
        "(Gmail + Drive read access), then ask me again."
    )


# Convenience for sub-modules that want a ready-to-use Credentials object.
def get_credentials_or_message() -> tuple[Optional[Credentials], Optional[str]]:
    """Like ensure_authenticated() but returns the Credentials object directly
    (or None + a user message). Useful for googleapiclient.discovery.build()."""
    token, msg = ensure_authenticated()
    if token is None:
        return None, msg
    return _get_valid_credentials(), None
