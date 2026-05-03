import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CREDENTIALS_PATH = "email_agent/credentials-web.json"
TOKEN_PATH = "email_agent/token.json"

print("Starting Google Login...")

# This will force the browser to open and ask for permission
flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
creds = flow.run_local_server(port=8080)

# Save the resulting token so the MCP agent can use it
with open(TOKEN_PATH, 'w') as token:
    token.write(creds.to_json())

print("✅ Success! token.json has been created. Your agent is now ready.")