import json
from google_auth_oauthlib.flow import InstalledAppFlow

# Since we are running this directly in the email_agent folder, 
# it will look right next to itself.
CREDENTIALS_PATH = "email_agent/credentials-web.json"

print("--- Starting Credential Test ---")

try:
    # 1. Test if it's valid JSON
    with open(CREDENTIALS_PATH, 'r') as f:
        data = json.load(f)
        print("✅ SUCCESS: File was found and is valid JSON.")
        
        # Print what type of credential it is
        if "installed" in data:
            print("✅ TYPE: OAuth Desktop App (Perfect)")
        elif "web" in data:
            print("✅ TYPE: OAuth Web App (Usually works)")
        elif "type" in data and data["type"] == "service_account":
            print("❌ FAILURE: This is a Service Account file. You need an OAuth Client ID.")
        else:
            print("❓ TYPE: Unknown format.")

    # 2. Test if the Google library accepts it
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_PATH, 
        scopes=['https://www.googleapis.com/auth/gmail.readonly']
    )
    print("✅ SUCCESS: The Google Auth library successfully loaded the file!")

except FileNotFoundError:
    print("❌ ERROR: credentials.json is still not in this folder.")
except json.JSONDecodeError:
    print("❌ ERROR: The JSON is corrupted. Did you accidentally delete a bracket or quote?")
except ValueError as e:
    print(f"❌ ERROR: The file is the wrong format for OAuth. Details: {e}")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")

print("--- Test Complete ---")