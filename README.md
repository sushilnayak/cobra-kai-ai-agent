## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up Google Cloud credentials:
   - Create a service account in Google Cloud
   - Download the service account key file
   - Set the environment variable:
     ```
     # On Windows
     set GOOGLE_APPLICATION_CREDENTIALS=path\to\your\credentials.json
     
     # On macOS/Linux
     export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
     ```
   - Set your Google Cloud project ID in a `.env` file:
     ```
     GCP_PROJECT=your-project-id
     ```

4. Run the application:
   ```
   python.exe main.py
   ```