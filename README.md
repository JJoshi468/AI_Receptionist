# Project Setup and Installation

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Project Structure
```
project_directory/
│
├── flask_app.py
├── config.py
├── templates/
│   ├── (template files)
│   └── ...
└── README.md
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/JJoshi468/AI_Receptionist
cd <project-directory>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
1. Open `config.py`
2. Change the project name as needed
3. Add a valid OpenAI API key

### 5. Run the Application
```bash
python flask_app.py
```
### 6. Sample user queries
It can be found in training data in 'improve model.py'.

## Accessing the Admin Page
Navigate to `/admin` in your browser after starting the application.

## Troubleshooting
- Ensure all dependencies are installed
- Verify your OpenAI API key is valid and active
- Check that you're running Python 3.8 or newer


# Google Calendar API Setup Guide for Python

## Prerequisites
- Python 3.7+
- Google Cloud Account
- Google Workspace account or personal Google account

## Step 1: Create a Google Cloud Project
1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" at the top of the page
3. Click "New Project"
4. Enter a project name and click "Create"

## Step 2: Enable Google Calendar API
1. In the Google Cloud Console, go to the Navigation Menu
2. Select "APIs & Services" > "Library"
3. Search for "Google Calendar API"
4. Click on the API and then click "Enable"

## Step 3: Create Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" and select "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in required application information
   - Add scopes: `.../auth/calendar.readonly` or `.../auth/calendar`
   - Add test users (your email)
4. For "Application type", select "Desktop app"
5. Name your OAuth 2.0 client
6. Click "Create"

## Step 4: Download credentials.json
1. After creating the client ID, click the download button (download icon)
2. Save the `credentials.json` file in your project directory

## Step 5: Install Required Python Libraries
```bash
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

## References
- [Google Calendar API Python Quickstart](https://developers.google.com/calendar/api/quickstart/python)
- [Google Cloud Console](https://console.cloud.google.com/)
