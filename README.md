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

## Accessing the Admin Page
Navigate to `/admin` in your browser after starting the application.

## Troubleshooting
- Ensure all dependencies are installed
- Verify your OpenAI API key is valid and active
- Check that you're running Python 3.8 or newer
