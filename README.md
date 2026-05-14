# CureForge - Integrated Research Agent System
### Tasks 6, 7, and 8 - Technical Submission

Welcome to the CureForge Integrated Research System. This project merges a smart hypothesis storage system, a real-time performance tracker, and automated medical research tools into one unified AI agent.

---

## 1. Project Overview
CureForge is an autonomous agent designed to assist in biomedical research. This submission implements three major upgrades:
- **Task 6: The Hypothesis Bank** - A smart memory system using SQLite and FAISS to store and retrieve research ideas.
- **Task 7: Metrics Coordinator** - A centralized dashboard system that tracks speed, errors, and tool usage.
- **Task 8: Research Tools** - Deep integration with PubMed and ClinicalTrials.gov for real-world data fetching.

---

## 2. Prerequisites
Before you start, make sure you have:
- **Python 3.10 or higher** installed.
- **An OpenAI API Key** (or similar) to power the AI logic.
- **Internet connection** for the research tools to talk to medical databases.

---

## 3. Installation (Step-by-Step)

### Step A: Extract the Files
1. Right-click `CureForge_Unified_Submission.zip`.
2. Choose **Extract All** and select a folder.

### Step B: Open the Terminal
1. Open your project folder.
2. Click the address bar at the top, type `cmd`, and press **Enter**.

### Step C: Create and Activate the Virtual Environment
Copy and paste these commands:
```
python -m venv venv
.\\venv\\Scripts\\activate
```
*(You will see `(venv)` appear at the start of the line.)*

### Step D: Install Dependencies
```
pip install -r requirements.txt
```
*(This installs everything from LangChain to FAISS and Pydantic.)*

---

## 4. Configuration (API Keys)
1. Find the file `.env.example` in the main folder.
2. Rename it to exactly `.env`.
3. Open it with Notepad.
4. Add your API key: `LITELLM_API_KEY=your_key_here`
5. Save and close the file.

---

## 5. Running the Project

### To run the AI Agent:
```
python app/main.py
```
*The agent will begin its research and print logs to the screen.*

### To run the Real-Time Dashboard (The URL):
1. Open a **second** CMD window in the same folder.
2. Activate the environment: `.\\venv\\Scripts\\activate`
3. Start the server: `python app/server.py`
4. Open your browser to: **http://localhost:8000**

---

## 6. Running Tests
To verify all modules are working correctly:
```
pytest app/tests/
```
*All 82 tests should pass.*

---

## 7. Project Structure
- `app/main.py`: The entry point for the research agent.
- `app/server.py`: The web server for the browser dashboard.
- `app/src/core/tools/hypothesis_bank`: Logic for the smart memory.
- `app/src/utils/metrics.py`: The central tracker for performance.
- `app/src/core/tools/phases/research_tools`: API integrations for PubMed/Trials.
- `app/tests/`: Complete testing suite for all tasks.

---

## 8. Troubleshooting
- **"ModuleNotFoundError"**: Make sure you activated the venv (Step 3C).
- **"Connection Refused"**: Ensure `server.py` is running in a separate window.
- **"API Error"**: Check that your API key in `.env` is correct and active.