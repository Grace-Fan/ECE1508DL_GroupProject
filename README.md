# ECE1508DL_2026_GroupProject

This project: 

## Project Structure

- **`chunking.py/`**  
  Contains all chunking strategies.

- **`rag_pipeline.py/`**  
  Contains all function for RAG pipline.

- **`main_experiment.py/`**  
  Runs dataset and evaluation.

- **`requirements.txt`**  
  Lists the Python dependencies required for the project.

## Setup 

### 1. Create and Activate a Virtual Environment

Run the following commands to set up a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Change LLM API

Change the LLM API in the code as showed:

```bash
client = OpenAI(
    api_key=os.environ.get('YOUR-API-KEY'),
    base_url="https://api.deepseek.com")
```

### 3. Run the code 

Run the full RAG system using the following command:
```bash
python main_experiment.py
```
