# German Invoice Extraction Agent
## Note: _My GOOGLE_API_KEY_ is also uploaded for testing purposes. Please let me know after reviewing, so that it can be removed :) 
A robust multi-agent system built with **LangGraph** and **Google Gemini** designed to extract structured data from German invoices. The system employs a workflow that includes extraction, automated auditing, and a human-in-the-loop fallback for low-confidence results.

## Features

- **Structured Extraction**: Utilizes Gemini models with Pydantic schemas to extract specific fields (Invoice Number, Date, Total Amount, IBAN, etc.).
- **Automated Auditing**: An audit node checks confidence scores and ensures critical fields are present.
- **Human-in-the-Loop**: Automatically flags invoices with low confidence or missing data for human review.
- **Async Processing**: Processes multiple invoices in parallel using `asyncio`.
- **Evaluation Suite**: Includes tools to benchmark extraction accuracy against ground truth datasets using Levenshtein distance.

## Dependencies

The project requires Python 3.11+ and the following packages:

- **Core**: `langgraph`, `langchain-google-genai`, `pydantic`, `python-dotenv`
- **Image Processing**: `pillow`
- **Evaluation & Data**: `datasets`, `Levenshtein`, `pandas`
- **Google SDK**: `google-generativeai`

### Installation

You can install all required dependencies using pip:

```bash
pip install langgraph langchain-google-genai pydantic python-dotenv pillow datasets Levenshtein google-generativeai pandas
```

## Setup

1. **API Key**: You need a Google Gemini API key. Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### 1. Running the Extraction Pipeline
To process a folder of images or a single image file:

```bash
python main.py --input_path /path/to/images/ --num_agents 5
```

- `--input_path`: Path to a directory of images or a single image file.
- `--num_agents`: Number of concurrent agents to run (default: 1).

**Outputs:**
- `approved_invoices.json`: Contains successfully extracted and audited data.
- `review_queue.json`: Contains filenames of invoices flagged for human review.
- `agent.log`: Detailed execution logs.

### 2. Running Evaluation
To evaluate the model's performance against a ground truth dataset (e.g., the Donut dataset):

```bash
python evaluate.py
```

This script compares predictions in `approved_invoices_donut.json` against `cleaned_ground_truth.json` and generates a detailed report in `evaluation_report.txt`.

## Project Structure

- **`main.py`**: Entry point. Configures the logger, compiles the LangGraph workflow, and manages the async worker pool.
- **`orchestrator.py`**: Defines the Pydantic data models (`GermanInvoice`) and initializes the Gemini LLM.
- **`invoice_agents.py`**: Contains the logic for the graph nodes: `extract_node`, `audit_node`, and `human_review_node`.
- **`evaluate.py`**: Handles data normalization, Levenshtein distance calculation, and report generation.
- **[Report](assesment/CloudFactory_report_reformatted.pdf)**: Detailed report covering the agentic workflow design, cost analysis, etc.