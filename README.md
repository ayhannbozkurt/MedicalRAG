# Medical RAG System

A Retrieval-Augmented Generation (RAG) system for medical information and drug interactions. This system uses local PDF and JSON data sources to provide accurate medical information.

## Features

- Drug interaction queries
- Medical literature search
- Academic research integration
- Modern Streamlit interface
- Local data processing

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ayhannbozkurt/Medical-RAG.git
cd Medical-RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

4. Run the application:
```bash
streamlit run app.py
```

## Data Sources

- Drug interaction database
- Medical research papers (PDF format)
- Clinical guidelines

## Usage

1. Enter your medical question in the text area
2. Click "Get Answer" to receive information
3. View the answer and supporting sources

## Security Note

This system processes all data locally and does not store any patient information. API keys are required for language model access only.

## License

MIT License
