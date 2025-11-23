# Academic Paper Insight Engine

## Overview
This tool processes collections of academic PDF research papers to extract insights, summarize content, and visualize relationships. It utilizes Natural Language Processing (NLP) to generate Knowledge Graphs, perform K-Means clustering on document embeddings, and summarize dense technical text. 

## Features
* **PDF Parsing:** Extracts text from PDF documents using PyMuPDF.
* **AI Summarization:** Uses T5-small (Transformer) models to generate concise summaries of long papers.
* **Knowledge Graph:** Builds a network graph connecting papers based on cosine similarity of their content (TF-IDF).
* **Semantic Clustering:** Clusters documents using BERT embeddings and visualizes them via PCA (Principal Component Analysis).
* **Visualization:** Includes Plotly and NetworkX visualizations for topic modeling and citation mapping.

## Installation

1.  **Clone the repository**
2.  **Install dependencies:**
    ```bash
    pip install torch transformers scikit-learn matplotlib networkx sentence-transformers pymupdf plotly dash requests pandas wordcloud
    ```

## Usage

1.  **Place PDFs:** Drop your PDF files into the `SamplePapers` directory.
2.  **Run the Processor:**
    ```bash
    python main.py
    ```
3.  **View Results:**
    * Console output will show centrality scores and summaries.
    * A Matplotlib window will open with the Knowledge Graph and Cluster visualization.
    * (Optional) Run `visualize.py` to start the Dash web interface.

## Structure
* `PDFProcessor.py`: Core logic for text extraction, embedding generation, and graph building.
* `visualize.py`: Visualization modules using Plotly and Dash.
* `main.py`: Execution script.
