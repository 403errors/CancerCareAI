# CancerCareAI: AI-Powered Patient Data Extraction

This project implements an AI-powered system for extracting cancer-related information from patient Electronic Health Record (EHR) notes. It addresses two main tasks:

1.  **Information Retrieval:** Retrieving relevant text chunks based on a user query.
2.  **Medical Data Extraction:** Extracting structured data (diagnosis and medication details) into a JSON format.

**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13bzx0MyOojzwq6f8PcUOp5o_LvXt6B1E?usp=sharing)** 

## Project Structure

The project is implemented in Python and is structured as a single, well-commented Jupyter Notebook (`CancerCareAI.ipynb`).   The notebook is divided into four main sections:

1.  **Project Setup and Data Loading:** Installs dependencies, imports libraries, and loads data from a GitHub repository.
2.  **Task 1 - Information Retrieval (Pipeline):** Implements a combined keyword-based (BM25) and semantic search (Sentence Transformers, CrossEncoder) pipeline for retrieving relevant sentences.
3.  **Task 2 - Medical Data Extraction (LLM-based Pipeline):** Uses a quantized Large Language Model (Qwen/Qwen2.5-7B-Instruct-1M) to extract structured data in JSON format.  Includes robust error handling for JSON parsing.
4.  **Putting it all Together (Main Execution Block):** Provides an interactive interface for the user to select a patient, choose a mode (information retrieval or data extraction), and view the results.

## Task 1: Information Retrieval

**Approach:**

The information retrieval task uses a multi-stage approach to combine the strengths of different retrieval methods:

1.  **Sentence Tokenization:** Input documents are split into individual sentences using `nltk.sent_tokenize`.  This provides a more granular level of retrieval compared to using entire documents.
2.  **BM25 Ranking:**  The `rank_bm25` library is used to perform keyword-based ranking.  This is effective for finding sentences that contain the exact query terms.
3.  **Semantic Search:**  The `sentence-transformers` library is used with the "all-MiniLM-L6-v2" model to find sentences that are semantically similar to the query, even if they don't share exact keywords.
4.  **Filtering:** The top *N* results from both BM25 and semantic search are combined.  Irrelevant/administrative sentences are removed using regular expression based filtering.
5.  **Cross-Encoder Reranking:** A CrossEncoder model ("cross-encoder/ms-marco-MiniLM-L-6-v2") is used to rerank the combined results.  CrossEncoders are more accurate than the Bi-Encoders used in the initial semantic search.
6.  **Score Normalization and Combination:** Scores from BM25, semantic search, and the CrossEncoder are normalized to a 0-1 range and combined using weighted averaging. This allows for tuning the influence of each method.

**[YouTube Video Demo (Task 1)](https://youtu.be/_N7l-hswtaU)**

## Task 2: Medical Data Extraction

**Approach:**

The medical data extraction task leverages the Qwen/Qwen2.5-7B-Instruct-1M large language model (LLM) with 4-bit quantization to extract structured data.

1.  **Model Loading:** The Qwen model and tokenizer are loaded using the `transformers` library.  4-bit quantization (using `bitsandbytes`) is applied to reduce memory usage, enabling the model to run on a T4 GPU in Google Colab.  If a GPU is not available, the model loading is skipped.
2.  **Prompt Engineering:** A carefully designed prompt is constructed to instruct the LLM to extract specific data elements (diagnosis characteristics and cancer-related medications) and output them in a strict JSON format.  The prompt includes:
    *   Clear instructions on the LLM's role and task.
    *   An example input and expected output.
    *   Specific guidelines for handling missing data (using `null`).
3.  **Inference:** The LLM generates text based on the prompt and input passage.  Inference parameters are set for deterministic output (greedy decoding, low temperature, top-k sampling).
4.  **JSON Extraction and Error Handling:**  The generated text is parsed to extract the JSON object.  Robust error handling is implemented to deal with potential `JSONDecodeError` exceptions, and includes a fallback mechanism to attempt to recover partial JSON outputs. A regular expression based approach is used to first find the JSON code block and then parse.
5. **Data Aggregation:** The `merge_extractions` function handles combining and deduplicating data extracted from multiple documents for the same patient. It prioritizes earlier diagnosis dates and combines medication information.

**[YouTube Video Demo (Task 2)](https://youtu.be/TzEx-vvSADw)**

## Running the Code

1.  **Open in Colab:** The recommended way to run the code is in Google Colab. Use the Colab link provided.
2.  **Runtime:** Ensure you are using a T4 GPU runtime (Runtime -> Change runtime type). This is *required* for the 4-bit quantization of the Qwen model. If bitsandbytes issues occur, try restarting the runtime.
3.  **Run All:** Execute all cells in the notebook (Runtime -> Run all).
4.  **Interactive Prompts:** The script will prompt you to:
    *   Select a patient.
    *   Choose a mode (1 for Information Retrieval, 2 for Medical Data Extraction).
    *   Enter a query (for Mode 1).

## Dependencies

*   sentence-transformers
*   rank\_bm25
*   pandas
*   nltk
*   bitsandbytes
*   accelerate
*   optimum
*   transformers
*   torch
*   requests

These dependencies are installed at the beginning of the `CancerCareAI.ipynb` notebook using `pip`.
