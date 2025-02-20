# Problem Statement

## Objective
You have been provided with a set of Electronic Health Record (EHR) notes for patients diagnosed with various types of cancer. Your goal is to:

1. Retrieve relevant chunks of text from these EHR documents based on a user query (e.g. “Has the patient undergone chemotherapy?”).
2. Extract structured medical data focusing on:
   - **Cancer diagnosis** (type, date, histology, stage).
   - **Cancer-related medications** prescribed to the patient (drug name, start/end dates, intent).

## Dataset
The dataset (in JSON format) includes multiple EHR notes per patient. Each note is an object with the following structure:

```python
patient_data = [
    {
        "docDate": "MM-DD-YYYY",  # The date of the document in MM-DD-YYYY format.
        "docTitle": "Some Title",  # A short title describing the type of note (e.g., “Pathology Report,” “Progress Note,” “Chemotherapy Cycle #2”).
        "docText": "Text content of the EHR Note",  # The full text of the EHR note, typically 400–500 words, containing relevant clinical information.
    },
    # … more documents for the same patient …
]
```

---

## Task 1: Information Retrieval
Prepare a pipeline to extract relevant information chunks/sentences from the EHR data for a given query.

An approach could involve:
1. Breaking down the data into semantically meaningful chunks.
2. Using a text-embedding model to calculate similarity between the query and chunks from EHR notes.
3. Re-ranking models to extract relevant chunks.

### Example:
**Input Query:** “Has the patient undergone chemotherapy?”

**Output Retrieved Chunks:**
1. “On her follow-up on 02/15/2022 post her first cycle of chemotherapy, her response to Doxorubicin and Cyclophosphamide was evaluated.”
2. “A follow-up mammogram conducted on 02/15/2022 showed reduced tumor size in the left breast, indicating a positive response to chemotherapy.”
3. **Chemotherapy Cycle 3 Follow-up (05/05/2022)**
   - Assessing response to Docetaxel and Trastuzumab
   - Vital Signs: Stable
   - The patient reported manageable side effects, including mild fatigue and neuropathy.

**Note:** Any approach can be used (keyword matching, semantic embeddings, retrieval systems, etc.).

---

## Task 2: Medical Data Extraction
Develop an **LLM-based pipeline** to extract structured medical information from EHR notes, focusing on cancer diagnoses and medications.

### Why Is This Important?
EHRs often contain scattered information across multiple documents and dates. Structuring this data helps healthcare professionals:

1. Quickly review a patient’s cancer diagnosis details and staging.
2. Track treatments and medications over time.
3. Use structured data for clinical decision support (e.g., trial matching, treatment monitoring).

### Task 2.1: Cancer Diagnosis Characteristics
Extract key diagnosis details:

- **Primary Cancer Condition** (e.g., Breast Cancer, Lung Cancer).
- **Diagnosis Date** (Earliest date confirming the cancer).
- **Histology** (Microscopic classification, e.g., Adenocarcinoma, Squamous Cell Carcinoma).
- **Stage** (TNM classification and overall group stage).

### Task 2.2: Cancer-Related Medications
Extract details about medications given specifically for cancer treatment:

- **Medication Name** (e.g., Doxorubicin, Trastuzumab).
- **Start Date** (Earliest mention of medication initiation).
- **End Date** (If available; otherwise leave blank/null).
- **Intent** (Reason for prescription, e.g., Adjuvant therapy post-surgery).

---

## Expected Output Format
The final pipeline should return a **structured JSON** output:

```json
{
    "diagnosis_characteristics": [
        {
            "primary_cancer_condition": "str",
            "diagnosis_date": "MM-DD-YYYY",
            "histology": ["str"],
            "stage": {
                "T": "str",
                "N": "str",
                "M": "str",
                "group_stage": "str"
            }
        }
    ],
    "cancer_related_medications": [
        {
            "medication_name": "str",
            "start_date": "MM-DD-YYYY",
            "end_date": "MM-DD-YYYY",
            "intent": "str"
        }
    ]
}
```

---

## Hints & Modeling
1. **Use Task 1 pipeline** to locate relevant passages before extracting fields.
2. **LLM-based extraction**: Prompt the model with specific instructions to parse text into structured JSON.
3. **Handling missing data**: If a field (e.g., end date) is unavailable, leave it empty.

---

## Example LLM Setup (Qwen1.5-7B-Chat with 4-bit Quantization)
Use **Qwen1.5-7B-Chat** in Google Colab with 4-bit quantization:

```python
!pip install -q bitsandbytes accelerate optimum

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = "cuda"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    use_safetensors=True,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
```