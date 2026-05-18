# DevBrain - Semantic Second Brain

This project is the practical core for studying the **Embeddings and Context Management** module within the Master's in Applied AI. The objective of **DevBrain** is to incrementally build a semantic information retrieval system and technical knowledge base (*RAG - Retrieval-Augmented Generation*) optimized for software developers.

## 📂 Project Structure

```text
devbrain-embeddings/
│
├── .gitignore
├── README.md
├── requirements.txt         # Project dependencies
├── data/                    # Technical documentation, Markdown, or test source code
│   └── sample_docs/
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Environment variables and global configurations
│   │
│   ├── core/                # Pure AI logic and Linear Algebra
│   │   ├── __init__.py
│   │   ├── embeddings.py    # Vector generation (Local models / APIs)
│   │   └── metrics.py       # Similarity metrics (Manual Cosine Similarity)
│   │
│   ├── ingestion/           # Text processing pipeline
│   │   ├── __init__.py
│   │   ├── loaders.py       # File readers (.md, .txt, .py)
│   │   └── chunkers.py      # Indexable segmentation logic
│   │
│   ├── storage/             # Persistence layer
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract storage interface
│   │   ├── local_matrix.py  # In-memory RAM matrix with NumPy
│   │   └── vector_db.py     # Vector Database integration (Chroma/Qdrant)
│   │
│   ├── api/                 # Service layer and endpoints
│   │   └── ...
│   │
│   └── labs/                # Academic & Research Sandbox (Step-by-step Labs)
│       ├── __init__.py
│       └── lab03/
│           └── run_cli.py   # Lab 03: Local embeddings & Math similarity validation

```
---

## 🛠️ Installation and Environment Setup Guide

To ensure that project dependencies do not interfere with your operating system's global libraries, the installation is carried out in an isolated manner using a **Python Virtual Environment (`venv`)**.

### Prerequisites

* Python 3.10 or higher installed on your system.
* Up-to-date `pip` package manager.

### Step 1: Clone the Project or Create the Working Directory

If you are initializing the project from scratch, run:

```bash
mkdir devbrain-embeddings
cd devbrain-embeddings

```

### Step 2: Create the Virtual Environment

Create an isolated virtual environment inside the project root folder by executing:

```bash
# On Linux, macOS, and Windows (Git Bash / Terminal)
python3 -m venv venv

```

*Note: This command will create a directory named `venv/` containing a self-contained local binary copy of Python and Pip.*

### Step 3: Activate the Virtual Environment

Before installing any packages, you must activate the environment to redirect your terminal's environment variables:

* **macOS:**
```bash
source venv/bin/activate

```


Once activated, you will notice that your terminal prompt will include the prefix `(venv)`, indicating that you are operating inside the isolated environment.

### Step 4: Install Dependencies

With the virtual environment active, install all required libraries using the `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### Step 5: Verify the Installation

To ensure that isolation is working correctly and you are not installing dependencies into the global system, verify the path of the executable binary:

* **macOS:**
```bash
which pip

```

**Expected Result:** The returned path must point directly to your project's folder (e.g., `/path/to/your/project/devbrain-embeddings/venv/.../pip`), never to the system directories like `/usr/bin/`.

---

## 🔬 Running Academic Labs

Because this project follows a structured package architecture, all laboratory scripts live inside the `src/labs/` directory. To prevent Python from throwing a `ModuleNotFoundError` due to absolute imports resolution, **never execute labs directly from their subfolders**.

Always run them from the **root directory** of the project using Python's module flag (`-m`):

### Lab 03: Cosine Similarity
To run the cosine similarity function:
```bash
# Ensure you are at the project root and venv is active
python -m src.labs.lab02.test_cosine_similarity
```

### Lab 03: Embedding Generation & Cosine Similarity
To run the first semantic similarity simulation and inspect the raw embeddings:
```bash
# Ensure you are at the project root and venv is active
python -m src.labs.lab03.run_cli
```

*Why this way?* Using `python -m` injects the current working directory into Python's `sys.path`, allowing the lab scripts to properly resolve the `src.core.*` imports under the hood.

#### 🧠 Lab 03: Understanding the Output & Vector Anatomy

When you execute Lab 03, the script will intercept and print a sample slice of the raw embeddings before calculating the cosine similarity. This section serves as a technical breakdown of how to interpret that raw data:

##### 1. Array Type (`<class 'numpy.ndarray'>`)
The embedding engine does not return standard Python lists. It outputs native NumPy arrays. In backend engineering, this is crucial for performance: NumPy stores data in contiguous memory blocks and delegates mathematical calculations to highly optimized C subroutines, avoiding the heavy overhead of native Python loops.

##### 2. Vector Shape & Fixed Dimensionality (`Shape: (384,)`)
No matter if your input text is a single word ("Docker") or a 50-word technical paragraph, **the output vector will always have a fixed size of 384 dimensions** (when using `all-MiniLM-L6-v2`). 
* In a 2D space, coordinates are `(X, Y)`. 
* In this semantic space, every text is mapped to exactly 384 coordinates. Each float represents the mathematical intensity or activation of an abstract semantic feature learned by the model during its training.

##### 3. Data Type Precision (`dtype: float32`)
The weights and coordinates are explicitly cast to 32-bit floating-point numbers (`float32`). 
* *Production impact:* Standard Python floats use 64 bits (`float64`). Forcing `float32` cuts the memory footprint and disk storage exactly **in half** without any noticeable loss in semantic retrieval precision.
* *Memory calculation:* 384 dimensions $\times$ 4 bytes (`float32`) = **1,536 bytes (~1.5 KB) per text chunk**. You can use this benchmark to estimate the RAM and storage scaling requirements for large-scale production indexing.
---

## 🛑 Deactivating the Environment

When you are done working on the project and wish to return to your terminal's global environment, simply run:

```bash
deactivate

```

This will instantly restore your system's original environment variables.
