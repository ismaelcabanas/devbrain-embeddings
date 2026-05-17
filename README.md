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
│   └── api/                 # Service layer and endpoints
│       ├── __init__.py
│       ├── main.py          # FastAPI initialization
│       └── routes.py        # Main endpoints (/api/chat, /api/ingest)
│
└── run_cli.py               # Command-line orchestrator for initial phases

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

## 🛑 Deactivating the Environment

When you are done working on the project and wish to return to your terminal's global environment, simply run:

```bash
deactivate

```

This will instantly restore your system's original environment variables.
