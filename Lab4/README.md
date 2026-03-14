# Lab 4
In this Lab I wrote a new pipeline to compute per species statistics for the Iris dataset.

## Setup

1. Navigate to the Lab4 directory
2. Create a new virtual environment with `python3.11 -m venv .venv`
3. Install dependencies `pip install -r requirements.txt`
4. Open the jupyter notebook and select this environment as the python kernel
5. Run cell in order  

## Pipeline Overview

The pipeline runs six tasks in sequence.

1. **Read CSV** — Reads the Iris dataset csv file downloaded from scikit-learn (150 samples, 3 classes)
2. **Parse CSV** — Extracts classes and features from each sample
2. **Filter rows** — Filters out any rows where parsing failed
3. **Aggregate stats** — Aggregates simple statistics over each class using a beam combine function
4. **Format results** — Formats the statistics
4. **Output results** — Writes the statistics to a file in the outputs directory