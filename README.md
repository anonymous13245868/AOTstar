# AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search

This repository contains the implementation of AOT*, a novel approach for retrosynthesis planning that combines AND-OR tree search with LLM-generated pathways.

## Overview

AOT* addresses the computational cost-efficiency challenges in retrosynthesis planning by integrating LLM-generated routes into an AND-OR tree structure, enabling efficient exploration of synthesis routes.

## Repository Structure

```
.
├── tree_nodes.py           # AND-OR tree node definitions and data structures
├── prompts.py              # LLM prompt templates for route generation
├── llm_tree_optimizer.py   # Core AOT* algorithm implementation
├── optimizer.py            # Base optimizer class and Oracle scoring
├── data_loader.py          # Dataset and inventory loading utilities
├── parallel_search.py      # Multi-threaded batch search execution
├── utils.py                # Helper functions for SMILES processing
├── main.py                 # Main entry point and argument parsing
├── scscore/                # SCScore model (clone required)
└── dataset/                # Data files (download required)
    ├── inventory.pkl
    ├── routes_train.pkl
    ├── routes_val.pkl
    └── ...
```

## Installation

### Requirements
```bash
pip install rdkit openai rdchiral
pip install numpy torch
pip install PyTDC PyYAML selfies
pip install "syntheseus[all]"
```

### Data Preparation

1. **Download the dataset** from this **anonymous** link:
   - [Dataset (Anonymous Dropbox Link)](https://www.dropbox.com/scl/fi/ukywjilefoqcgl2vbk9bo/dataset.zip?rlkey=36rbxckdhn2ot6uj88m09xn7a&st=ke62cj36&dl=0)
   - Extract the contents to the `dataset/` directory in your project root

2. **Set up SCScore for synthesis complexity evaluation:**
   ```bash
   # Clone the SCScore repository into your project directory
   git clone https://github.com/connorcoley/scscore.git scscore
   ```
   
   The SCScore model is used to evaluate molecular synthesis complexity. Our scripts automatically load the model weights from:
   ```
   scscore/models/full_reaxys_model_2048bool/model.ckpt-10654.as_numpy.json.gz
   ```
   
   The model will be automatically initialized in `optimizer.py` (line 104):
   ```python
   self.sc_Oracle.restore(
       os.path.join('scscore', 'models', 'full_reaxys_model_2048bool', 
                    'model.ckpt-10654.as_numpy.json.gz'), 
       FP_len=2048
   )
   ```

3. **Configure paths** - Update the following paths in your code to match your local setup:

   **In `optimizer.py`:**
   ```python
   # Line 153 - Update rule-based set path if scscore is in different location
   args.rule_based_set_path = 'scscore/data/data_processed.csv'
   ```

   **In `data_loader.py`:**
   ```python
   # Lines 17-22 - Update dataset paths to absolute paths if needed
   dataset_paths = {
       'pistachio_hard': "/path/to/your/dataset/pistachio_hard_targets.txt",
       'pistachio_reachable': "/path/to/your/dataset/pistachio_reachable_targets.txt",
       # ... update all paths
   }
   ```

## Configuration

### API Setup

Configure your LLM API credentials in `optimizer.py` (around line 1050):

```python
client = OpenAI(
    api_key="your_api_key_here",
    base_url="your_api_endpoint_here"
)
```

### Available Parameters

- `--dataset`: Target dataset (`pistachio_hard`, `pistachio_reachable`, `uspto_190`, `pistachio_easy`)
- `--api_model`: LLM model to use (e.g., `deepseek`, `gpt-4o`)
- `--threads`: Number of parallel search threads (default: 5)
- `--max_oracle_calls`: Maximum oracle evaluations per target (default: 1000)
- `--expansion`: Number of routes generated per expansion (default: 1)
- `--start_idx`, `--end_idx`: Molecule index range for batch processing
- `--api_temperature`: Temperature for LLM generation (default: 0.7)

## Running Experiments

### Single Molecule Test
```bash
python main.py --single_test "CC(C)c1ccc(cc1)C(=O)O" --api_model gpt-4o
```

### Batch Evaluation
```bash
# Run on first 100 molecules of pistachio_hard dataset
python main.py --dataset pistachio_hard --threads 5 --start_idx 0 --end_idx 100
```

### Full Dataset Evaluation
```bash
# Evaluate complete dataset with parallel processing
python main.py --dataset pistachio_reachable --threads 10 --api_model deepseek
```

### Test Mode (with small inventory)
```bash
python main.py --test_mode --single_test "CCO"
```

## Output Format

Results are saved in `{logs_path}/{dataset}/{model}/`:
- `search_config.json`: Search configuration and parameters
- `result_{idx:05d}.json`: Individual molecule results


Each result contains:
- Solution tree structure (if found)
- Search statistics (iterations, nodes explored)

## Algorithm Details

AOT* operates through:
1. **Root Expansion**: LLM generates initial synthesis routes
2. **Selection**: UCB-based selection of promising AND nodes
3. **Expansion**: Generate new routes for unsolved reactants
4. **Evaluation**: Scoring via incorporating SC score + availability score
5. **Backpropagation**: Update node statistics through tree

The search continues until a complete solution is found or iteration limits are reached.