# Gemini Code Assistant Context: 自偏移推理训练 (Self-Offset Reasoning Training)

## Project Overview

This project is a proof-of-concept for an innovative machine learning technique called "Self-Offset Reasoning Training." The core idea is to train a model to understand and perform logical transformations by learning from "noisy" but logically equivalent propositions.

Specifically, the system trains a sequence-to-sequence model to convert a proposition in disjunctive normal form (e.g., `(~p | q)`) into its logically equivalent contrapositive form (e.g., `~q -> ~p`). The input is considered "noisy" because `(~p | q)` is a less direct representation of the implication `p -> q`.

The project is written in Python and utilizes `numpy` for a simplified baseline model and `torch` for more advanced implementations. It is structured to support different training environments, including local CPU, CUDA-accelerated GPU, and remote/distributed setups.

**Key Technologies:**
- **Language:** Python 3.9+
- **ML Frameworks:** NumPy, PyTorch
- **Core Libraries:** `tqdm`, `matplotlib`, `pyyaml`, `rich`
- **Environments:** Local (CPU), CUDA, Remote (Cloud)

**Architecture:**
- `src/logic_transformer`: Contains the core logic, including a character-level `Tokenizer`, data loading utilities (`data_utils.py`), and logical transformation rules (`logic_rules.py`).
- `simple_model.py`: A baseline sequence-to-sequence model implemented purely in NumPy for initial validation.
- `breakthrough_training_system_refactored.py`: An advanced and refactored training script that implements modern ML concepts like an Experience Replay Buffer, Adaptive Learning Rate Scheduler, and Curriculum Learning. This appears to be the main training driver.
- `configs/`: JSON files for configuring model hyperparameters, training parameters, and data paths.
- `data/`: Contains JSON/JSONL training and validation datasets with different complexity levels.
- `scripts/`: Contains various scripts for training, evaluation, and data generation.
- `requirements.txt`, `requirements_cuda.txt`, `requirements_remote.txt`: Python dependency files for different target environments.

## Building and Running

### 1. Environment Setup

Install the required dependencies based on your target environment.

**For local CPU training:**
```bash
pip install -r requirements.txt
```

**For NVIDIA GPU (CUDA) training:**
```bash
pip install -r requirements_cuda.txt
```

### 2. Data Generation (Optional)

The repository contains pre-generated datasets. If you need to regenerate them, use the provided scripts. The "robust" version is recommended.
```bash
python generate_robust_dataset.py
```

### 3. Training

The main, recommended training script is `breakthrough_training_system_refactored.py`. It uses a comprehensive configuration and advanced training techniques.

**To run the main training system (on CPU):**
```bash
python breakthrough_training_system_refactored.py
```
*Outputs are saved to `outputs/breakthrough_refactored/`.*

**To run CUDA-accelerated training:**
```bash
python train_cuda.py --auto-batch-size --use-mixed-precision
```

### 4. Evaluation

Use the `clean_evaluation_system.py` script to evaluate a trained model.
```bash
python clean_evaluation_system.py
```

### 5. Running Tests

The project includes unit tests for the core logic rules.
```bash
cd tests
python test_rules.py
```

## Development Conventions

- **Configuration:** The project uses a centralized, nested dictionary structure for configuration, as seen in `create_breakthrough_config()` in the main training script. This is preferred over scattered parameters.
- **Code Style:** The code is generally well-structured with type hints and clear function/class separation. It follows standard Python conventions.
- **Modularity:** Core logic (tokenizer, models, data utilities) is separated into the `src/logic_transformer` package, promoting reusability.
- **Training Strategy:** The `breakthrough_training_system_refactored.py` script indicates a sophisticated training strategy:
    - **Curriculum Learning:** The model is trained on progressively more complex data based on the epoch number.
    - **Experience Replay:** A buffer stores past experiences, which are sampled alongside new data to improve stability and data efficiency.
    - **Target Network:** A separate, slow-updating target model is used to stabilize training, a common technique in reinforcement learning.
    - **Adaptive Learning Rate:** The learning rate is adjusted automatically based on validation loss.
- **Error Handling:** The refactored code shows robust error handling and detailed logging.
