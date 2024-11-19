# Frozen Lake Deep Q-Learning Implementation

An implementation of Deep Q-Learning (DQN) for solving the FrozenLake environment from Gymnasium. This project demonstrates a complete DQN pipeline with experience replay, target networks, and various training utilities.

## Project Structure
```
frozen-lake-dql/
├── frozen_lake_dql/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── dqn_agent.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── dqn_model.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── memory.py
│   │   └── visualization.py
│   ├── __init__.py
│   └── main.py
├── tests/
├── notebooks/
├── models/
├── pyproject.toml
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.10 or higher
- Poetry (Python dependency management tool)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/frozen-lake-dql.git
cd frozen-lake-dql
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Project Components

### 1. Deep Q-Network Model (`models/dqn_model.py`)
The core neural network architecture for the DQN agent.

#### Key Components:
- `DQN` class: A PyTorch neural network implementation
  - Architecture: Input layer → Hidden layer (configurable size) → Output layer
  - Input: State representation
  - Output: Q-values for each possible action
  - Activation: ReLU for hidden layer
  - Methods:
    - `forward()`: Forward pass through the network
    - `save()`: Save model weights
    - `load()`: Load model weights

### 2. DQN Agent (`agents/dqn_agent.py`)
The main reinforcement learning agent implementing the DQN algorithm.

#### Key Features:
- `DQNAgent` class:
  - Experience Replay: Stores and samples past experiences
  - Target Network: Stabilizes training by providing fixed Q-value targets
  - ε-greedy Exploration: Balances exploration and exploitation
  
#### Key Methods:
- `select_action()`: Choose actions using ε-greedy policy
- `step()`: Process environment step and store experience
- `_optimize_model()`: Update network weights using sampled experiences
- Memory management and model persistence

### 3. Training Framework (`training/trainer.py`)
Manages the training process and interaction between agent and environment.

#### `FrozenLakeTrainer` class:
- Environment setup and management
- Training loop implementation
- Evaluation procedures
- Progress tracking and logging

#### Key Features:
- Episode-based training
- Periodic evaluation
- Performance metrics tracking
- Environment rendering options
- Model checkpointing

### 4. Utilities (`utils/`)

#### a. Configuration (`config.py`)
Defines configuration dataclasses:
- `DQNConfig`: Agent and model hyperparameters
  - Network architecture
  - Learning parameters
  - Exploration settings
- `TrainingConfig`: Training process settings
  - Episode counts
  - Environment parameters
  - Logging intervals

#### b. Memory Management (`memory.py`)
Implements experience replay functionality:
- `ReplayMemory` class:
  - Fixed-size circular buffer
  - Random experience sampling
  - Efficient memory management
  - Transition storage and retrieval

#### c. Visualization (`visualization.py`)
Provides visualization tools for analysis:
- Training metrics plotting
- Q-value heatmaps
- Episode visualization
- Performance analytics

### 5. Main Script (`main.py`)
Entry point for training and evaluation:
- Command-line interface
- Experiment management
- Training workflow
- Result visualization and saving

## Usage

### Basic Training
```bash
python -m frozen_lake_dql.main
```

### Custom Training Configuration
```bash
python -m frozen_lake_dql.main \
    --episodes 2000 \
    --hidden-size 128 \
    --lr 0.0005 \
    --slippery
```

### Evaluation
```bash
python -m frozen_lake_dql.main \
    --load-model path/to/model.pt \
    --eval-episodes 100 \
    --render-eval
```

## Implementation Details

### Deep Q-Learning Algorithm
1. **State Processing**
   - One-hot encoding of discrete states
   - Normalization and preprocessing
   - Tensor conversion for neural network input

2. **Action Selection**
   - ε-greedy policy with decay
   - Balance between exploration and exploitation
   - Action space: Left, Down, Right, Up

3. **Learning Process**
   - Experience collection and storage
   - Batch sampling and replay
   - TD-learning updates
   - Target network synchronization

4. **Optimization**
   - Adam optimizer
   - MSE loss function
   - Gradient clipping
   - Learning rate scheduling

### Training Pipeline
1. **Episode Loop**
   - Environment reset
   - State observation
   - Action selection and execution
   - Reward collection
   - Experience storage

2. **Optimization Loop**
   - Batch sampling
   - Forward pass
   - Loss calculation
   - Backpropagation
   - Network update

3. **Evaluation**
   - Periodic policy assessment
   - Performance metrics
   - Visualization generation

### Performance Monitoring
1. **Metrics Tracked**
   - Episode rewards
   - Step counts
   - Exploration rate
   - Learning progress
   - Q-value statistics

2. **Visualization Tools**
   - Training curves
   - Q-value heatmaps
   - Episode replays
   - Performance analytics

## Project Organization
- `frozen_lake_dql/`: Core package
- `tests/`: Unit tests
- `notebooks/`: Experimental notebooks
- `models/`: Saved model weights
- `experiments/`: Training runs and results

## Development Tools
- Poetry for dependency management
- Pytest for testing
- Black for code formatting
- Flake8 for linting
- isort for import sorting
