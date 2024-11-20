# Implementing Deep Reinforcement Learning (Deep Q-Learning) for the Frozen Lake Environment ❄️ with PyTorch 🔥

To bring the concepts of Deep Q-Learning to life, I've developed a Python project that implements the algorithm using PyTorch. This implementation focuses on solving the Frozen Lake environment from Gymnasium, showcasing how reinforcement learning agents can learn optimal policies in uncertain and dynamic settings. The project is organized into modular components to enhance readability, maintainability, and scalability, making it easier for others to understand and build upon the work.

## Project Structure

The project follows a structured layout that separates different aspects of the implementation into dedicated modules. This organization promotes clean code practices and facilitates easier navigation through the codebase. Here's an overview of the project's directory structure:

```
frozen-lake-dql/
├── frozen_lake_dql/
│   ├── agents/
│   │   └── dqn_agent.py
│   ├── models/
│   │   └── dqn_model.py
│   ├── training/
│   │   └── trainer.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── memory.py
│   │   └── visualization.py
│   └── main.py
├── tests/
├── notebooks/
├── models/
├── pyproject.toml
└── README.md
```

- **`frozen_lake_dql/`**: The main package containing all source code.
    - **`agents/`**: Houses the implementation of the reinforcement learning agents.
        - **`dqn_agent.py`**: Implements the Deep Q-Network (DQN) agent, responsible for interacting with the environment and learning from experiences.
    - **`models/`**: Contains neural network architectures used by agents.
        - **`dqn_model.py`**: Defines the neural network model that approximates the Q-value function.
    - **`training/`**: Manages the training process and agent-environment interactions.
        - **`trainer.py`**: Provides a training loop, evaluation procedures, and progress tracking.
    - **`utils/`**: Includes utility modules for configuration, memory management, and visualization.
        - **`config.py`**: Specifies configuration settings and hyperparameters.
        - **`memory.py`**: Implements experience replay memory for storing past experiences.
        - **`visualization.py`**: Offers functions to visualize training progress and results.
    - **`main.py`**: The entry point script for running training and evaluation.
- **`tests/`**: Contains unit tests to ensure code reliability.
- **`notebooks/`**: Jupyter notebooks for experimentation and detailed analysis.
- **`models/`**: Directory to save trained model weights.
- **`pyproject.toml`**: Configuration file for Poetry, managing dependencies and virtual environments.
- **`README.md`**: Provides documentation and instructions for the project.

## Setting Up the Environment with Poetry

To ensure a consistent and reproducible development environment, the project utilizes **Poetry** for dependency management and virtual environment handling. Poetry simplifies the setup process by automating the installation of all required packages and managing Python versions.

### Prerequisites

- **Python 3.10** or higher
- **Poetry** installed on your system

### Installation Steps

1. **Clone the Repository:**
    
    Open your terminal and clone the repository using:
    
    ```bash
    git clone <https://github.com/yourusername/frozen-lake-dql.git>
    cd frozen-lake-dql
    ```
    
2. **Install Dependencies with Poetry:**
    
    Ensure Poetry is installed. If not, follow the [Poetry installation guide](https://python-poetry.org/docs/#installation).
    
    Install the project dependencies:
    
    ```bash
    poetry install
    ```
    
3. **Activate the Virtual Environment:**
    
    Enter the virtual environment managed by Poetry:
    
    ```bash
    poetry shell
    ```
    
    This command ensures that you are using the isolated environment with all the correct dependencies.
    

## Running the Project

With the environment set up, you can now train and evaluate the DQN agent.

### Basic Training

To start training the agent with default settings, run:

```bash
python -m frozen_lake_dql.main
```

### Custom Training Configuration

You can customize the training process by specifying arguments:

```bash
python -m frozen_lake_dql.main \\
    --episodes 2000 \\
    --hidden-size 128 \\
    --lr 0.0005 \\
    --slippery
```

- `-episodes`: Number of training episodes.
- `-hidden-size`: Number of neurons in the hidden layer of the neural network.
- `-lr`: Learning rate for the optimizer.
- `-slippery`: Whether to use the slippery version of the Frozen Lake environment.

### Evaluation

To evaluate a trained model:

```bash
python -m frozen_lake_dql.main \\
    --load-model path/to/model.pt \\
    --eval-episodes 100 \\
    --render-eval
```

- `-load-model`: Path to the saved model file.
- `-eval-episodes`: Number of episodes to run for evaluation.
- `-render-eval`: Render the environment during evaluation.

## Module Overview

The project is divided into several modules, each responsible for a specific part of the implementation. Below is a high-level explanation of each module and its functionality.

### 1. DQN Model (`models/dqn_model.py`)

This module defines the neural network architecture used by the DQN agent. The `DQN` class inherits from PyTorch's `nn.Module` and represents a feedforward neural network.

- **Input Layer**: Accepts the state representation from the environment. In the Frozen Lake environment, states are one-hot encoded vectors corresponding to the discrete grid positions.
- **Hidden Layer**: A configurable layer that allows you to adjust the network's capacity. It uses the ReLU activation function to introduce non-linearity.
- **Output Layer**: Produces Q-values for each possible action, providing the agent with estimates of the expected rewards for taking each action from the current state.
- **Methods**:
    - `forward()`: Defines the forward pass computation through the network.
    - `save()` and `load()`: Utility functions to persist the model weights for later use.

This modular design allows for easy experimentation with different network architectures by simply adjusting parameters without altering the core codebase.

### 2. DQN Agent (`agents/dqn_agent.py`)

The agent module encapsulates the logic for decision-making and learning from interactions with the environment. The `DQNAgent` class manages several key components:

- **Action Selection**: Implements the epsilon-greedy policy to balance exploration and exploitation. The agent chooses random actions with probability ε and the best-known action with probability 1-ε.
- **Experience Replay**: Stores past experiences in a replay memory to break the correlation between consecutive samples. This approach stabilizes training by providing the agent with a more diverse training dataset.
- **Network Optimization**: Contains methods to sample batches from the replay memory and perform gradient descent updates on the policy network. It also handles the periodic synchronization of the target network.
- **Attributes**:
    - **Policy Network**: The main network that gets updated every training step.
    - **Target Network**: A copy of the policy network used to calculate target Q-values, updated less frequently to provide stability.
    - **Optimizer and Loss Function**: Configured to update the network weights based on the difference between predicted and target Q-values.

By abstracting these functionalities into the `DQNAgent` class, the code remains organized and focused, making it easier to understand the agent's behavior and learning process.

### 3. Trainer (`training/trainer.py`)

The trainer module is responsible for managing the overall training process. The `FrozenLakeTrainer` class orchestrates the interaction between the agent and the environment.

- **Environment Setup**: Initializes the Gymnasium Frozen Lake environment with specified configurations, such as map size and slipperiness.
- **Training Loop**: Iteratively runs episodes where the agent takes actions, receives rewards, and learns from experiences.
- **Evaluation**: Periodically assesses the agent's performance to monitor progress and adjust training parameters if necessary.
- **Logging and Saving**: Keeps track of important metrics like cumulative rewards and steps per episode. It also handles saving the model weights at specified intervals.

The trainer encapsulates the high-level flow of training, allowing for easy adjustments to the training process and ensuring that the main script remains clean and manageable.

### 4. Utilities (`utils/`)

The utilities package includes helper modules that support the main functionality of the project.

### a. Configuration (`utils/config.py`)

This module defines data classes for storing configuration settings and hyperparameters.

- **`DQNConfig`**: Contains parameters related to the agent and neural network, such as learning rate, discount factor, and network architecture.
- **`TrainingConfig`**: Holds settings for the training process, including the number of episodes, maximum steps per episode, evaluation intervals, and model saving paths.

Using data classes for configuration promotes cleaner code and easier management of parameters across different parts of the project.

### b. Memory Management (`utils/memory.py`)

Implements the experience replay mechanism through the `ReplayMemory` class.

- **Experience Storage**: Uses a deque (double-ended queue) to store experiences as tuples of (state, action, next_state, reward, done).
- **Sampling**: Provides a method to randomly sample batches of experiences for training, which helps in breaking the temporal correlations between experiences.
- **Capacity Management**: Maintains a fixed memory size, automatically discarding the oldest experiences when new ones are added beyond the capacity.

This module ensures that the agent can effectively learn from past experiences while maintaining computational efficiency.

### c. Visualization (`utils/visualization.py`)

Offers functions to visualize the agent's training progress and performance.

- **Training Metrics Plotting**: Generates graphs for rewards and steps per episode, allowing for the analysis of learning trends over time.
- **Q-Value Heatmaps**: Visualizes the learned Q-values for different state-action pairs, providing insights into the agent's decision-making process.
- **Episode Visualization**: Creates detailed plots of states, actions, and rewards within a single episode to examine the agent's behavior.

Visualization tools are essential for diagnosing issues, understanding the learning process, and communicating results effectively.

### 5. Main Script (`main.py`)

The `main.py` script serves as the entry point for running the training and evaluation processes.

- **Argument Parsing**: Utilizes the `argparse` library to allow users to specify training parameters and configurations via command-line arguments.
- **Experiment Management**: Sets up directories for saving models and logs, ensuring that results are organized and reproducible.
- **Training Workflow**: Initializes configurations, creates the trainer and agent instances, and starts the training loop.
- **Exception Handling**: Manages interruptions gracefully, saving the model state if the training is stopped prematurely.
- **Result Visualization**: After training, it calls visualization functions to generate plots and saves them in the designated directories.

By handling the execution flow in `main.py`, the project maintains a clear separation between configuration, execution, and implementation, making it user-friendly and easy to extend.

## Conclusion

This project provides a comprehensive implementation of Deep Q-Learning for the Frozen Lake environment, highlighting the practical aspects of building and training a reinforcement learning agent. Through modular design and the use of modern tools like PyTorch and Poetry, the project demonstrates how to structure code for clarity and scalability.

By walking through the project structure and explaining the purpose of each module, we've connected theoretical concepts to tangible code, bridging the gap between understanding and application. Whether you're new to reinforcement learning or looking to deepen your knowledge, this implementation serves as a valuable resource for exploring how agents learn and make decisions in uncertain environments.

The use of visualization tools and detailed configuration options also allows for experimentation and deeper analysis, encouraging further exploration into the nuances of reinforcement learning algorithms.