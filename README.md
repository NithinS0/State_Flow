# FlowState-RL Adaptive Difficulty

FlowState-RL is an AI-driven adaptive difficulty system for a 2D shooter game. It utilizes Reinforcement Learning (PPO via Stable-Baselines3) to dynamically adjust the game's difficulty in real-time, aiming to keep the player in a state of "Flow" (neither bored nor frustrated).

## Features
- **Adaptive Difficulty:** The game adjusts enemy speed, spawn rates, and health dynamically to match player skill.
- **PPO Agent:** An intelligent agent trained using Proximal Policy Optimization adjusts the parameters in real-time.
- **Live Metrics Dashboard:** A Flask-based web dashboard that provides real-time visualizations of player state (Flow, Bored, Anxious), performance, and difficulty metrics.
- **Pygame Client:** A 2D top-down shooter used as the testbed environment.

## Project Structure
- `flowstate_rl/game`: Contains the Pygame environment (`main.py`), training scripts (`train.py`), and evaluation code.
- `flowstate_rl/dashboard`: Contains the Flask server (`server.py`) and HTML/JS frontend to monitor the game.
- `flowstate_rl/environment`: Houses persona definitions and simulated environments for training the agent.
- `flowstate_rl/agent`: Contains the PPO agent wrapper and RL configurations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NithinS0/State_Flow.git
   cd State_Flow
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have Pygame, Flask, Flask-CORS, stable-baselines3, and torch installed).*

## Usage

### Running the Dashboard
To monitor the game in real-time, start the Flask dashboard:
```bash
python -m flowstate_rl.dashboard.server
```
Then open your browser and navigate to `http://localhost:5000`.

### Running the Game
To play the game and have the RL agent adapt to your skill level:
```bash
python -m flowstate_rl.game.main
```

### Training the Agent
To train the PPO agent from scratch using simulated personas (expert, average, beginner):
```bash
python -m flowstate_rl.game.train --persona all
```

## How It Works
1. **Metrics Collection:** The game continuously tracks player performance (kills, damage taken, accuracy, etc.).
2. **State Evaluation:** The metrics determine the player's current cognitive state (Flow, Boredom, or Frustration).
3. **Agent Action:** The RL agent observes the player state and game metrics and outputs difficulty adjustments (modifying spawn rates, enemy speed, etc.).
4. **Live Visualization:** The dashboard reads the outputted `data/live_metrics.json` and updates the UI.

## License
MIT License
