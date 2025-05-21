Safe Multi-Agent Reinforcement Learning (MARL) Algorithm
This repository hosts the implementation of a novel Safe Multi-Agent Reinforcement Learning (MARL) algorithm, developed as part of a Master of Technology minor project in Robotics at the Indian Institute of Technology Delhi.

📄 Abstract
Multi-agent reinforcement learning (MARL) faces significant challenges in safety-critical applications. This project introduces a new algorithm that integrates a deep policy gradient optimization technique with a Lagrangian multiplier framework to enable agents to learn cooperative strategies while adhering to constraints and avoiding catastrophic outcomes. Our core innovation lies in effectively balancing the competing objectives of maximizing task rewards and minimizing cost accumulation, dynamically adjusting the influence of the cost constraint during learning. This approach aims to yield policies with enhanced safety characteristics, making them more suitable for real-world deployment.

✨ Key Features
Constrained Policy Optimization: Leverages a Lagrangian multiplier framework to explicitly manage safety constraints during the learning process.

Deep Policy Gradient Integration: Combines deep policy gradient methods for effective learning in complex, high-dimensional state and action spaces.

Decentralized Execution with Centralized Training (Inspired by MADDPG): Designed for applicability in scenarios with limited communication or partial observability.

Reward-Cost Trade-off Balance: Achieves a balance between maximizing task rewards and minimizing accumulated costs, leading to safer and performant policies.

PyTorch Implementation: Built using Python with PyTorch for efficient deep learning operations.

🚀 Methodology Overview
The algorithm formulates the problem as a constrained Markov game, where the objective is to maximize the expected return while satisfying safety constraints defined by expected costs for each agent. Lagrangian multipliers are incorporated into the optimization process to enable a dynamic trade-off between reward maximization and cost minimization. The approach utilizes an actor-critic architecture with target networks for stability.

🧪 Experimental Setup & Results
The algorithm's performance was evaluated in a modified PettingZoo multi-particle environment, adapted to include a cost function penalizing collisions between agents.

Key Findings:

The algorithm successfully learns policies that respect cost constraints while maintaining high task performance.

Agents adapt their behavior in response to varying cost constraints, adopting more conservative strategies as constraints tighten.

Quantitative results demonstrate a clear and effective trade-off between reward and cost, validating the algorithm's ability to address safe MARL challenges.

The use of a Q-network for cost learning significantly improves overall performance and reward levels compared to baselines.

🛠️ Installation
To set up the environment and run the code, follow these steps:

Clone the repository:

git clone https://github.com/DhruvaBhardwaj404/Constrained-Multi-Agent-Deep-Policy-Gradient-Optimization-Algorithm.git
cd Constrained-Multi-Agent-Deep-Policy-Gradient-Optimization-Algorithm

Create a Conda environment (recommended):

conda create -n safe_marl python=3.9 # Or your preferred Python version
conda activate safe_marl

Install dependencies:

pip install -r requirements.txt # (You might need to create this file based on your project's dependencies like torch, numpy, pettingzoo, etc.)

Note: Ensure you have PyTorch installed, compatible with your system's CUDA version if you plan to use a GPU.

🏃 Usage

To train the algorithm, you can run the main training script:

python main.py M/C/CNQ 
M => simple MADPG (Adds cost to the reward for each agent)
C => CMDAPG with Q-Network for Learning Cost Function
CNQ => CMADPG without Q-Network for Learning Cost Function

To visualize the training progress using TensorBoard:

tensorboard --logdir runs/

🔮 Future Work
Based on the project's objectives and findings, potential future research directions include:

Real-World Validation: Deploying the algorithm on physical robotic systems.

Extension to More Complex Environments: Adapting the algorithm for continuous state/action spaces, partial observability, or non-stationary agents.

Scalability Improvements: Investigating decentralized training methods or hierarchical control structures for larger agent numbers.

Advanced Optimization Techniques: Incorporating trust region methods, proximal policy optimization, or evolutionary algorithms.

Probabilistic Safety Guarantees: Developing more sophisticated techniques for quantifying uncertainty in safety assessments.

Multi-Constraint Handling: Exploring methods for balancing multiple simultaneous safety constraints.

Safe Exploration: Developing safer exploration strategies that minimize constraint violations.

Comparison with other Safe RL methods: Benchmarking against other state-of-the-art Safe RL algorithms (e.g., Barrier Functions, Lyapunov methods).

🤝 Acknowledgements
This project was completed under the supervision of Prof. James Arambam Singh at the Indian Institute of Technology Delhi.

📝 License
This project is licensed under the MIT License - see the LICENSE file for details. 
