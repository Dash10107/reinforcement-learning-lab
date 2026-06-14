# Contributing to the Reinforcement Learning Portfolio

Thank you for showing interest in contributing to the Reinforcement Learning Portfolio! We want this to be the most beginner-friendly and educational repository for reinforcement learning on GitHub, and your contributions help make that possible.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behaviour to `dakshcjain@gmail.com`.

---

## How Can I Contribute?

### 1. Propose a New RL Project
Have an interesting environment or algorithm you want to add? We welcome projects that:
*   Are highly visual and interactive (using Gradio or similar).
*   Target a specific educational theme (e.g. exploration strategies, continuous control, hierarchical RL, offline RL).
*   Are clearly documented and well-commented.

### 2. Improve Existing Code
*   Optimise training efficiency of the agents.
*   Enhance visual elements or charts in the dashboards.
*   Resolve compatibility issues with new versions of PyTorch, Gymnasium, or Gradio.
*   Fix bugs or add test cases.

### 3. Polish Documentation
*   Fix typos or improve wording in project READMEs.
*   Enhance `CONCEPTS.md` or `GETTING_STARTED.md` with better explanations.
*   Add more interactive "Things to Try" experiments.

---

## Local Development Setup

To set up a local development environment, follow these steps:

1.  **Fork and Clone the Repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/rl-portfolio.git
    cd rl-portfolio
    ```

2.  **Set Up the Environment**
    Create and activate a virtual environment, then install the dependencies:
    *   **Windows (PowerShell)**:
        ```powershell
        .\setup_env.ps1
        .venv\Scripts\activate
        ```
    *   **Mac / Linux**:
        ```bash
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        ```

3.  **Run formatting and lint checks**
    We use Ruff to maintain consistent style standards. Make sure your changes comply:
    ```bash
    pip install ruff
    ruff check .
    ruff format .
    ```

---

## Creating a Pull Request

When submitting a Pull Request (PR), please follow these guidelines:

*   **Branch Naming**: Use descriptive branch names like `feature/new-bandit-algorithm` or `fix/sac-lunar-lander-imports`.
*   **Keep PRs Focused**: Try to address a single issue or implement a single feature per PR.
*   **Update READMEs**: If you change an algorithm's hyperparameters, state/action space, or UI, update the corresponding project's README.
*   **Test Your Changes**: Run your web application locally to ensure it starts, trains correctly, and renders all visual elements without errors.
*   **Use the Template**: Complete the checklist in our Pull Request Template.
