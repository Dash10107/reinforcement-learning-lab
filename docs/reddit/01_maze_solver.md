**Target Subreddits to Post In:** r/Python, r/learnmachinelearning, r/artificial

**Title:**
Why do video game NPCs still feel so incredibly stupid in 2026?

**Body:**
We have Artificial Intelligence that can paint masterpieces and write complex Python code, yet Skyrim guards still walk repeatedly into walls. Why is navigation so hard? 

It all comes down to the math of **Exploration vs. Exploitation**. 

I wanted to visually see why this happens, so I built a Python simulation comparing two famous Reinforcement Learning algorithms: **Q-Learning** and **SARSA**. The behavioral difference is crazy. 

Q-Learning is an arrogant, optimistic AI. It assumes it will always take the perfect path, so it confidently marches its character right next to the edge of cliffs and walls. 

SARSA is cautious. It accounts for its own random mistakes. In the exact same maze, SARSA learns to take wider, safer paths down the middle of corridors just in case it slips.

I ended up turning this into an interactive visual project. It's part of a massive open-source Reinforcement Learning portfolio I'm building on GitHub. If you want to see the PyTorch code and watch the AI's "brain" light up in real-time, you can check it out here: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `maze_preview.png` to your Reddit post!)*