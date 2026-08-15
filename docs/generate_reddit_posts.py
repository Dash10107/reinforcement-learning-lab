import os

os.makedirs(r'C:\Users\daksh\OneDrive\Desktop\ReinforcementLearning\docs\reddit', exist_ok=True)

posts = {
    "01_maze_solver.md": """**Target Subreddits to Post In:** r/Python, r/learnmachinelearning, r/artificial

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

*(Optional: Attach `maze_preview.png` to your Reddit post!)*""",

    "02_bandit_optimizer.md": """**Target Subreddits to Post In:** r/SideProject, r/SaaS, r/MachineLearning

**Title:**
What if standard A/B testing is actually costing your website thousands of dollars?

**Body:**
Everyone tells you to run A/B tests. But here is the dirty secret: if you test a "Start Free Trial" button against a "Buy Now for $100" button, the free trial will ALWAYS win the click-through-rate. 

If your algorithm optimizes purely for clicks, it routes all your traffic to the free trial. You get clicks, but you bleed massive revenue. 

How do companies like Netflix avoid this? They use **Multi-Armed Bandits** with **Thompson Sampling**. Instead of looking at a flat average, it uses a Beta Distribution. It learns to completely ignore the high-click "Free Trial" banner and physically shifts its probability curve to chase the rare—but massive—payouts of the "Buy Now" button.

I built an open-source Python simulator to visualize exactly how Thompson Sampling shifts its probabilities to chase Expected Value instead of clicks. It's part of an interactive RL portfolio I'm open-sourcing on GitHub. If you want to play with the live dashboard, grab the code here: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `bandit_cover.png` to your Reddit post!)*""",

    "03_warehouse_marl.md": """**Target Subreddits to Post In:** r/MachineLearning, r/artificial, r/robotics

**Title:**
How does Amazon keep 1,000 warehouse robots from crashing into each other every 5 seconds?

**Body:**
If you try to train 50 AI robots using standard reinforcement learning, it's a complete disaster. 

Why? Because the environment is "non-stationary". Every robot is learning at the exact same time. If one robot finds a slightly better path, it instantly ruins the math for the other 49. They take one wrong step and mathematically "unlearn" everything.

The industry fix for this in Multi-Agent Reinforcement Learning (MARL) is **Proximal Policy Optimization (PPO)**. 

PPO uses a brilliant clipping mechanism that mathematically forces the AI to only take tiny, conservative learning steps. It physically prevents the neural network's policy from drifting too far from what it already knows works.

To understand the math behind this, I built a PyTorch swarm simulation. It's part of an interactive 11-project Reinforcement Learning portfolio I'm open-sourcing on GitHub. If you want to see how PPO actually forces the swarm to synchronize without crashing, check out the repo: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `warehouse_marl_cover.png` to your Reddit post!)*""",

    "04_rocket_lander.md": """**Target Subreddits to Post In:** r/programming, r/Python, r/MachineLearning

**Title:**
Why would an AI trained to play chess immediately explode a SpaceX rocket?

**Body:**
Standard AI algorithms (like Deep Q-Networks) are great, but they only output *discrete* actions: Up, Down, Left, Right. 

If you try to land a rocket, you can't just toggle the engine "100% On" or "0% Off". You need precise, *continuous* thrust. The secret to continuous control in robotics is **Soft Actor-Critic (SAC)**. 

Instead of guessing a single action, SAC outputs a Gaussian Distribution (a Mean thrust, and a Standard Deviation of how confident the AI is). 

But the real magic is its **Entropy Bonus**. SAC literally pays the AI a mathematical bonus to try landing in weird, creative ways. It forces the AI to find multiple solutions to the exact same problem, resulting in incredibly robust policies.

I built a physics simulation landing a rocket using PyTorch and SAC to visualize this. It's one of the projects in a massive interactive RL portfolio I've open-sourced on GitHub. If you want to see how the "Entropy Bonus" forces the AI to land creatively, grab the code here: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `rocket_sac_cover.png` to your Reddit post!)*""",

    "05_smart_grid.md": """**Target Subreddits to Post In:** r/learnmachinelearning, r/Python

**Title:**
How do algorithms mathematically know the difference between "quick cash" and "long-term wealth"?

**Body:**
The **Bellman Equation** is the backbone of modern AI, but textbooks make it look terrifying. In reality, it's just a mathematical scale weighing immediate rewards against future potential.

To visualize this, imagine an AI managing a smart battery on the power grid. It has to decide: Do I sell my power now for $5, or hold it because prices might hit $20 tonight? 

By tweaking just one variable (the discount factor or Gamma), the AI transitions from an aggressive day trader into a patient, long-term investor.

I built an interactive smart grid trading bot to watch this equation in real-time. It's part of a full Reinforcement Learning curriculum I've put on GitHub. If you are struggling to understand how Deep Q-Learning actually works under the hood, I broke down the PyTorch code here: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `smart_grid_cover.png` to your Reddit post!)*""",

    "06_digital_calligrapher.md": """**Target Subreddits to Post In:** r/artificial, r/MachineLearning

**Title:**
How do you write a math formula to teach an AI what "polite" means?

**Body:**
Everyone talks about **RLHF** (Reinforcement Learning from Human Feedback) for ChatGPT, but how does it actually work? You can't code a mathematical formula for a "good poem" or a "polite answer". 

Instead, we train a Reward Model based entirely on vibes. We show a human two answers, and the human clicks the better one. Using the **Bradley-Terry model**, the AI maps those binary clicks into a smooth mathematical landscape of "Human Taste". Then we unleash the AI (usually via PPO) to maximize that taste.

To visualize this physically instead of through text, I built a simulated robot arm that learns to draw calligraphy based purely on human clicks. 

I've open-sourced the whole thing (along with 10 other interactive RL projects) on GitHub. If you want an intuitive, visual understanding of RLHF under the hood with PyTorch code, check out the repo: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `rlhf_calligraphy_cover.png` to your Reddit post!)*"""
}

for filename, content in posts.items():
    with open(os.path.join(r'C:\Users\daksh\OneDrive\Desktop\ReinforcementLearning\docs\reddit', filename), 'w', encoding='utf-8') as f:
        f.write(content)

print("Regenerated 6 reddit posts in docs/reddit/ without the visible search keywords block.")
