**Target Subreddits to Post In:** r/artificial, r/MachineLearning

**Title:**
How do you write a math formula to teach an AI what "polite" means?

**Body:**
Everyone talks about **RLHF** (Reinforcement Learning from Human Feedback) for ChatGPT, but how does it actually work? You can't code a mathematical formula for a "good poem" or a "polite answer". 

Instead, we train a Reward Model based entirely on vibes. We show a human two answers, and the human clicks the better one. Using the **Bradley-Terry model**, the AI maps those binary clicks into a smooth mathematical landscape of "Human Taste". Then we unleash the AI (usually via PPO) to maximize that taste.

To visualize this physically instead of through text, I built a simulated robot arm that learns to draw calligraphy based purely on human clicks. 

I've open-sourced the whole thing (along with 10 other interactive RL projects) on GitHub. If you want an intuitive, visual understanding of RLHF under the hood with PyTorch code, check out the repo: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `rlhf_calligraphy_cover.png` to your Reddit post!)*