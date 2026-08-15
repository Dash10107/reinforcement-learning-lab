**Target Subreddits to Post In:** r/programming, r/Python, r/MachineLearning

**Title:**
Why would an AI trained to play chess immediately explode a SpaceX rocket?

**Body:**
Standard AI algorithms (like Deep Q-Networks) are great, but they only output *discrete* actions: Up, Down, Left, Right. 

If you try to land a rocket, you can't just toggle the engine "100% On" or "0% Off". You need precise, *continuous* thrust. The secret to continuous control in robotics is **Soft Actor-Critic (SAC)**. 

Instead of guessing a single action, SAC outputs a Gaussian Distribution (a Mean thrust, and a Standard Deviation of how confident the AI is). 

But the real magic is its **Entropy Bonus**. SAC literally pays the AI a mathematical bonus to try landing in weird, creative ways. It forces the AI to find multiple solutions to the exact same problem, resulting in incredibly robust policies.

I built a physics simulation landing a rocket using PyTorch and SAC to visualize this. It's one of the projects in a massive interactive RL portfolio I've open-sourced on GitHub. If you want to see how the "Entropy Bonus" forces the AI to land creatively, grab the code here: 
👉 **[Open Source Reinforcement Learning Lab](https://github.com/Dash10107/reinforcement-learning-lab)**

*(Optional: Attach `rocket_sac_cover.png` to your Reddit post!)*