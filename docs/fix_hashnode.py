import os
import re

src_dir = r"C:\Users\daksh\OneDrive\Desktop\ReinforcementLearning\docs\articles"
dst_dir = r"C:\Users\daksh\OneDrive\Desktop\ReinforcementLearning\docs\hashnode"

for filename in os.listdir(src_dir):
    if not filename.endswith('.md'):
        continue
        
    filepath = os.path.join(src_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    fm_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        continue
        
    fm = fm_match.group(1)
    
    title = re.search(r'title:\s*"(.*?)"', fm)
    title = title.group(1) if title else "Untitled"
    
    desc = re.search(r'description:\s*"(.*?)"', fm)
    desc = desc.group(1) if desc else ""
    
    cover = re.search(r'(?:cover_image|cover):\s*"(.*?)"', fm)
    cover = cover.group(1) if cover else ""
    
    slug = re.sub(r'^\d+_', '', filename).replace('.md', '').replace('_', '-')
    
    new_fm = f"""---
title: "{title}"
subtitle: "{desc}"
slug: {slug}
tags: machine-learning, python, artificial-intelligence, data-science
cover: "{cover}"
---"""

    new_content = content.replace(fm_match.group(0), new_fm)
    
    # Replace embed links safely
    new_content = re.sub(r'\*\*Reinforcement Learning Lab on GitHub\*\*\{% embed (.*?) %\}', r'**[Reinforcement Learning Lab on GitHub](\1)**', new_content)
    
    dst_path = os.path.join(dst_dir, filename)
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

print("Finished formatting for Hashnode!")
