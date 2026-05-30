import os
import re
import glob

files = glob.glob('/home/llm/AnhLM/TTCS/CK/Plan_6/Chương_2/docs/sub_*.tex') + \
        glob.glob('/home/llm/AnhLM/TTCS/CK/Plan_6/Chương_3/docs/sub_*.tex')

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace \paragraph{...} with \vspace{10pt}\noindent\textbf{...}
    # We use regex to match \paragraph{...} safely
    # This matches \paragraph{text}
    new_content = re.sub(r'\\paragraph\{([^}]+)\}', r'\\vspace{10pt}\\noindent\\textbf{\1}', content)
    
    if new_content != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed: {fpath}")
