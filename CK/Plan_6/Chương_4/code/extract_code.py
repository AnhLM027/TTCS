import os
import json
import glob

def extract_outputs_to_string(outputs):
    """
    Extracts text-based output from notebook cell outputs.
    """
    out_text = []
    for out in outputs:
        if out.get('output_type') == 'stream':
            text = out.get('text', [])
            if isinstance(text, list):
                out_text.extend(text)
            else:
                out_text.append(str(text))
        elif out.get('output_type') in ['execute_result', 'display_data']:
            data = out.get('data', {})
            text = data.get('text/plain', [])
            if isinstance(text, list):
                out_text.extend(text)
            else:
                out_text.append(str(text))
        elif out.get('output_type') == 'error':
            ename = out.get('ename', '')
            evalue = out.get('evalue', '')
            out_text.append(f"{ename}: {evalue}\n")
            
    full_out = "".join(out_text)
    if not full_out.strip():
        return ""
        
    commented_out = ["\n# --- OUTPUT ---\n"]
    for line in full_out.splitlines():
        commented_out.append(f"# {line}\n")
    commented_out.append("# --------------\n")
    return "".join(commented_out)

def extract_code_from_ipynb(ipynb_path, py_path):
    """
    Extracts all code cells from a .ipynb notebook and writes them to a .py file.
    Also extracts text output and formats it as comments.
    """
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        code_cells = []
        cell_count = 0
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell_count += 1
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_str = "".join(source)
                else:
                    source_str = str(source)
                
                outputs = cell.get('outputs', [])
                output_str = extract_outputs_to_string(outputs)
                
                # Only add if the cell is not empty
                if source_str.strip():
                    code_cells.append(f"# %% [cell {cell_count}]\n{source_str}\n{output_str}\n")
        
        # Write extracted code to the .py file
        with open(py_path, 'w', encoding='utf-8') as f:
            f.write(f'# File extracted from: {os.path.basename(ipynb_path)}\n')
            f.write('# Code cells and text outputs\n\n')
            f.writelines(code_cells)
            
        print(f"Successfully extracted: '{os.path.basename(ipynb_path)}' -> '{os.path.basename(py_path)}' ({cell_count} cells)")
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(ipynb_path)}: {e}")
        return False

def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    ipynb_files = glob.glob(os.path.join(dir_path, "*.ipynb"))
    
    if not ipynb_files:
        print(f"No .ipynb files found in: {dir_path}")
        return
        
    print(f"Found {len(ipynb_files)} .ipynb files. Starting extraction...")
    
    success_count = 0
    for ipynb_file in sorted(ipynb_files):
        py_file = ipynb_file.rsplit('.', 1)[0] + ".py"
        if extract_code_from_ipynb(ipynb_file, py_file):
            success_count += 1
            
    print(f"\nExtraction complete! Successfully processed {success_count}/{len(ipynb_files)} files.")

if __name__ == "__main__":
    main()
