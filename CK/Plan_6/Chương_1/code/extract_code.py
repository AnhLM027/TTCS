import os
import json
import glob

def extract_code_from_ipynb(ipynb_path, py_path):
    """
    Extracts all code cells from a .ipynb notebook and writes them to a .py file.
    Uses VS Code/Jupyter interactive cell separators (# %%) for better readability.
    """
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        code_cells = []
        cell_count = 0
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell_count += 1
                # Join source lines. Source can be a string or a list of strings.
                source = cell.get('source', [])
                if isinstance(source, list):
                    source_str = "".join(source)
                else:
                    source_str = str(source)
                
                # Only add if the cell is not empty
                if source_str.strip():
                    code_cells.append(f"# %% [cell {cell_count}]\n{source_str}\n\n")
        
        # Write extracted code to the .py file
        with open(py_path, 'w', encoding='utf-8') as f:
            # Write a header
            f.write(f'# File extracted from: {os.path.basename(ipynb_path)}\n')
            f.write('# Cleaned code cells only\n\n')
            f.writelines(code_cells)
            
        print(f"Successfully extracted: '{os.path.basename(ipynb_path)}' -> '{os.path.basename(py_path)}' ({cell_count} cells)")
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(ipynb_path)}: {e}")
        return False

def main():
    # Directory containing the notebooks (relative to the script itself)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    
    # Find all .ipynb files in the directory
    ipynb_files = glob.glob(os.path.join(dir_path, "*.ipynb"))
    
    if not ipynb_files:
        print(f"No .ipynb files found in: {dir_path}")
        return
        
    print(f"Found {len(ipynb_files)} .ipynb files. Starting extraction...")
    
    success_count = 0
    for ipynb_file in sorted(ipynb_files):
        # Generate output .py filename by replacing extension
        py_file = ipynb_file.rsplit('.', 1)[0] + ".py"
        
        if extract_code_from_ipynb(ipynb_file, py_file):
            success_count += 1
            
    print(f"\nExtraction complete! Successfully processed {success_count}/{len(ipynb_files)} files.")

if __name__ == "__main__":
    main()
