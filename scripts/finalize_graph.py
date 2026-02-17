from src.knowledge_graph.visualization import visualize_knowledge_graph
from src.knowledge_graph.config import load_config
from src.knowledge_graph.entity_standardization import standardize_entities, infer_relationships
import json
import os

def create_final_graph(json_file, output_html):
    print(f"Loading data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Loaded {len(all_results)} raw triples.")
    
    # Load config
    config = load_config('config.toml')
    
    # --- STEP 1: CLEAN & STANDARDIZE ---
    if config.get("standardization", {}).get("enabled", False):
        print("\n=== PHASE 2: ENTITY STANDARDIZATION ===")
        print("Cleaning up duplicate concepts (e.g., merging 'FVG' and 'Fair Value Gap')...")
        try:
            all_results = standardize_entities(all_results, config)
            print(f"Standardization complete. Items remaining: {len(all_results)}")
        except Exception as e:
             print(f"Skipping standardization due to error: {e}")

    # --- STEP 2: INFER CONNECTIONS ---
    if config.get("inference", {}).get("enabled", False):
        print("\n=== PHASE 3: RELATIONSHIP INFERENCE ===")
        print("Connecting hidden dots between concepts...")
        try:
            all_results = infer_relationships(all_results, config)
            print(f"Inference complete. Total items: {len(all_results)}")
        except Exception as e:
            print(f"Skipping inference due to error: {e}")
    
    # --- STEP 3: VISUALIZE ---
    print(f"\nGeneratng FINAL interactive graph to {output_html}...")
    visualize_knowledge_graph(all_results, output_html, config=config)
    print("\nDONE! You can now open the file.")

if __name__ == "__main__":
    create_final_graph('ict_graph_final.json', 'ict_knowledge_brain.html')
