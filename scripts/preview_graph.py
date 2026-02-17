from src.knowledge_graph.visualization import visualize_knowledge_graph
from src.knowledge_graph.config import load_config
import json
import os

def create_preview(json_file, output_html):
    print(f"Loading data from {json_file}...")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    print(f"Data loaded: {len(data)} triples found.")
    
    # Load default config
    config = load_config('config.toml')
    
    # Generate visualization
    print(f"Generating preview graph to {output_html}...")
    visualize_knowledge_graph(data, output_html, config=config)
    print("Done!")

if __name__ == "__main__":
    create_preview('ict_graph_preview.json', 'ict_graph_preview.html')
