#!/bin/bash
cd "$(dirname "$0")"

echo "=== ICT AI Knowledge Graph Setup ==="
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies (this may take a minute)..."
pip install -r requirements.txt

echo "Combining your ICT data..."
python3 prepare_data.py

echo ""
echo "=== SETUP COMPLETE ==="
echo "Next Steps:"
echo "1. Edit config.toml in this folder."
echo "   - If you have an OpenAI Key, uncomment Option 1 and paste your key."
echo "   - If you want to run locally for free, install Ollama (ollama.ai) and uncomment Option 2."
echo ""
echo "2. Run the generator:"
echo "   source venv/bin/activate"
echo "   python generate-graph.py --input data/ict_knowledge_combined.txt --output ict_graph.html"
echo ""
