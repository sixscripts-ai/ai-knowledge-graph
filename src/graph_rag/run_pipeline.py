#!/usr/bin/env python3
"""
ICT GraphRAG Pipeline — Full Runner
=====================================
Runs the complete pipeline:
    1. Load knowledge graph from all sources
    2. Generate fine-tuning training data
    3. Export in multiple formats
    4. Run the logic engine with sample scenarios
    5. (Optional) Start interactive GraphRAG chat

Usage:
    # Full pipeline
    python -m src.graph_rag.run_pipeline
    
    # Just generate training data
    python -m src.graph_rag.run_pipeline --training-only
    
    # Just run logic engine tests
    python -m src.graph_rag.run_pipeline --logic-only
    
    # Interactive GraphRAG chat
    python -m src.graph_rag.run_pipeline --chat
    
    # Fine-tune with unsloth (requires GPU)
    python -m src.graph_rag.run_pipeline --finetune
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.graph_rag.graph_store import ICTGraphStore
from src.graph_rag.training_generator import TrainingDataGenerator
from src.graph_rag.logic_engine import TradeReasoner


def run_graph_store():
    """Phase 1: Load and test the graph store."""
    print("\n" + "=" * 70)
    print("  PHASE 1: Loading Knowledge Graph")
    print("=" * 70)

    store = ICTGraphStore()
    store.load_all()

    stats = store.stats()
    print(f"\nGraph loaded:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Components: {stats['connected_components']}")
    print(f"  Sources: {', '.join(stats['sources'])}")

    print(f"\n  Top relation types:")
    for rel, count in list(stats['relation_types'].items())[:10]:
        print(f"    {rel}: {count}")

    print(f"\n  Node types:")
    for ntype, count in stats['node_types'].items():
        print(f"    {ntype}: {count}")

    # Test queries
    print("\n--- Test Queries ---")
    
    # FVG neighbors
    fvg_neighbors = store.get_neighbors("fair_value_gap")
    print(f"\nFair Value Gap has {len(fvg_neighbors)} direct connections:")
    for n in fvg_neighbors[:5]:
        print(f"  {n['direction']:>3} | {n['relation']:<20} | {n['node']}")

    # Path finding
    path = store.find_path_with_relations("liquidity_sweep", "silver_bullet")
    if path:
        print(f"\nPath: liquidity_sweep → silver_bullet ({len(path)} hops):")
        for step in path:
            print(f"  {step['from']} --[{step['relation']}]--> {step['to']}")
    else:
        print("\nNo direct path from liquidity_sweep to silver_bullet")

    # Models for FVG
    models = store.get_models_for_pattern("fvg")
    print(f"\nModels that use FVG: {[m['model'] for m in models]}")

    # Related concepts
    related = store.get_related_concepts("displacement")
    print(f"\nDisplacement relations:")
    for rel_type, concepts in list(related.items())[:5]:
        print(f"  {rel_type}: {', '.join(concepts[:3])}")

    return store


def run_training_generator(store: ICTGraphStore):
    """Phase 3: Generate training data."""
    print("\n" + "=" * 70)
    print("  PHASE 3: Generating Fine-Tuning Data")
    print("=" * 70)

    gen = TrainingDataGenerator(store)
    total = gen.generate_all()

    print(f"\nGeneration stats:")
    stats = gen.get_stats()
    for category, count in stats['by_category'].items():
        print(f"  {category}: {count}")
    print(f"\n  Estimated tokens: {stats['estimated_tokens']:,}")

    # Export in all formats
    output_dir = Path(__file__).resolve().parent.parent.parent / "training_output"

    print("\nExporting...")
    gen.export("ict_training_chatml.jsonl", fmt="chatml", output_dir=output_dir)
    gen.export("ict_training_alpaca.jsonl", fmt="alpaca", output_dir=output_dir)
    gen.export("ict_training_sharegpt.jsonl", fmt="sharegpt", output_dir=output_dir)

    # Also generate train/test split
    gen.export_train_test_split(test_ratio=0.1, fmt="chatml", output_dir=output_dir)

    return gen


def run_logic_engine(store: ICTGraphStore):
    """Phase 5: Test the logic engine."""
    print("\n" + "=" * 70)
    print("  PHASE 5: Logic Engine Test Scenarios")
    print("=" * 70)

    reasoner = TradeReasoner(store)

    scenarios = [
        {
            "name": "A+ Silver Bullet",
            "patterns": ["fvg", "displacement", "liquidity_sweep", "order_block"],
            "htf_bias": "bullish",
            "session": "ny_am",
        },
        {
            "name": "Weak FVG Only",
            "patterns": ["fvg"],
            "htf_bias": "neutral",
            "session": "ny_am",
        },
        {
            "name": "Unicorn Setup",
            "patterns": ["order_block", "fvg", "displacement", "liquidity_sweep"],
            "htf_bias": "bearish",
            "session": "london",
        },
        {
            "name": "Judas Swing London",
            "patterns": ["judas_swing", "displacement", "fvg"],
            "htf_bias": "bullish",
            "session": "london",
        },
        {
            "name": "No Displacement (Should Fail)",
            "patterns": ["fvg", "order_block"],
            "htf_bias": "bullish",
            "session": "ny_am",
        },
        {
            "name": "Full AMD Cycle",
            "patterns": ["accumulation", "manipulation", "displacement", "fvg", "liquidity_sweep"],
            "htf_bias": "bearish",
            "session": "ny_am",
        },
    ]

    for scenario in scenarios:
        name = scenario.pop("name")
        print(f"\n--- {name} ---")
        result = reasoner.quick_check(**scenario)
        print(result.summary())
        scenario["name"] = name  # Restore

    return reasoner


def run_finetune():
    """Phase 4: Fine-tune with unsloth (requires GPU + unsloth installed)."""
    print("\n" + "=" * 70)
    print("  PHASE 4: LoRA Fine-Tuning Setup")
    print("=" * 70)

    output_dir = Path(__file__).resolve().parent.parent.parent / "training_output"
    train_path = output_dir / "ict_train.jsonl"

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        print("Run the training generator first: python -m src.graph_rag.run_pipeline --training-only")
        return

    # Print the fine-tuning script
    print(f"""
Fine-tuning data ready at: {train_path}

To fine-tune with unsloth (recommended for local LoRA):

    pip install unsloth

    Then run:
    
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-3b-instruct-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
    )
    
    # Load your training data
    dataset = load_dataset("json", data_files="{train_path}", split="train")
    
    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,
            learning_rate=2e-4,
            fp16=True,
            output_dir="outputs",
        ),
    )
    trainer.train()
    
    # Save LoRA adapter
    model.save_pretrained("ict_vex_lora")
    tokenizer.save_pretrained("ict_vex_lora")

Alternative: Fine-tune via Ollama (simpler, CPU-compatible):

    # Create a Modelfile
    cat > Modelfile << 'EOF'
    FROM llama3.2
    SYSTEM "You are VEX, an expert ICT trading assistant..."
    EOF
    
    # Then use ollama create
    ollama create vex-ict -f Modelfile

For Axolotl (more options):
    
    pip install axolotl
    # See axolotl docs for config.yaml setup
""")


def main():
    parser = argparse.ArgumentParser(description="ICT GraphRAG Pipeline")
    parser.add_argument("--training-only", action="store_true",
                        help="Only generate training data")
    parser.add_argument("--logic-only", action="store_true",
                        help="Only run logic engine tests")
    parser.add_argument("--chat", action="store_true",
                        help="Start interactive GraphRAG chat")
    parser.add_argument("--finetune", action="store_true",
                        help="Show fine-tuning instructions")
    parser.add_argument("--export-neo4j", action="store_true",
                        help="Export graph for Neo4j import")
    args = parser.parse_args()

    # Phase 1 always runs
    store = run_graph_store()

    if args.training_only:
        run_training_generator(store)
        return

    if args.logic_only:
        run_logic_engine(store)
        return

    if args.chat:
        from src.graph_rag.graph_retriever import GraphRAGRetriever
        retriever = GraphRAGRetriever(store)
        # Import and run the chat loop
        from src.graph_rag.graph_retriever import main as chat_main
        chat_main()
        return

    if args.finetune:
        run_finetune()
        return

    if args.export_neo4j:
        output_dir = Path(__file__).resolve().parent.parent.parent / "neo4j_export"
        store.export_for_neo4j(output_dir)
        return

    # Full pipeline
    run_training_generator(store)
    run_logic_engine(store)

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
Next steps:
  1. Review training data in training_output/
  2. Fine-tune: python -m src.graph_rag.run_pipeline --finetune
  3. Interactive chat: python -m src.graph_rag.run_pipeline --chat
  4. Export for Neo4j: python -m src.graph_rag.run_pipeline --export-neo4j
""")


if __name__ == "__main__":
    main()
