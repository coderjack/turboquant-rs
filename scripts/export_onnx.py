#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 to ONNX format for use with OnnxEmbedder.

Usage:
    pip install transformers[onnx] optimum torch
    python scripts/export_onnx.py [--output models/minilm]

Creates:
    <output>/model.onnx
    <output>/tokenizer.json
"""
import argparse
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(description="Export MiniLM to ONNX")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/minilm"),
        help="Output directory (default: models/minilm)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"HuggingFace model name (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {args.model} to ONNX...")

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    # Export to ONNX via optimum
    model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Verify the export
    onnx_files = list(args.output.glob("*.onnx"))
    assert onnx_files, f"No .onnx files found in {args.output}"
    assert (args.output / "tokenizer.json").exists(), "tokenizer.json not found"

    print(f"Done! Model exported to {args.output}/")
    print(f"  ONNX model: {onnx_files[0].name}")
    print(f"  Tokenizer:  tokenizer.json")
    print()
    print("To use with OnnxEmbedder:")
    print(f'  let embedder = OnnxEmbedder::load(Path::new("{args.output}"), 384)?;')


if __name__ == "__main__":
    main()
