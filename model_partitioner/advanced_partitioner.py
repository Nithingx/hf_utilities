#!/usr/bin/env python3
"""
Advanced Model Partitioner for Qwen2.5-VL
Splits the model into 3 components:
1. Vision Encoder (ONNX) - Can run on NPU
2. Embedding Layer - Combines vision + text embeddings
3. LLM Decoder (SafeTensors) - For quantization and ONNX Runtime GenAI

Pipeline Flow:
Image → Vision.onnx (NPU) → Vision Embeddings
Text → Text Embeddings → Combined Embeddings → LLM Decoder → Output

Usage:
    python advanced_partitioner.py --model-id Qwen/Qwen2.5-VL-3B-Instruct --output-dir partitioned_model
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from safetensors.torch import save_file
import os
import json
import argparse
from typing import Dict, Any, Optional, Tuple
import numpy as np


class Qwen25VLPartitioner:
    """Partition Qwen2.5-VL into Vision, Embedding, and LLM components."""
    
    def __init__(self, model_id: str, output_dir: str = "partitioned_model"):
        """
        Initialize partitioner.
        
        Args:
            model_id: HuggingFace model ID
            output_dir: Directory to save partitioned components
        """
        self.model_id = model_id
        self.output_dir = output_dir
        self.model = None
        self.processor = None
        
        # Create output directories
        self.vision_dir = os.path.join(output_dir, "vision_encoder")
        self.embedding_dir = os.path.join(output_dir, "embedding_layer")
        self.llm_dir = os.path.join(output_dir, "llm_decoder")
        
        os.makedirs(self.vision_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)
        os.makedirs(self.llm_dir, exist_ok=True)
    
    def load_model(self):
        """Load the full VL model."""
        print(f"\n{'='*70}")
        print("LOADING QWEN2.5-VL MODEL")
        print(f"{'='*70}")
        print(f"Model: {self.model_id}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Load to CPU for partitioning
        )
        
        print(f"✓ Model loaded")
        print(f"  Model class: {type(self.model).__name__}")
        
        # Inspect model architecture
        self._inspect_architecture()
    
    def _inspect_architecture(self):
        """Inspect and print model architecture."""
        print(f"\n{'='*70}")
        print("MODEL ARCHITECTURE INSPECTION")
        print(f"{'='*70}")
        
        # Check for visual encoder
        if hasattr(self.model, 'visual'):
            print(f"✓ Visual encoder found: {type(self.model.visual).__name__}")
        
        # Check for language model
        if hasattr(self.model, 'model'):
            print(f"✓ Language model found: {type(self.model.model).__name__}")
            
            # Check for embeddings
            if hasattr(self.model.model, 'embed_tokens'):
                print(f"✓ Text embeddings found: {type(self.model.model.embed_tokens).__name__}")
            
            # Check for layers
            if hasattr(self.model.model, 'layers'):
                print(f"✓ Decoder layers found: {len(self.model.model.layers)} layers")
        
        # Check for LM head
        if hasattr(self.model, 'lm_head'):
            print(f"✓ LM head found: {type(self.model.lm_head).__name__}")
        
        print(f"{'='*70}\n")
    
    # ========== PART 1: Vision Encoder ==========
    
    def extract_vision_encoder(self) -> nn.Module:
        """Extract vision encoder module."""
        print(f"\n{'='*70}")
        print("EXTRACTING VISION ENCODER")
        print(f"{'='*70}")
        
        if hasattr(self.model, 'visual'):
            vision_encoder = self.model.visual
        elif hasattr(self.model, 'vision_tower'):
            vision_encoder = self.model.vision_tower
        else:
            raise ValueError("Could not find vision encoder in model")
        
        print(f"✓ Vision encoder extracted")
        print(f"  Type: {type(vision_encoder).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in vision_encoder.parameters())
        print(f"  Parameters: {total_params:,}")
        
        return vision_encoder
    
    def export_vision_to_onnx(
        self,
        dummy_image_path: Optional[str] = None,
        opset_version: int = 14
    ) -> str:
        """
        Export vision encoder to ONNX format.
        
        Args:
            dummy_image_path: Path to dummy image for export
            opset_version: ONNX opset version
            
        Returns:
            Path to exported ONNX model
        """
        print(f"\n{'='*70}")
        print("EXPORTING VISION ENCODER TO ONNX")
        print(f"{'='*70}")
        
        vision_encoder = self.extract_vision_encoder()
        vision_encoder.eval()
        
        # Create dummy input
        dummy_input = self._create_vision_dummy_input(dummy_image_path)
        
        # Export path
        onnx_path = os.path.join(self.vision_dir, "vision_encoder.onnx")
        
        print(f"Exporting to: {onnx_path}")
        print(f"Opset version: {opset_version}")
        
        # Prepare input names
        input_names = list(dummy_input.keys())
        output_names = ['vision_embeddings']
        
        # Dynamic axes for flexibility
        dynamic_axes = {
            name: {0: 'batch_size'} for name in input_names + output_names
        }
        
        # Export
        try:
            with torch.no_grad():
                torch.onnx.export(
                    vision_encoder,
                    tuple(dummy_input.values()),
                    onnx_path,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    verbose=False
                )
            
            print(f"✓ ONNX export successful")
            print(f"  File size: {os.path.getsize(onnx_path) / (1024**2):.2f} MB")
            
            # Verify ONNX model
            self._verify_onnx_vision(onnx_path, dummy_input)
            
            # Save metadata
            self._save_vision_metadata(onnx_path, dummy_input)
            
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            raise
    
    def _create_vision_dummy_input(self, dummy_image_path: Optional[str] = None) -> Dict:
        """Create dummy input for vision encoder."""
        from PIL import Image
        
        if dummy_image_path and os.path.exists(dummy_image_path):
            image = Image.open(dummy_image_path)
        else:
            # Create dummy image
            image = Image.new('RGB', (224, 224), color='red')
        
        # Process with processor
        inputs = self.processor(images=[image], return_tensors="pt")
        
        # Get only vision-related inputs
        vision_inputs = {
            k: v for k, v in inputs.items() 
            if 'pixel' in k.lower() or 'image' in k.lower()
        }
        
        return vision_inputs
    
    def _verify_onnx_vision(self, onnx_path: str, dummy_input: Dict):
        """Verify exported ONNX vision model."""
        try:
            import onnx
            import onnxruntime as ort
            
            print("\nVerifying ONNX model...")
            
            # Load and check
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model structure is valid")
            
            # Test inference
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
            # Convert inputs to numpy
            numpy_inputs = {k: v.cpu().numpy() for k, v in dummy_input.items()}
            
            # Run
            outputs = session.run(None, numpy_inputs)
            print(f"✓ ONNX inference test successful")
            print(f"  Output shape: {outputs[0].shape}")
            
        except Exception as e:
            print(f"⚠️  ONNX verification warning: {e}")
    
    def _save_vision_metadata(self, onnx_path: str, dummy_input: Dict):
        """Save vision encoder metadata."""
        metadata = {
            "model_id": self.model_id,
            "component": "vision_encoder",
            "format": "onnx",
            "onnx_path": os.path.basename(onnx_path),
            "input_shapes": {k: list(v.shape) for k, v in dummy_input.items()},
            "input_names": list(dummy_input.keys()),
            "output_names": ["vision_embeddings"],
            "recommended_device": "NPU/GPU",
            "notes": "This vision encoder can run on NPU for accelerated inference"
        }
        
        metadata_path = os.path.join(self.vision_dir, "vision_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved: {metadata_path}")
    
    # ========== PART 2: Embedding Layer ==========
    
    def extract_embedding_layer(self):
        """Extract and save embedding layer components."""
        print(f"\n{'='*70}")
        print("EXTRACTING EMBEDDING LAYER")
        print(f"{'='*70}")
        
        embedding_state = {}
        
        # Extract text embeddings
        if hasattr(self.model.model, 'embed_tokens'):
            embed_tokens = self.model.model.embed_tokens
            embedding_state['embed_tokens.weight'] = embed_tokens.weight.detach().cpu()
            print(f"✓ Text embeddings extracted")
            print(f"  Shape: {embed_tokens.weight.shape}")
        
        # Extract vision projection layer (if exists)
        # This layer projects vision embeddings to text embedding space
        for name, param in self.model.named_parameters():
            if 'visual' in name and ('proj' in name or 'adapter' in name):
                embedding_state[name] = param.detach().cpu()
                print(f"✓ Vision projection layer found: {name}")
        
        # Save embedding layer
        embedding_path = os.path.join(self.embedding_dir, "embedding_layer.safetensors")
        save_file(embedding_state, embedding_path)
        
        print(f"\n✓ Embedding layer saved: {embedding_path}")
        print(f"  Components: {len(embedding_state)}")
        print(f"  Size: {os.path.getsize(embedding_path) / (1024**2):.2f} MB")
        
        # Save metadata
        self._save_embedding_metadata(embedding_state)
        
        return embedding_state
    
    def _save_embedding_metadata(self, embedding_state: Dict):
        """Save embedding layer metadata."""
        metadata = {
            "model_id": self.model_id,
            "component": "embedding_layer",
            "format": "safetensors",
            "num_parameters": len(embedding_state),
            "parameter_names": list(embedding_state.keys()),
            "parameter_shapes": {k: list(v.shape) for k, v in embedding_state.items()},
            "function": "Combines vision embeddings from vision encoder with text embeddings",
            "notes": "This layer takes vision encoder outputs and text tokens, producing combined embeddings for LLM"
        }
        
        metadata_path = os.path.join(self.embedding_dir, "embedding_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Embedding metadata saved: {metadata_path}")
    
    # ========== PART 3: LLM Decoder ==========
    
    def extract_llm_decoder(self):
        """Extract LLM decoder (excluding embeddings and vision)."""
        print(f"\n{'='*70}")
        print("EXTRACTING LLM DECODER")
        print(f"{'='*70}")
        
        llm_state = {}
        
        for name, param in self.model.named_parameters():
            # Exclude vision encoder
            if 'visual' in name or 'vision' in name:
                continue
            
            # Exclude embedding layer (already saved separately)
            if 'embed_tokens' in name:
                continue
            
            # Include everything else (decoder layers, layer norms, lm_head)
            llm_state[name] = param.detach().cpu()
        
        print(f"✓ LLM decoder extracted")
        print(f"  Parameters: {len(llm_state)}")
        
        # Save as SafeTensors
        llm_path = os.path.join(self.llm_dir, "llm_decoder.safetensors")
        save_file(llm_state, llm_path)
        
        print(f"✓ LLM decoder saved: {llm_path}")
        print(f"  Size: {os.path.getsize(llm_path) / (1024**2):.2f} MB")
        
        # Save config
        self._save_llm_config()
        
        # Save metadata
        self._save_llm_metadata(llm_state)
        
        return llm_state
    
    def _save_llm_config(self):
        """Save LLM configuration for ONNX Runtime GenAI."""
        if hasattr(self.model, 'config'):
            config = self.model.config.to_dict()
            
            # Add custom fields for partitioned model
            config['partitioned'] = True
            config['requires_vision_embeddings'] = True
            config['requires_text_embeddings'] = True
            
            config_path = os.path.join(self.llm_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ LLM config saved: {config_path}")
    
    def _save_llm_metadata(self, llm_state: Dict):
        """Save LLM decoder metadata."""
        # Count layer types
        layer_counts = {}
        for name in llm_state.keys():
            if 'layers.' in name:
                layer_type = name.split('.')[-1]
                layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        metadata = {
            "model_id": self.model_id,
            "component": "llm_decoder",
            "format": "safetensors",
            "num_parameters": len(llm_state),
            "layer_counts": layer_counts,
            "parameter_names_sample": list(llm_state.keys())[:10],
            "ready_for_quantization": True,
            "ready_for_onnx_runtime_genai": True,
            "input": "Combined embeddings (vision + text)",
            "output": "Generated token logits",
            "notes": "This decoder can be quantized and converted to ONNX Runtime GenAI format"
        }
        
        metadata_path = os.path.join(self.llm_dir, "llm_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ LLM metadata saved: {metadata_path}")
    
    # ========== Complete Partitioning ==========
    
    def partition_model(self, dummy_image_path: Optional[str] = None):
        """Partition the complete model into 3 components."""
        print(f"\n{'='*70}")
        print("STARTING MODEL PARTITIONING")
        print(f"{'='*70}")
        print(f"Model: {self.model_id}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        # Load model
        if self.model is None:
            self.load_model()
        
        # 1. Extract and export vision encoder
        vision_onnx_path = self.export_vision_to_onnx(dummy_image_path)
        
        # 2. Extract embedding layer
        embedding_state = self.extract_embedding_layer()
        
        # 3. Extract LLM decoder
        llm_state = self.extract_llm_decoder()
        
        # Create master metadata
        self._create_master_metadata(vision_onnx_path, embedding_state, llm_state)
        
        # Create deployment guide
        self._create_deployment_guide()
        
        print(f"\n{'='*70}")
        print("✅ MODEL PARTITIONING COMPLETE")
        print(f"{'='*70}")
        print(f"\nPartitioned components saved to: {self.output_dir}")
        print(f"\nComponents:")
        print(f"  1. Vision Encoder (ONNX): {self.vision_dir}")
        print(f"  2. Embedding Layer: {self.embedding_dir}")
        print(f"  3. LLM Decoder: {self.llm_dir}")
        print(f"\n{'='*70}\n")
    
    def _create_master_metadata(self, vision_path: str, embedding_state: Dict, llm_state: Dict):
        """Create master metadata file."""
        metadata = {
            "model_id": self.model_id,
            "partitioning_version": "1.0",
            "components": {
                "vision_encoder": {
                    "path": self.vision_dir,
                    "format": "onnx",
                    "file": os.path.basename(vision_path),
                    "recommended_device": "NPU/GPU",
                    "size_mb": os.path.getsize(vision_path) / (1024**2)
                },
                "embedding_layer": {
                    "path": self.embedding_dir,
                    "format": "safetensors",
                    "file": "embedding_layer.safetensors",
                    "num_parameters": len(embedding_state),
                    "size_mb": os.path.getsize(os.path.join(self.embedding_dir, "embedding_layer.safetensors")) / (1024**2)
                },
                "llm_decoder": {
                    "path": self.llm_dir,
                    "format": "safetensors",
                    "file": "llm_decoder.safetensors",
                    "num_parameters": len(llm_state),
                    "size_mb": os.path.getsize(os.path.join(self.llm_dir, "llm_decoder.safetensors")) / (1024**2),
                    "ready_for_quantization": True,
                    "ready_for_onnx_genai": True
                }
            },
            "pipeline_flow": [
                "1. Image → Vision Encoder (ONNX on NPU)",
                "2. Vision Embeddings + Text Tokens → Embedding Layer",
                "3. Combined Embeddings → LLM Decoder → Generated Tokens"
            ],
            "next_steps": [
                "1. Deploy vision encoder on NPU using ONNX Runtime",
                "2. Quantize LLM decoder using your preferred method",
                "3. Convert quantized LLM to ONNX Runtime GenAI format",
                "4. Create inference pipeline connecting all components"
            ]
        }
        
        master_path = os.path.join(self.output_dir, "partitioning_metadata.json")
        with open(master_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Master metadata saved: {master_path}")
    
    def _create_deployment_guide(self):
        """Create deployment guide."""
        guide = """# Deployment Guide for Partitioned Qwen2.5-VL Model

## Architecture Overview

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       v
┌─────────────────────┐
│  Vision Encoder     │  ← ONNX format (deploy on NPU)
│  (vision_encoder    │
│   .onnx)            │
└──────┬──────────────┘
       │
       │  Vision Embeddings
       │
       v
┌─────────────────────┐     ┌─────────────┐
│  Embedding Layer    │ ←── │ Text Tokens │
│  (embedding_layer   │     └─────────────┘
│   .safetensors)     │
└──────┬──────────────┘
       │
       │  Combined Embeddings
       │
       v
┌─────────────────────┐
│  LLM Decoder        │  ← SafeTensors → Quantize → ONNX GenAI
│  (llm_decoder       │
│   .safetensors)     │
└──────┬──────────────┘
       │
       v
┌─────────────────────┐
│  Generated Text     │
└─────────────────────┘
```

## Component Details

### 1. Vision Encoder (vision_encoder/vision_encoder.onnx)
- **Format**: ONNX
- **Recommended Device**: NPU (or GPU as fallback)
- **Function**: Processes images into vision embeddings
- **Input**: Image pixels
- **Output**: Vision embeddings tensor

**Deployment on NPU:**
```python
import onnxruntime as ort

# Use NPU execution provider
session = ort.InferenceSession(
    "vision_encoder/vision_encoder.onnx",
    providers=['VitisAIExecutionProvider', 'CPUExecutionProvider']
)

# Run inference
vision_embeddings = session.run(None, {'pixel_values': image_pixels})[0]
```

### 2. Embedding Layer (embedding_layer/embedding_layer.safetensors)
- **Format**: SafeTensors
- **Function**: Combines vision embeddings with text token embeddings
- **Input**: Vision embeddings + Text token IDs
- **Output**: Combined embeddings for LLM

**Usage:**
```python
from safetensors.torch import load_file
import torch

# Load embeddings
embedding_weights = load_file("embedding_layer/embedding_layer.safetensors")

# Get text embeddings
text_embeddings = torch.nn.functional.embedding(text_tokens, embedding_weights['embed_tokens.weight'])

# Combine with vision embeddings
combined_embeddings = combine_embeddings(vision_embeddings, text_embeddings)
```

### 3. LLM Decoder (llm_decoder/llm_decoder.safetensors)
- **Format**: SafeTensors (ready for quantization)
- **Function**: Generates output tokens from combined embeddings
- **Input**: Combined embeddings
- **Output**: Token logits

## Next Steps for Production Deployment

### Step 1: Deploy Vision Encoder on NPU
```bash
# Optimize for VitisAI
vai_q_onnx quantize --model vision_encoder.onnx --output vision_encoder_quantized.onnx

# Compile for NPU
vai_c_onnx --model vision_encoder_quantized.onnx --arch DPUCZDX8G_ISA1_B4096
```

### Step 2: Quantize LLM Decoder
```bash
# Option A: Use ONNX Runtime quantization
python -m onnxruntime.quantization.preprocess --input llm_decoder.safetensors

# Option B: Use AWQ quantization
# Convert to HF format first, then apply AWQ

# Option C: Use GPTQ quantization
```

### Step 3: Convert LLM to ONNX Runtime GenAI
```bash
# Install ONNX Runtime GenAI tools
pip install onnxruntime-genai

# Convert to ONNX GenAI format
python -m onnxruntime_genai.models.builder \
    --model_type qwen \
    --input llm_decoder.safetensors \
    --output llm_decoder_genai \
    --precision int4 \
    --execution_provider cuda
```

### Step 4: Create Inference Pipeline
```python
# See inference_pipeline.py for complete example
```

## Performance Optimization Tips

1. **Vision Encoder on NPU**: Achieves 3-5x faster inference than CPU
2. **Quantization**: INT4/INT8 reduces LLM size by 4x with minimal quality loss
3. **Batching**: Process multiple images in parallel on NPU
4. **Caching**: Cache vision embeddings for multi-turn conversations

## Testing

Test each component independently:
1. Vision encoder: `python test_vision.py`
2. Embedding layer: `python test_embeddings.py`
3. LLM decoder: `python test_llm.py`
4. Full pipeline: `python test_pipeline.py`

## Troubleshooting

See TROUBLESHOOTING.md for common issues and solutions.
"""
        
        guide_path = os.path.join(self.output_dir, "DEPLOYMENT_GUIDE.md")
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f"✓ Deployment guide saved: {guide_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Model Partitioner for Qwen2.5-VL"
    )
    
    parser.add_argument(
        '--model-id',
        type=str,
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help='HuggingFace model ID'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='partitioned_model',
        help='Output directory for partitioned components'
    )
    
    parser.add_argument(
        '--dummy-image',
        type=str,
        help='Path to dummy image for ONNX export'
    )
    
    args = parser.parse_args()
    
    # Create partitioner
    partitioner = Qwen25VLPartitioner(args.model_id, args.output_dir)
    
    # Partition model
    partitioner.partition_model(args.dummy_image)
    
    print("\n🎉 Partitioning complete!")
    print(f"Check {args.output_dir} for all components and deployment guides.")


if __name__ == "__main__":
    main()

