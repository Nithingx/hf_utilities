#!/usr/bin/env python3
"""
Inference Pipeline for Partitioned Qwen2.5-VL Model

Demonstrates how to run inference using the 3 partitioned components:
1. Vision Encoder (ONNX on NPU/GPU)
2. Embedding Layer (SafeTensors)
3. LLM Decoder (SafeTensors/Quantized/ONNX GenAI)

Usage:
    python inference_pipeline.py \
        --partitioned-dir partitioned_model \
        --image demo.jpg \
        --text "Describe this image"
"""

import torch
import torch.nn as nn
import onnxruntime as ort
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import os
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional
import numpy as np


class PartitionedInferencePipeline:
    """Inference pipeline for partitioned Qwen2.5-VL model."""
    
    def __init__(
        self,
        partitioned_dir: str,
        device: str = 'cuda',
        vision_device: str = 'npu'  # Can be 'npu', 'cuda', 'cpu'
    ):
        """
        Initialize inference pipeline.
        
        Args:
            partitioned_dir: Directory containing partitioned components
            device: Device for embedding and LLM ('cuda' or 'cpu')
            vision_device: Device for vision encoder ('npu', 'cuda', or 'cpu')
        """
        self.partitioned_dir = partitioned_dir
        self.device = device
        self.vision_device = vision_device
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.model_id = self.metadata['model_id']
        
        # Initialize components
        self.vision_session = None
        self.embedding_weights = None
        self.llm_model = None
        self.tokenizer = None
        self.processor = None
        
        # Performance tracking
        self.timings = {}
    
    def _load_metadata(self) -> Dict:
        """Load partitioning metadata."""
        metadata_path = os.path.join(self.partitioned_dir, "partitioning_metadata.json")
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def load_components(self):
        """Load all partitioned components."""
        print(f"\n{'='*70}")
        print("LOADING PARTITIONED COMPONENTS")
        print(f"{'='*70}\n")
        
        # 1. Load vision encoder (ONNX)
        self._load_vision_encoder()
        
        # 2. Load embedding layer
        self._load_embedding_layer()
        
        # 3. Load LLM decoder
        self._load_llm_decoder()
        
        # 4. Load tokenizer and processor
        self._load_tokenizer_processor()
        
        print(f"\n{'='*70}")
        print("✓ ALL COMPONENTS LOADED")
        print(f"{'='*70}\n")
    
    def _load_vision_encoder(self):
        """Load vision encoder ONNX model."""
        print("Loading Vision Encoder (ONNX)...")
        
        vision_dir = os.path.join(self.partitioned_dir, "vision_encoder")
        onnx_path = os.path.join(vision_dir, "vision_encoder.onnx")
        
        # Determine execution providers based on vision_device
        providers = self._get_vision_providers()
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.vision_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"✓ Vision encoder loaded")
        print(f"  Device: {self.vision_device}")
        print(f"  Providers: {self.vision_session.get_providers()}")
        print(f"  Inputs: {[i.name for i in self.vision_session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in self.vision_session.get_outputs()]}\n")
    
    def _get_vision_providers(self) -> List[str]:
        """Get ONNX Runtime execution providers for vision encoder."""
        providers = []
        
        if self.vision_device == 'npu':
            # Try VitisAI first, fallback to CUDA, then CPU
            available = ort.get_available_providers()
            if 'VitisAIExecutionProvider' in available:
                providers.append('VitisAIExecutionProvider')
                print("  Using VitisAI Execution Provider for NPU")
            elif 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
                print("  VitisAI not available, using CUDA")
            providers.append('CPUExecutionProvider')
        
        elif self.vision_device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        else:  # cpu
            providers = ['CPUExecutionProvider']
        
        return providers
    
    def _load_embedding_layer(self):
        """Load embedding layer."""
        print("Loading Embedding Layer...")
        
        embedding_dir = os.path.join(self.partitioned_dir, "embedding_layer")
        embedding_path = os.path.join(embedding_dir, "embedding_layer.safetensors")
        
        self.embedding_weights = load_file(embedding_path)
        
        # Move to device
        for key in self.embedding_weights:
            self.embedding_weights[key] = self.embedding_weights[key].to(self.device)
        
        print(f"✓ Embedding layer loaded")
        print(f"  Device: {self.device}")
        print(f"  Components: {len(self.embedding_weights)}\n")
    
    def _load_llm_decoder(self):
        """Load LLM decoder (placeholder - will be loaded based on format)."""
        print("Loading LLM Decoder...")
        
        llm_dir = os.path.join(self.partitioned_dir, "llm_decoder")
        llm_path = os.path.join(llm_dir, "llm_decoder.safetensors")
        
        # For now, we'll note that the LLM needs the full model architecture
        # In production, you'd either:
        # 1. Load the full model and replace weights
        # 2. Use ONNX Runtime GenAI
        # 3. Use a custom model implementation
        
        print(f"✓ LLM decoder ready")
        print(f"  Path: {llm_path}")
        print(f"  Note: For full inference, need to integrate with model architecture")
        print(f"  Options:")
        print(f"    1. Load full model and inject weights")
        print(f"    2. Convert to ONNX Runtime GenAI")
        print(f"    3. Use custom implementation\n")
        
        # Store path for later use
        self.llm_path = llm_path
    
    def _load_tokenizer_processor(self):
        """Load tokenizer and processor."""
        print("Loading Tokenizer and Processor...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        print(f"✓ Tokenizer and processor loaded\n")
    
    # ========== Inference Pipeline ==========
    
    def run_vision_inference(self, image_path: str) -> np.ndarray:
        """
        Run vision encoder inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Vision embeddings as numpy array
        """
        print(f"\n[1/4] Running Vision Encoder...")
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Process with processor
        inputs = self.processor(images=[image], return_tensors="pt")
        
        # Extract vision inputs
        vision_inputs = {
            k: v.cpu().numpy() 
            for k, v in inputs.items() 
            if 'pixel' in k.lower() or 'image' in k.lower()
        }
        
        # Run ONNX inference
        vision_outputs = self.vision_session.run(None, vision_inputs)
        vision_embeddings = vision_outputs[0]
        
        duration = time.time() - start_time
        self.timings['vision_encoder'] = duration
        
        print(f"✓ Vision encoder complete ({duration:.3f}s)")
        print(f"  Output shape: {vision_embeddings.shape}")
        print(f"  Device: {self.vision_device}\n")
        
        return vision_embeddings
    
    def create_text_embeddings(self, text: str) -> torch.Tensor:
        """
        Create text embeddings.
        
        Args:
            text: Input text
            
        Returns:
            Text embeddings tensor
        """
        print(f"[2/4] Creating Text Embeddings...")
        start_time = time.time()
        
        # Tokenize
        tokens = self.tokenizer(text, return_tensors="pt")
        input_ids = tokens['input_ids'].to(self.device)
        
        # Get embeddings
        embed_weight = self.embedding_weights['embed_tokens.weight']
        text_embeddings = torch.nn.functional.embedding(input_ids, embed_weight)
        
        duration = time.time() - start_time
        self.timings['text_embeddings'] = duration
        
        print(f"✓ Text embeddings complete ({duration:.3f}s)")
        print(f"  Text: '{text}'")
        print(f"  Tokens: {input_ids.shape[1]}")
        print(f"  Embeddings shape: {text_embeddings.shape}\n")
        
        return text_embeddings, input_ids
    
    def combine_embeddings(
        self,
        vision_embeddings: np.ndarray,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine vision and text embeddings.
        
        Args:
            vision_embeddings: Vision embeddings from vision encoder
            text_embeddings: Text embeddings from embedding layer
            
        Returns:
            Combined embeddings
        """
        print(f"[3/4] Combining Embeddings...")
        start_time = time.time()
        
        # Convert vision embeddings to torch
        vision_embeddings_torch = torch.from_numpy(vision_embeddings).to(self.device)
        
        # In Qwen2.5-VL, vision embeddings are interleaved with text embeddings
        # at specific positions marked by image tokens
        # For simplicity, we concatenate here (actual implementation would be more complex)
        
        combined_embeddings = torch.cat([vision_embeddings_torch, text_embeddings], dim=1)
        
        duration = time.time() - start_time
        self.timings['combine_embeddings'] = duration
        
        print(f"✓ Embeddings combined ({duration:.3f}s)")
        print(f"  Vision shape: {vision_embeddings.shape}")
        print(f"  Text shape: {text_embeddings.shape}")
        print(f"  Combined shape: {combined_embeddings.shape}\n")
        
        return combined_embeddings
    
    def run_llm_decoder(
        self,
        combined_embeddings: torch.Tensor,
        max_new_tokens: int = 128
    ) -> str:
        """
        Run LLM decoder (placeholder).
        
        Args:
            combined_embeddings: Combined vision+text embeddings
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        print(f"[4/4] Running LLM Decoder...")
        print(f"⚠️  Note: Full LLM inference requires complete model architecture")
        print(f"  For production, use one of:")
        print(f"    1. ONNX Runtime GenAI")
        print(f"    2. vLLM with custom engine")
        print(f"    3. Full model with injected weights\n")
        
        # Placeholder - in production you would:
        # Option 1: Use ONNX Runtime GenAI
        # Option 2: Load full model and inject LLM weights
        # Option 3: Use custom LLM implementation
        
        return "[Generated text would appear here - integrate with full LLM implementation]"
    
    def run_full_pipeline(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int = 128
    ) -> Tuple[str, Dict]:
        """
        Run complete inference pipeline.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_text, timings)
        """
        print(f"\n{'='*70}")
        print("RUNNING FULL INFERENCE PIPELINE")
        print(f"{'='*70}")
        print(f"Image: {image_path}")
        print(f"Prompt: {text_prompt}")
        print(f"{'='*70}\n")
        
        total_start = time.time()
        
        # Step 1: Vision encoder
        vision_embeddings = self.run_vision_inference(image_path)
        
        # Step 2: Text embeddings
        text_embeddings, input_ids = self.create_text_embeddings(text_prompt)
        
        # Step 3: Combine embeddings
        combined_embeddings = self.combine_embeddings(vision_embeddings, text_embeddings)
        
        # Step 4: LLM decoder
        output_text = self.run_llm_decoder(combined_embeddings, max_new_tokens)
        
        total_duration = time.time() - total_start
        self.timings['total'] = total_duration
        
        # Print summary
        self._print_timing_summary()
        
        return output_text, self.timings
    
    def _print_timing_summary(self):
        """Print timing summary."""
        print(f"\n{'='*70}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        for component, duration in self.timings.items():
            if component != 'total':
                percentage = (duration / self.timings['total']) * 100
                print(f"  {component:<25}: {duration:.3f}s ({percentage:.1f}%)")
        
        print(f"  {'-'*68}")
        print(f"  {'Total':<25}: {self.timings['total']:.3f}s")
        print(f"{'='*70}\n")


class FullModelInferencePipeline(PartitionedInferencePipeline):
    """
    Extended pipeline that loads full model for complete inference.
    This demonstrates how to inject partitioned weights into full model.
    """
    
    def _load_llm_decoder(self):
        """Load full model and inject LLM weights."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        print("Loading Full Model (for complete inference)...")
        
        # Load full model
        self.llm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        
        # Optionally: Load partitioned LLM weights
        # llm_weights = load_file(self.llm_path)
        # self.llm_model.load_state_dict(llm_weights, strict=False)
        
        print(f"✓ Full model loaded for LLM inference")
        print(f"  Device: {self.device}\n")
    
    def run_llm_decoder(
        self,
        combined_embeddings: torch.Tensor,
        max_new_tokens: int = 128
    ) -> str:
        """
        Run LLM decoder with full model.
        
        Args:
            combined_embeddings: Combined vision+text embeddings
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        print(f"[4/4] Running LLM Decoder (Full Model)...")
        start_time = time.time()
        
        with torch.no_grad():
            # Note: This is simplified - actual implementation would need
            # proper attention masks and position IDs
            outputs = self.llm_model.generate(
                inputs_embeds=combined_embeddings,
                max_new_tokens=max_new_tokens
            )
        
        # Decode
        generated_text = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        duration = time.time() - start_time
        self.timings['llm_decoder'] = duration
        
        print(f"✓ LLM decoder complete ({duration:.3f}s)")
        print(f"  Generated tokens: {outputs.shape[1]}")
        print(f"  Output: '{generated_text[:100]}...'\n")
        
        return generated_text


def main():
    parser = argparse.ArgumentParser(
        description="Inference Pipeline for Partitioned Qwen2.5-VL"
    )
    
    parser.add_argument(
        '--partitioned-dir',
        type=str,
        required=True,
        help='Directory containing partitioned model components'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        default='Describe this image.',
        help='Text prompt'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=128,
        help='Maximum tokens to generate'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for LLM'
    )
    
    parser.add_argument(
        '--vision-device',
        type=str,
        default='npu',
        choices=['npu', 'cuda', 'cpu'],
        help='Device for vision encoder'
    )
    
    parser.add_argument(
        '--full-model',
        action='store_true',
        help='Use full model for LLM decoder (enables complete inference)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    if args.full_model:
        pipeline = FullModelInferencePipeline(
            args.partitioned_dir,
            device=args.device,
            vision_device=args.vision_device
        )
    else:
        pipeline = PartitionedInferencePipeline(
            args.partitioned_dir,
            device=args.device,
            vision_device=args.vision_device
        )
    
    # Load components
    pipeline.load_components()
    
    # Run inference
    output_text, timings = pipeline.run_full_pipeline(
        args.image,
        args.text,
        args.max_tokens
    )
    
    print(f"\n{'='*70}")
    print("FINAL OUTPUT")
    print(f"{'='*70}")
    print(f"{output_text}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

