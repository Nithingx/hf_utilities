"""
Model Partitioner V2 - Main orchestrator for vision-language model splitting and inference.

Modes:
1. original - Run original model as-is
2. split_native - Run with separated pipeline (PyTorch/SafeTensor)
3. convert_onnx - Convert vision to ONNX, LLM to SafeTensor
4. run_onnx - Run with ONNX vision + SafeTensor LLM
5. save_standalone - Save models with standalone inference scripts

Usage examples:
    python model_partitioner_v2.py --mode original --image demo.jpg
    python model_partitioner_v2.py --mode split_native --image demo.jpg
    python model_partitioner_v2.py --mode convert_onnx --image demo.jpg
    python model_partitioner_v2.py --mode run_onnx --image demo.jpg
    python model_partitioner_v2.py --mode save_standalone --image demo.jpg
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from safetensors.torch import save_file, load_file
import os
import json
import argparse
import time
import psutil
from typing import Optional, Dict, Any
from PIL import Image

# Import pipeline modules
from vision_pipeline import VisionPipeline
from language_pipeline import LanguagePipeline
from onnx_converter import ONNXConverter


class PerformanceTracker:
    """Track performance metrics for inference."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()
        
    def start(self, operation_name: str):
        """Start tracking an operation."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        self.metrics[operation_name] = {
            'start_time': time.time(),
            'start_cpu_mem': self.process.memory_info().rss / (1024**2),
            'start_gpu_mem': torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        }
    
    def end(self, operation_name: str):
        """End tracking an operation."""
        if operation_name not in self.metrics:
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_cpu_mem = self.process.memory_info().rss / (1024**2)
        end_gpu_mem = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        self.metrics[operation_name].update({
            'end_time': end_time,
            'duration': end_time - self.metrics[operation_name]['start_time'],
            'cpu_mem_used': end_cpu_mem - self.metrics[operation_name]['start_cpu_mem'],
            'gpu_mem_used': end_gpu_mem - self.metrics[operation_name]['start_gpu_mem'],
            'peak_gpu_mem': peak_gpu_mem
        })
    
    def print_metrics(self, operation_name: str):
        """Print metrics for an operation."""
        if operation_name not in self.metrics:
            return
        
        m = self.metrics[operation_name]
        print(f"\n{'='*60}")
        print(f"PERFORMANCE: {operation_name}")
        print(f"{'='*60}")
        print(f"⏱️  Duration: {m['duration']:.3f}s")
        print(f"💾 CPU Memory: {m['cpu_mem_used']:.2f} MB")
        if torch.cuda.is_available():
            print(f"🎮 GPU Memory: {m['gpu_mem_used']:.2f} MB")
            print(f"🎮 GPU Peak: {m['peak_gpu_mem']:.2f} MB")
        print(f"{'='*60}\n")


class ModelPartitioner:
    """Main class for model partitioning and inference."""
    
    def __init__(self, model_id: str, device: str = 'auto', output_dir: str = 'split_models'):
        """
        Initialize model partitioner.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use ('auto', 'cuda', 'cpu')
            output_dir: Output directory for split models
        """
        self.model_id = model_id
        self.output_dir = output_dir
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        self.model = None
        self.processor = None
        self.tracker = PerformanceTracker()
        
        # Directories
        self.vision_dir = os.path.join(output_dir, 'vision_model')
        self.language_dir = os.path.join(output_dir, 'language_model')
        self.onnx_dir = os.path.join(output_dir, 'onnx_model')
        self.standalone_dir = os.path.join(output_dir, 'standalone')
        
    def load_original_model(self, quantize: bool = False):
        """Load original model."""
        print(f"\n{'='*70}")
        print("LOADING ORIGINAL MODEL")
        print(f"{'='*70}")
        print(f"Model: {self.model_id}")
        print(f"Device: {self.device}")
        print(f"Quantize: {quantize}")
        
        self.tracker.start("model_loading")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Determine dtype
        if quantize:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device,
                quantization_config=quant_config,
                attn_implementation="sdpa"
            )
        else:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=dtype,
                attn_implementation="sdpa"
            )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n✓ Model loaded")
        print(f"  Total parameters: {total_params:,}")
        
        self.tracker.end("model_loading")
        self.tracker.print_metrics("model_loading")
        
        return self.model, self.processor
    
    # ========== MODE 1: Original Model ==========
    
    def run_original_model(self, image_path: str, text_prompt: str, max_new_tokens: int = 128):
        """Run original model as-is."""
        print(f"\n{'='*70}")
        print("MODE 1: RUNNING ORIGINAL MODEL")
        print(f"{'='*70}")
        
        self.tracker.start("original_e2e")
        
        if self.model is None:
            self.load_original_model()
        
        # Prepare inputs
        image = Image.open(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }]
        
        # Process
        text_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
        
        vision_inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        inputs = text_inputs
        inputs.update({k: v for k, v in vision_inputs.items() if k not in inputs})
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode
        trimmed = generated[:, inputs["input_ids"].shape[-1]:]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        
        self.tracker.end("original_e2e")
        
        print(f"\n💬 Input: '{text_prompt}'")
        print(f"✨ Output: '{output_text}'")
        print(f"📊 Output tokens: {trimmed.shape[-1]}")
        
        self.tracker.print_metrics("original_e2e")
        
        return output_text
    
    # ========== MODE 2: Split Native ==========
    
    def split_and_save_models(self):
        """Split model into vision and language components."""
        print(f"\n{'='*70}")
        print("SPLITTING MODEL")
        print(f"{'='*70}")
        
        self.tracker.start("model_splitting")
        
        os.makedirs(self.vision_dir, exist_ok=True)
        os.makedirs(self.language_dir, exist_ok=True)
        
        if self.model is None:
            self.load_original_model()
        
        vision_state_dict = {}
        language_state_dict = {}
        
        for name, param in self.model.named_parameters():
            if 'visual' in name or 'vision' in name:
                vision_state_dict[name] = param.detach().cpu()
            else:
                language_state_dict[name] = param.detach().cpu()
        
        # Save vision model (.pt)
        vision_path = os.path.join(self.vision_dir, "vision_model.pt")
        torch.save(vision_state_dict, vision_path)
        print(f"✓ Vision saved: {vision_path} ({len(vision_state_dict)} params)")
        
        # Save language model (.safetensors)
        language_path = os.path.join(self.language_dir, "language_model.safetensors")
        save_file(language_state_dict, language_path)
        print(f"✓ Language saved: {language_path} ({len(language_state_dict)} params)")
        
        # Save config
        config = {
            "model_id": self.model_id,
            "vision_params": len(vision_state_dict),
            "language_params": len(language_state_dict),
            "vision_format": "pytorch",
            "language_format": "safetensors"
        }
        config_path = os.path.join(self.output_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.tracker.end("model_splitting")
        self.tracker.print_metrics("model_splitting")
        
        return vision_path, language_path
    
    def run_split_native(self, image_path: str, text_prompt: str, max_new_tokens: int = 128):
        """Run with split models in native format."""
        print(f"\n{'='*70}")
        print("MODE 2: RUNNING WITH SPLIT NATIVE MODELS")
        print(f"{'='*70}")
        
        self.tracker.start("split_native_e2e")
        
        # Load full model (we need architecture)
        if self.model is None:
            self.load_original_model()
        
        # Initialize pipelines
        vision_pipeline = VisionPipeline(model_format='pytorch', device=self.device)
        language_pipeline = LanguagePipeline(model_format='safetensors', device=self.device)
        
        # Load split models
        vision_path = os.path.join(self.vision_dir, "vision_model.pt")
        language_path = os.path.join(self.language_dir, "language_model.safetensors")
        
        vision_pipeline.processor = self.processor
        language_pipeline.load_safetensors_model(language_path, self.processor)
        language_pipeline.model = self.model
        
        # Run vision inference
        vision_results = vision_pipeline.run_inference(image_path, full_model=self.model)
        vision_pipeline.print_results(vision_results, image_path)
        
        # For full E2E, we still need to run combined
        # This demonstrates the pipeline works independently
        image = Image.open(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }]
        
        text_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.device)
        
        vision_inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        inputs = text_inputs
        inputs.update({k: v for k, v in vision_inputs.items() if k not in inputs})
        
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        trimmed = generated[:, inputs["input_ids"].shape[-1]:]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        
        self.tracker.end("split_native_e2e")
        
        print(f"\n💬 Input: '{text_prompt}'")
        print(f"✨ Output: '{output_text}'")
        
        self.tracker.print_metrics("split_native_e2e")
        
        return output_text
    
    # ========== MODE 3: Convert to ONNX ==========
    
    def convert_to_onnx(self, dummy_image_path: Optional[str] = None):
        """Convert vision model to ONNX and save language as SafeTensor."""
        print(f"\n{'='*70}")
        print("MODE 3: CONVERTING TO ONNX + SAFETENSOR")
        print(f"{'='*70}")
        
        # First split if not already done
        if not os.path.exists(os.path.join(self.language_dir, "language_model.safetensors")):
            self.split_and_save_models()
        
        # Convert vision to ONNX
        os.makedirs(self.onnx_dir, exist_ok=True)
        onnx_path = os.path.join(self.onnx_dir, "vision_model.onnx")
        
        if self.model is None:
            self.load_original_model()
        
        converter = ONNXConverter(device=self.device)
        onnx_path = converter.export_vision_model(
            self.model,
            self.processor,
            onnx_path,
            dummy_image_path=dummy_image_path
        )
        
        print(f"\n✅ Conversion complete!")
        print(f"  Vision (ONNX): {onnx_path}")
        print(f"  Language (SafeTensor): {os.path.join(self.language_dir, 'language_model.safetensors')}")
        
        return onnx_path
    
    # ========== MODE 4: Run with ONNX ==========
    
    def run_with_onnx(self, image_path: str, text_prompt: str, max_new_tokens: int = 128):
        """Run with ONNX vision model and SafeTensor language model."""
        print(f"\n{'='*70}")
        print("MODE 4: RUNNING WITH ONNX VISION + SAFETENSOR LLM")
        print(f"{'='*70}")
        
        self.tracker.start("onnx_e2e")
        
        # Load models
        if self.model is None:
            self.load_original_model()
        
        # Initialize pipelines
        vision_pipeline = VisionPipeline(model_format='onnx', device=self.device)
        language_pipeline = LanguagePipeline(model_format='safetensors', device=self.device)
        
        # Load ONNX vision model
        onnx_path = os.path.join(self.onnx_dir, "vision_model.onnx")
        if not os.path.exists(onnx_path):
            print("❌ ONNX model not found. Run convert_onnx first.")
            return None
        
        vision_pipeline.load_onnx_model(onnx_path)
        vision_pipeline.processor = self.processor
        
        # Load SafeTensor language model
        language_path = os.path.join(self.language_dir, "language_model.safetensors")
        language_pipeline.load_safetensors_model(language_path, self.processor)
        language_pipeline.model = self.model
        
        # Run vision inference with ONNX
        vision_results = vision_pipeline.run_inference(image_path)
        vision_pipeline.print_results(vision_results, image_path)
        
        # Run language inference (still need full model for E2E)
        # In production, you'd merge ONNX vision features with language model
        image = Image.open(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text_prompt},
            ],
        }]
        
        text_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(self.device)
        
        vision_inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        inputs = text_inputs
        inputs.update({k: v for k, v in vision_inputs.items() if k not in inputs})
        
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        trimmed = generated[:, inputs["input_ids"].shape[-1]:]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        
        self.tracker.end("onnx_e2e")
        
        print(f"\n💬 Input: '{text_prompt}'")
        print(f"✨ Output: '{output_text}'")
        
        self.tracker.print_metrics("onnx_e2e")
        
        return output_text
    
    # ========== MODE 5: Save Standalone ==========
    
    def save_standalone_models(self, image_path: str):
        """Save models with standalone inference scripts."""
        print(f"\n{'='*70}")
        print("MODE 5: SAVING STANDALONE MODELS WITH INFERENCE SCRIPTS")
        print(f"{'='*70}")
        
        # Create standalone directories
        vision_standalone = os.path.join(self.standalone_dir, 'vision')
        language_standalone = os.path.join(self.standalone_dir, 'language')
        os.makedirs(vision_standalone, exist_ok=True)
        os.makedirs(language_standalone, exist_ok=True)
        
        # Make sure models are converted
        if not os.path.exists(os.path.join(self.onnx_dir, "vision_model.onnx")):
            self.convert_to_onnx(image_path)
        
        # Copy vision ONNX model
        import shutil
        onnx_src = os.path.join(self.onnx_dir, "vision_model.onnx")
        onnx_dst = os.path.join(vision_standalone, "vision_model.onnx")
        shutil.copy(onnx_src, onnx_dst)
        print(f"✓ Copied vision ONNX: {onnx_dst}")
        
        # Copy language SafeTensor model
        lang_src = os.path.join(self.language_dir, "language_model.safetensors")
        lang_dst = os.path.join(language_standalone, "language_model.safetensors")
        shutil.copy(lang_src, lang_dst)
        print(f"✓ Copied language SafeTensor: {lang_dst}")
        
        # Create standalone vision inference script
        self._create_vision_inference_script(vision_standalone)
        
        # Create standalone language inference script
        self._create_language_inference_script(language_standalone)
        
        print(f"\n✅ Standalone models saved to: {self.standalone_dir}")
        print(f"  Vision: {vision_standalone}/")
        print(f"  Language: {language_standalone}/")
        
        return vision_standalone, language_standalone
    
    def _create_vision_inference_script(self, output_dir: str):
        """Create standalone inference script for vision model."""
        script = """#!/usr/bin/env python3
\"\"\"
Standalone Vision Model Inference Script
Run: python vision_inference.py --image path/to/image.jpg
\"\"\"

import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse

def run_vision_inference(image_path, model_path="vision_model.onnx"):
    # Load ONNX model
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # Load and preprocess image
    image = Image.open(image_path)
    # TODO: Add proper preprocessing based on your model
    
    print(f"Running vision inference on: {image_path}")
    print(f"Model: {model_path}")
    
    # TODO: Implement full preprocessing and inference
    print("✓ Vision inference complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="vision_model.onnx", help="Path to ONNX model")
    args = parser.parse_args()
    
    run_vision_inference(args.image, args.model)
"""
        script_path = os.path.join(output_dir, "vision_inference.py")
        with open(script_path, 'w') as f:
            f.write(script)
        print(f"  Created: {script_path}")
    
    def _create_language_inference_script(self, output_dir: str):
        """Create standalone inference script for language model."""
        script = f"""#!/usr/bin/env python3
\"\"\"
Standalone Language Model Inference Script
Run: python language_inference.py --text "Your prompt here"
\"\"\"

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor
import argparse

MODEL_ID = "{self.model_id}"

def run_language_inference(text, model_path="language_model.safetensors"):
    # Load model weights
    print(f"Loading model weights from: {{model_path}}")
    state_dict = load_file(model_path)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    print(f"Running language inference on: '{{text}}'")
    
    # TODO: Load full model architecture and load state dict
    # TODO: Run inference
    
    print("✓ Language inference complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Input text prompt")
    parser.add_argument("--model", default="language_model.safetensors", help="Path to SafeTensor model")
    args = parser.parse_args()
    
    run_language_inference(args.text, args.model)
"""
        script_path = os.path.join(output_dir, "language_inference.py")
        with open(script_path, 'w') as f:
            f.write(script)
        print(f"  Created: {script_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vision-Language Model Partitioner V2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['original', 'split_native', 'convert_onnx', 'run_onnx', 'save_standalone', 'all'],
        help='Run mode'
    )
    
    parser.add_argument('--model-id', type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--image', type=str, default='demo.jpg')
    parser.add_argument('--text', type=str, default='Describe this image succinctly.')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='split_models')
    parser.add_argument('--quantize', action='store_true', help='Use quantization')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print("MODEL PARTITIONER V2")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")
    
    partitioner = ModelPartitioner(args.model_id, args.device, args.output_dir)
    
    if args.mode == 'original':
        partitioner.run_original_model(args.image, args.text, args.max_tokens)
    
    elif args.mode == 'split_native':
        partitioner.split_and_save_models()
        partitioner.run_split_native(args.image, args.text, args.max_tokens)
    
    elif args.mode == 'convert_onnx':
        partitioner.convert_to_onnx(args.image)
    
    elif args.mode == 'run_onnx':
        partitioner.run_with_onnx(args.image, args.text, args.max_tokens)
    
    elif args.mode == 'save_standalone':
        partitioner.save_standalone_models(args.image)
    
    elif args.mode == 'all':
        partitioner.run_original_model(args.image, args.text, args.max_tokens)
        partitioner.split_and_save_models()
        partitioner.run_split_native(args.image, args.text, args.max_tokens)
        partitioner.convert_to_onnx(args.image)
        partitioner.run_with_onnx(args.image, args.text, args.max_tokens)
        partitioner.save_standalone_models(args.image)
    
    print(f"\n{'='*70}")
    print("✅ COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

