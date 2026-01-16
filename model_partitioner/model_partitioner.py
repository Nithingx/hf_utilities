import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from safetensors.torch import save_file, load_file
import os
import json
import argparse
import time
import psutil
from typing import Optional, Dict, Any

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # 3B instruct checkpoint

# ========== Performance Tracking ==========

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
            'start_cpu_mem': self.process.memory_info().rss / (1024**2),  # MB
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
        print(f"PERFORMANCE METRICS: {operation_name}")
        print(f"{'='*60}")
        print(f"⏱️  Duration: {m['duration']:.3f} seconds")
        print(f"💾 CPU Memory: {m['cpu_mem_used']:.2f} MB")
        if torch.cuda.is_available():
            print(f"🎮 GPU Memory Used: {m['gpu_mem_used']:.2f} MB")
            print(f"🎮 GPU Peak Memory: {m['peak_gpu_mem']:.2f} MB")
        print(f"{'='*60}\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return self.metrics

# ========== Model Loading with Quantization Support ==========

def load_model_and_processor(model_id: str, device: str = 'auto', quantize: bool = False, quantization_config: Optional[Dict] = None):
    """
    Load model and processor with optional quantization.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on ('auto', 'cuda', 'cpu')
        quantize: Whether to quantize the model
        quantization_config: Configuration for quantization
    
    Returns:
        tuple: (model, processor)
    """
    print(f"\n{'='*70}")
    print(f"LOADING MODEL: {model_id}")
    print(f"{'='*70}")
    
    # Determine dtype
    if quantize:
        print("⚙️  Quantization enabled")
        dtype = torch.float16  # Use float16 for quantization compatibility
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"📊 Data type: {dtype}")
    print(f"🎮 Device: {device}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"✓ Processor loaded")
    
    # Load model with optional quantization
    if quantize:
        from transformers import BitsAndBytesConfig
        
        # Default quantization config (4-bit)
        if quantization_config is None:
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }
        
        quant_config = BitsAndBytesConfig(**quantization_config)
        print(f"🔧 Quantization config: {quantization_config}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            quantization_config=quant_config,
            attn_implementation="sdpa"
        )
        print(f"✓ Model loaded with quantization")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation="sdpa"
        )
        print(f"✓ Model loaded")
    
    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Total parameters: {total_params:,}")
    print(f"📦 Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    return model, processor

# ========== Split and Save Models ==========

def split_and_save_models(model, output_dir="split_models", tracker: Optional[PerformanceTracker] = None):
    """
    Split the vision-language model into vision and language components
    and save them separately.
    Vision model: saved as .pt (PyTorch format)
    Language model: saved as .safetensors format
    
    Args:
        model: Qwen2_5_VLForConditionalGeneration model
        output_dir: Directory to save the split models
        tracker: Performance tracker instance
    """
    if tracker:
        tracker.start("model_splitting")
    
    os.makedirs(output_dir, exist_ok=True)
    
    vision_dir = os.path.join(output_dir, "vision_model")
    language_dir = os.path.join(output_dir, "language_model")
    os.makedirs(vision_dir, exist_ok=True)
    os.makedirs(language_dir, exist_ok=True)
    
    vision_state_dict = {}
    language_state_dict = {}
    
    print("\n" + "="*70)
    print("SPLITTING MODEL COMPONENTS")
    print("="*70)
    
    for name, param in model.named_parameters():
        if 'visual' in name or 'vision' in name:
            vision_state_dict[name] = param.detach().cpu()
        else:
            language_state_dict[name] = param.detach().cpu()
    
    # Save vision model as .pt (PyTorch format)
    vision_path = os.path.join(vision_dir, "vision_model.pt")
    torch.save(vision_state_dict, vision_path)
    print(f"✓ Vision model saved to: {vision_path}")
    print(f"  - Format: PyTorch (.pt)")
    print(f"  - Parameters: {len(vision_state_dict)}")
    print(f"  - Size: {os.path.getsize(vision_path) / (1024**2):.2f} MB")
    
    # Save language model as .safetensors format
    language_path = os.path.join(language_dir, "language_model.safetensors")
    save_file(language_state_dict, language_path)
    print(f"✓ Language model saved to: {language_path}")
    print(f"  - Format: SafeTensors (.safetensors)")
    print(f"  - Parameters: {len(language_state_dict)}")
    print(f"  - Size: {os.path.getsize(language_path) / (1024**2):.2f} MB")
    
    # Save model configuration
    config_path = os.path.join(output_dir, "model_config.json")
    config_dict = {
        "model_id": MODEL_ID,
        "vision_params": len(vision_state_dict),
        "language_params": len(language_state_dict),
        "vision_format": "pytorch",
        "language_format": "safetensors",
        "vision_param_names": list(vision_state_dict.keys()),
        "language_param_names": list(language_state_dict.keys())
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Configuration saved to: {config_path}")
    
    if tracker:
        tracker.end("model_splitting")
        tracker.print_metrics("model_splitting")
    
    return vision_state_dict, language_state_dict

# ========== Load Split Models ==========

def load_split_models(output_dir="split_models"):
    """
    Load the previously saved vision and language models.
    Vision model: loaded from .pt format
    Language model: loaded from .safetensors format
    
    Args:
        output_dir: Directory where the split models are saved
    
    Returns:
        tuple: (vision_state_dict, language_state_dict)
    """
    vision_path = os.path.join(output_dir, "vision_model", "vision_model.pt")
    language_path = os.path.join(output_dir, "language_model", "language_model.safetensors")
    
    # Load vision model from PyTorch format
    vision_state_dict = torch.load(vision_path, map_location='cpu')
    print(f"✓ Vision model loaded from: {vision_path} (PyTorch format)")
    
    # Load language model from SafeTensors format
    language_state_dict = load_file(language_path)
    print(f"✓ Language model loaded from: {language_path} (SafeTensors format)")
    
    return vision_state_dict, language_state_dict

# ========== Inference with Vision Model Only ==========

def run_vision_inference(model, processor, image_path, device='cuda' if torch.cuda.is_available() else 'cpu', tracker: Optional[PerformanceTracker] = None):
    """
    Run inference using only the vision model to extract image features.
    
    Args:
        model: Full Qwen2_5_VLForConditionalGeneration model
        processor: AutoProcessor for the model
        image_path: Path to the image file
        device: Device to run inference on
        tracker: Performance tracker instance
    
    Returns:
        dict: Dictionary containing vision features and metadata
    """
    if tracker:
        tracker.start("vision_inference")
    
    print("\n" + "="*70)
    print("VISION MODEL INFERENCE")
    print("="*70)
    
    # Process image
    from PIL import Image
    if isinstance(image_path, str):
        image = Image.open(image_path)
        print(f"📷 Image loaded: {image_path}")
    else:
        image = image_path
        print(f"📷 Image object provided")
    
    print(f"📏 Image size: {image.size}")
    
    # Get vision inputs
    vision_inputs = processor(images=[image], return_tensors="pt").to(device)
    
    # Extract vision features using the visual component
    with torch.no_grad():
        if hasattr(model, 'visual'):
            vision_features = model.visual(**vision_inputs)
        elif hasattr(model, 'vision_tower'):
            vision_features = model.vision_tower(**vision_inputs)
        else:
            # For Qwen2.5-VL, access the vision encoder
            vision_features = model.model.visual(**vision_inputs) if hasattr(model.model, 'visual') else None
    
    # Get feature statistics
    result = None
    if vision_features is not None:
        if isinstance(vision_features, torch.Tensor):
            feat_tensor = vision_features
        elif hasattr(vision_features, 'last_hidden_state'):
            feat_tensor = vision_features.last_hidden_state
        else:
            feat_tensor = vision_features[0] if isinstance(vision_features, tuple) else vision_features
        
        print(f"\n✓ Vision features extracted successfully")
        print(f"  - Shape: {feat_tensor.shape}")
        print(f"  - dtype: {feat_tensor.dtype}")
        print(f"  - Device: {feat_tensor.device}")
        print(f"  - Mean: {feat_tensor.mean().item():.4f}")
        print(f"  - Std: {feat_tensor.std().item():.4f}")
        print(f"  - Min: {feat_tensor.min().item():.4f}")
        print(f"  - Max: {feat_tensor.max().item():.4f}")
        
        result = {
            'features': feat_tensor,
            'shape': feat_tensor.shape,
            'raw_output': vision_features
        }
    else:
        print("⚠ Could not extract vision features")
    
    if tracker:
        tracker.end("vision_inference")
        tracker.print_metrics("vision_inference")
    
    return result

# ========== Inference with Language Model Only ==========

def run_language_inference(model, processor, text_prompt, device='cuda' if torch.cuda.is_available() else 'cpu', max_new_tokens=128, tracker: Optional[PerformanceTracker] = None):
    """
    Run inference using only the language model for text-only tasks.
    
    Args:
        model: Full Qwen2_5_VLForConditionalGeneration model
        processor: AutoProcessor for the model
        text_prompt: Text prompt for generation
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        tracker: Performance tracker instance
    
    Returns:
        str: Generated text
    """
    if tracker:
        tracker.start("language_inference")
    
    print("\n" + "="*70)
    print("LANGUAGE MODEL INFERENCE")
    print("="*70)
    
    # Create text-only message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    # Process text input
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_tensors="pt"
    ).to(device)
    
    print(f"💬 Text prompt: '{text_prompt}'")
    print(f"  - Input shape: {inputs.shape}")
    print(f"  - Input tokens: {inputs.shape[-1]}")
    
    # Generate using language model
    with torch.no_grad():
        generated = model.generate(inputs, max_new_tokens=max_new_tokens)
    
    # Decode output
    trimmed = generated[:, inputs.shape[-1]:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    
    print(f"\n✓ Generated {trimmed.shape[-1]} tokens")
    print(f"✓ Output: '{text}'")
    
    # Calculate tokens per second
    if tracker and "language_inference" in tracker.metrics:
        duration = time.time() - tracker.metrics["language_inference"]['start_time']
        tokens_per_sec = trimmed.shape[-1] / duration if duration > 0 else 0
        print(f"⚡ Tokens/sec: {tokens_per_sec:.2f}")
    
    if tracker:
        tracker.end("language_inference")
        tracker.print_metrics("language_inference")
    
    return text

# ========== Inference with Both Models (Manual Pipeline) ==========

def run_combined_inference(model, processor, image_path, text_prompt, device='cuda' if torch.cuda.is_available() else 'cpu', max_new_tokens=128, tracker: Optional[PerformanceTracker] = None):
    """
    Run inference using both vision and language models in a manual pipeline.
    
    Args:
        model: Full Qwen2_5_VLForConditionalGeneration model
        processor: AutoProcessor for the model
        image_path: Path to the image file
        text_prompt: Text prompt for the image
        device: Device to run inference on
        max_new_tokens: Maximum number of tokens to generate
        tracker: Performance tracker instance
    
    Returns:
        dict: Dictionary containing all outputs and intermediate results
    """
    if tracker:
        tracker.start("combined_inference")
    
    print("\n" + "="*70)
    print("COMBINED VISION-LANGUAGE INFERENCE (END-TO-END)")
    print("="*70)
    
    # Step 1: Extract vision features
    print("\n[Step 1] Extracting vision features...")
    vision_result = run_vision_inference(model, processor, image_path, device)
    
    # Step 2: Prepare multimodal inputs
    print("\n[Step 2] Preparing multimodal inputs...")
    from PIL import Image
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path if isinstance(image_path, str) else image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    # Process inputs
    text_inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_tensors="pt"
    ).to(device)
    
    vision_inputs = processor(
        images=[image],
        return_tensors="pt"
    ).to(device)
    
    # Merge inputs
    inputs = text_inputs
    inputs.update({k: v for k, v in vision_inputs.items() if k not in inputs})
    
    print(f"💬 Text prompt: '{text_prompt}'")
    print(f"  - Text input tokens: {text_inputs.shape[-1]}")
    print(f"  - Vision input keys: {list(vision_inputs.keys())}")
    
    # Step 3: Generate using full model
    print("\n[Step 3] Generating with combined model...")
    gen_start = time.time()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    gen_time = time.time() - gen_start
    
    # Decode output
    trimmed = generated[:, inputs["input_ids"].shape[-1]:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    
    print(f"\n✓ Generated {trimmed.shape[-1]} tokens in {gen_time:.3f}s")
    print(f"⚡ Tokens/sec: {trimmed.shape[-1]/gen_time:.2f}")
    print(f"✓ Output: '{text}'")
    
    if tracker:
        tracker.end("combined_inference")
        tracker.print_metrics("combined_inference")
    
    return {
        'vision_features': vision_result,
        'generated_text': text,
        'generated_tokens': trimmed,
        'input_tokens': inputs["input_ids"],
        'generation_time': gen_time,
        'tokens_per_sec': trimmed.shape[-1]/gen_time
    }

# ========== Main Execution ==========

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Vision-Language Model Partitioner and Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split model and run all inference modes
  python model_partitioner.py --mode all --image demo.jpg
  
  # Run vision-only inference
  python model_partitioner.py --mode vision --image demo.jpg
  
  # Run language-only inference
  python model_partitioner.py --mode language --text "Explain AI in one sentence"
  
  # Run end-to-end inference on GPU with quantization
  python model_partitioner.py --mode e2e --image demo.jpg --device cuda --quantize
  
  # Split model without running inference
  python model_partitioner.py --mode split --no-inference
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['split', 'vision', 'language', 'e2e', 'all'],
        default='all',
        help='Inference mode: split (split only), vision (vision only), language (language only), e2e (end-to-end), all (run all modes)'
    )
    
    # Model configuration
    parser.add_argument(
        '--model-id',
        type=str,
        default=MODEL_ID,
        help='HuggingFace model ID'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Enable 4-bit quantization for the model'
    )
    
    parser.add_argument(
        '--quantization-bits',
        type=int,
        choices=[4, 8],
        default=4,
        help='Quantization bits (4 or 8)'
    )
    
    # Input/Output
    parser.add_argument(
        '--image',
        type=str,
        default='demo.jpg',
        help='Path to input image for vision and e2e modes'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        default='Explain the concept of artificial intelligence in one sentence.',
        help='Text prompt for language-only mode'
    )
    
    parser.add_argument(
        '--image-text',
        type=str,
        default='Describe this image succinctly.',
        help='Text prompt for e2e mode (image description)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='split_models',
        help='Directory to save split models'
    )
    
    # Generation parameters
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=128,
        help='Maximum number of tokens to generate'
    )
    
    # Control flags
    parser.add_argument(
        '--no-inference',
        action='store_true',
        help='Skip inference, only split the model'
    )
    
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Skip splitting, only run inference'
    )
    
    parser.add_argument(
        '--load-split',
        action='store_true',
        help='Load previously split models for verification'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        default=True,
        help='Enable performance tracking (default: True)'
    )
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize performance tracker
    tracker = PerformanceTracker() if args.performance else None
    
    print("\n" + "="*70)
    print("VISION-LANGUAGE MODEL PARTITIONER & INFERENCE TOOL")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"Model: {args.model_id}")
    print("="*70)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"\n🎮 Using device: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Load model and processor
    model, processor = load_model_and_processor(
        args.model_id,
        device=args.device,
        quantize=args.quantize
    )
    
    # Split and save models
    if not args.no_split and args.mode in ['split', 'all']:
        vision_state, language_state = split_and_save_models(model, args.output_dir, tracker)
    
    # Skip inference if requested
    if args.no_inference:
        print("\n✓ Model splitting complete. Skipping inference as requested.")
        return
    
    # Run inference based on mode
    if args.mode == 'vision' or args.mode == 'all':
        if os.path.exists(args.image):
            print("\n" + "="*70)
            print("MODE: VISION-ONLY INFERENCE")
            print("="*70)
            vision_output = run_vision_inference(model, processor, args.image, device, tracker)
        else:
            print(f"\n⚠️  Image not found: {args.image}")
    
    if args.mode == 'language' or args.mode == 'all':
        print("\n" + "="*70)
        print("MODE: LANGUAGE-ONLY INFERENCE")
        print("="*70)
        language_output = run_language_inference(
            model, 
            processor, 
            args.text,
            device,
            args.max_new_tokens,
            tracker
        )
    
    if args.mode == 'e2e' or args.mode == 'all':
        if os.path.exists(args.image):
            print("\n" + "="*70)
            print("MODE: END-TO-END INFERENCE")
            print("="*70)
            combined_output = run_combined_inference(
                model, 
                processor, 
                args.image, 
                args.image_text,
                device,
                args.max_new_tokens,
                tracker
            )
        else:
            print(f"\n⚠️  Image not found: {args.image}")
    
    # Load and verify split models if requested
    if args.load_split:
        print("\n" + "="*70)
        print("LOADING SPLIT MODELS FOR VERIFICATION")
        print("="*70)
        
        vision_loaded, language_loaded = load_split_models(args.output_dir)
        
        print(f"\n✓ Vision model parameters: {len(vision_loaded)}")
        print(f"  Sample keys: {list(vision_loaded.keys())[:3]}")
        print(f"\n✓ Language model parameters: {len(language_loaded)}")
        print(f"  Sample keys: {list(language_loaded.keys())[:3]}")
    
    # Print overall summary
    if tracker:
        print("\n" + "="*70)
        print("OVERALL PERFORMANCE SUMMARY")
        print("="*70)
        for operation, metrics in tracker.get_summary().items():
            print(f"\n{operation}:")
            print(f"  ⏱️  Duration: {metrics['duration']:.3f}s")
            print(f"  💾 CPU Memory: {metrics['cpu_mem_used']:.2f} MB")
            if torch.cuda.is_available():
                print(f"  🎮 GPU Memory: {metrics['gpu_mem_used']:.2f} MB")
                print(f"  🎮 GPU Peak: {metrics['peak_gpu_mem']:.2f} MB")
    
    print("\n" + "="*70)
    print("✅ ALL OPERATIONS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
