"""
Language Pipeline - Handles language model inference in different formats.
Supports: PyTorch (.pt), SafeTensors (.safetensors), AWQ (quantized)
"""

import torch
from typing import Optional, Dict, Any, List
import time
from safetensors.torch import load_file


class LanguagePipeline:
    """Pipeline for language model inference."""
    
    def __init__(self, model_format: str = 'pytorch', device: str = 'cuda', quantized: bool = False):
        """
        Initialize language pipeline.
        
        Args:
            model_format: 'pytorch', 'safetensors', or 'awq'
            device: 'cuda' or 'cpu'
            quantized: Whether to use quantization
        """
        self.model_format = model_format
        self.device = device
        self.quantized = quantized
        self.model = None
        self.processor = None
        self.state_dict = None
        
    def load_pytorch_model(self, model_path: str, processor):
        """Load PyTorch language model from state dict."""
        print(f"Loading PyTorch language model from: {model_path}")
        self.state_dict = torch.load(model_path, map_location=self.device)
        self.processor = processor
        print(f"✓ Loaded {len(self.state_dict)} language parameters")
        return self.state_dict
    
    def load_safetensors_model(self, model_path: str, processor):
        """Load language model from SafeTensors format."""
        print(f"Loading SafeTensors language model from: {model_path}")
        self.state_dict = load_file(model_path)
        self.processor = processor
        print(f"✓ Loaded {len(self.state_dict)} language parameters (SafeTensors)")
        return self.state_dict
    
    def load_awq_model(self, model_path: str, processor):
        """
        Load AWQ quantized language model.
        Note: This is a placeholder for future AWQ integration.
        """
        print(f"Loading AWQ quantized language model from: {model_path}")
        print("⚠️  AWQ support coming soon - using SafeTensors for now")
        
        # For now, load as safetensors
        # TODO: Integrate AutoAWQ when needed
        # from awq import AutoAWQForCausalLM
        # self.model = AutoAWQForCausalLM.from_quantized(model_path)
        
        self.state_dict = load_file(model_path)
        self.processor = processor
        print(f"✓ Loaded {len(self.state_dict)} parameters")
        return self.state_dict
    
    def load_model_weights(self, full_model, state_dict: Optional[Dict] = None):
        """
        Load state dict into full model (for separated pipeline).
        
        Args:
            full_model: Full VL model
            state_dict: State dict to load (uses self.state_dict if None)
        """
        if state_dict is None:
            state_dict = self.state_dict
        
        if state_dict is None:
            raise ValueError("No state dict available to load")
        
        # Load only language model weights
        missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
        print(f"✓ Loaded language weights into model")
        print(f"  Missing keys: {len(missing)} (expected - vision params)")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        self.model = full_model
        return full_model
    
    def preprocess_text(self, text: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        """
        Preprocess text for language model.
        
        Args:
            text: Input text
            return_tensors: 'pt' for PyTorch tensors
            
        Returns:
            Dictionary of preprocessed inputs
        """
        if self.processor is None:
            raise ValueError("Processor not set")
        
        # Create text-only message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        
        # Process text input
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=return_tensors
        )
        
        if return_tensors == "pt":
            inputs = inputs.to(self.device)
        
        return inputs
    
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 128, **kwargs) -> torch.Tensor:
        """
        Generate text using language model.
        
        Args:
            inputs: Input tensor
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        with torch.no_grad():
            generated = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        
        return generated
    
    def decode_output(self, generated: torch.Tensor, input_length: int) -> str:
        """
        Decode generated tokens to text.
        
        Args:
            generated: Generated token tensor
            input_length: Length of input tokens (to trim)
            
        Returns:
            Decoded text
        """
        if self.processor is None:
            raise ValueError("Processor not set")
        
        # Trim input tokens
        trimmed = generated[:, input_length:]
        
        # Decode
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return text
    
    def run_inference(self, text: str, max_new_tokens: int = 128, **kwargs) -> Dict[str, Any]:
        """
        Run language inference with performance tracking.
        
        Args:
            text: Input text
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing results and metadata
        """
        start_time = time.time()
        
        # Preprocess
        inputs = self.preprocess_text(text)
        preprocess_time = time.time() - start_time
        input_length = inputs.shape[-1]
        
        # Generate
        gen_start = time.time()
        generated = self.generate(inputs, max_new_tokens, **kwargs)
        gen_time = time.time() - gen_start
        
        # Decode
        decode_start = time.time()
        output_text = self.decode_output(generated, input_length)
        decode_time = time.time() - decode_start
        
        total_time = time.time() - start_time
        output_tokens = generated.shape[-1] - input_length
        tokens_per_sec = output_tokens / gen_time if gen_time > 0 else 0
        
        return {
            'input_text': text,
            'output_text': output_text,
            'input_tokens': input_length,
            'output_tokens': output_tokens,
            'total_tokens': generated.shape[-1],
            'preprocessing_time': preprocess_time,
            'generation_time': gen_time,
            'decoding_time': decode_time,
            'total_time': total_time,
            'tokens_per_sec': tokens_per_sec
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print inference results."""
        print(f"\n{'='*70}")
        print(f"LANGUAGE PIPELINE INFERENCE - {self.model_format.upper()}")
        print(f"{'='*70}")
        print(f"🎮 Device: {self.device}")
        print(f"🔧 Quantized: {self.quantized}")
        print(f"\n💬 Input: '{results['input_text']}'")
        print(f"\n✨ Output: '{results['output_text']}'")
        print(f"\nToken Statistics:")
        print(f"  Input tokens: {results['input_tokens']}")
        print(f"  Output tokens: {results['output_tokens']}")
        print(f"  Total tokens: {results['total_tokens']}")
        print(f"\nPerformance:")
        print(f"  ⏱️  Preprocessing: {results['preprocessing_time']:.3f}s")
        print(f"  ⏱️  Generation: {results['generation_time']:.3f}s")
        print(f"  ⏱️  Decoding: {results['decoding_time']:.3f}s")
        print(f"  ⏱️  Total: {results['total_time']:.3f}s")
        print(f"  ⚡ Tokens/sec: {results['tokens_per_sec']:.2f}")
        print(f"{'='*70}\n")

