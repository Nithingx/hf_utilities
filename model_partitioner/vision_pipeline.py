"""
Vision Pipeline - Handles vision model inference in different formats.
Supports: PyTorch (.pt), ONNX (.onnx)
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Union
from PIL import Image
import time


class VisionPipeline:
    """Pipeline for vision model inference."""
    
    def __init__(self, model_format: str = 'pytorch', device: str = 'cuda'):
        """
        Initialize vision pipeline.
        
        Args:
            model_format: 'pytorch' or 'onnx'
            device: 'cuda' or 'cpu'
        """
        self.model_format = model_format
        self.device = device
        self.model = None
        self.processor = None
        
    def load_pytorch_model(self, model_path: str, processor):
        """Load PyTorch vision model from state dict."""
        print(f"Loading PyTorch vision model from: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device)
        self.processor = processor
        print(f"✓ Loaded {len(state_dict)} vision parameters")
        return state_dict
    
    def load_onnx_model(self, model_path: str):
        """Load ONNX vision model."""
        import onnxruntime as ort
        
        print(f"Loading ONNX vision model from: {model_path}")
        
        # Configure ONNX Runtime session
        providers = self._get_onnx_providers()
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"✓ ONNX model loaded")
        print(f"  Providers: {self.model.get_providers()}")
        print(f"  Input names: {[i.name for i in self.model.get_inputs()]}")
        print(f"  Output names: {[o.name for o in self.model.get_outputs()]}")
        
        return self.model
    
    def _get_onnx_providers(self):
        """Get ONNX Runtime execution providers."""
        providers = []
        
        # Check for VitisAI EP (for future use)
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            if 'VitisAIExecutionProvider' in available_providers:
                print("  🎯 VitisAI Execution Provider available")
                providers.append('VitisAIExecutionProvider')
        except:
            pass
        
        # Add CUDA if available
        if self.device == 'cuda' and torch.cuda.is_available():
            providers.append('CUDAExecutionProvider')
        
        # CPU fallback
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Preprocess image for vision model.
        
        Args:
            image: Path to image or PIL Image
            
        Returns:
            Dictionary of preprocessed inputs
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        if self.processor is None:
            raise ValueError("Processor not set. Load PyTorch model first or set processor manually.")
        
        # Process image
        inputs = self.processor(images=[image], return_tensors="pt")
        
        # Convert to numpy for ONNX if needed
        if self.model_format == 'onnx':
            inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def infer_pytorch(self, full_model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run inference using PyTorch vision model.
        
        Args:
            full_model: Full VL model (to access vision encoder)
            inputs: Preprocessed inputs
            
        Returns:
            Vision features tensor
        """
        with torch.no_grad():
            # Try different vision encoder access patterns
            if hasattr(full_model, 'visual'):
                features = full_model.visual(**inputs)
            elif hasattr(full_model, 'vision_tower'):
                features = full_model.vision_tower(**inputs)
            elif hasattr(full_model, 'model') and hasattr(full_model.model, 'visual'):
                features = full_model.model.visual(**inputs)
            else:
                raise ValueError("Could not find vision encoder in model")
            
            # Extract tensor from output
            if isinstance(features, torch.Tensor):
                return features
            elif hasattr(features, 'last_hidden_state'):
                return features.last_hidden_state
            else:
                return features[0] if isinstance(features, tuple) else features
    
    def infer_onnx(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run inference using ONNX vision model.
        
        Args:
            inputs: Preprocessed inputs as numpy arrays
            
        Returns:
            Vision features as numpy array
        """
        if self.model is None:
            raise ValueError("ONNX model not loaded")
        
        # Run ONNX inference
        outputs = self.model.run(None, inputs)
        return outputs[0]  # Return first output
    
    def run_inference(self, image: Union[str, Image.Image], full_model=None) -> Dict[str, Any]:
        """
        Run vision inference with performance tracking.
        
        Args:
            image: Path to image or PIL Image
            full_model: Full model (required for PyTorch mode)
            
        Returns:
            Dictionary containing features and metadata
        """
        start_time = time.time()
        
        # Preprocess
        inputs = self.preprocess_image(image)
        preprocess_time = time.time() - start_time
        
        # Inference
        infer_start = time.time()
        if self.model_format == 'pytorch':
            if full_model is None:
                raise ValueError("full_model required for PyTorch inference")
            features = self.infer_pytorch(full_model, inputs)
            features_np = features.cpu().numpy()
        else:  # onnx
            features_np = self.infer_onnx(inputs)
        
        infer_time = time.time() - infer_start
        total_time = time.time() - start_time
        
        return {
            'features': features_np,
            'shape': features_np.shape,
            'dtype': str(features_np.dtype),
            'mean': float(np.mean(features_np)),
            'std': float(np.std(features_np)),
            'min': float(np.min(features_np)),
            'max': float(np.max(features_np)),
            'preprocessing_time': preprocess_time,
            'inference_time': infer_time,
            'total_time': total_time
        }
    
    def print_results(self, results: Dict[str, Any], image_path: str):
        """Print inference results."""
        print(f"\n{'='*70}")
        print(f"VISION PIPELINE INFERENCE - {self.model_format.upper()}")
        print(f"{'='*70}")
        print(f"📷 Image: {image_path}")
        print(f"🎮 Device: {self.device}")
        print(f"\nFeature Statistics:")
        print(f"  Shape: {results['shape']}")
        print(f"  dtype: {results['dtype']}")
        print(f"  Mean: {results['mean']:.4f}")
        print(f"  Std: {results['std']:.4f}")
        print(f"  Min: {results['min']:.4f}")
        print(f"  Max: {results['max']:.4f}")
        print(f"\nPerformance:")
        print(f"  ⏱️  Preprocessing: {results['preprocessing_time']:.3f}s")
        print(f"  ⏱️  Inference: {results['inference_time']:.3f}s")
        print(f"  ⏱️  Total: {results['total_time']:.3f}s")
        print(f"{'='*70}\n")

