"""
ONNX Converter - Converts vision model to ONNX format.
"""

import torch
import torch.onnx
import os
from typing import Optional, Tuple
import numpy as np


class ONNXConverter:
    """Convert PyTorch vision model to ONNX format."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize ONNX converter.
        
        Args:
            device: Device for conversion ('cuda' or 'cpu')
        """
        self.device = device
    
    def export_vision_model(
        self,
        model,
        processor,
        output_path: str,
        dummy_image_path: Optional[str] = None,
        opset_version: int = 14,
        dynamic_axes: bool = True
    ) -> str:
        """
        Export vision model to ONNX format.
        
        Args:
            model: Full VL model
            processor: Model processor
            output_path: Path to save ONNX model
            dummy_image_path: Path to dummy image for tracing
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes
            
        Returns:
            Path to saved ONNX model
        """
        print(f"\n{'='*70}")
        print("EXPORTING VISION MODEL TO ONNX")
        print(f"{'='*70}")
        
        # Get vision model component
        if hasattr(model, 'visual'):
            vision_model = model.visual
        elif hasattr(model, 'vision_tower'):
            vision_model = model.vision_tower
        elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
            vision_model = model.model.visual
        else:
            raise ValueError("Could not find vision encoder in model")
        
        vision_model.eval()
        
        # Create dummy input
        print("Creating dummy input...")
        dummy_input = self._create_dummy_input(processor, dummy_image_path)
        
        # Prepare for export
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Define input/output names
        input_names = list(dummy_input.keys())
        output_names = ['output']
        
        # Define dynamic axes if requested
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                name: {0: 'batch_size'} for name in input_names + output_names
            }
        
        print(f"Input names: {input_names}")
        print(f"Output names: {output_names}")
        print(f"Opset version: {opset_version}")
        print(f"Dynamic axes: {dynamic_axes}")
        
        # Export to ONNX
        print("\nExporting model...")
        try:
            # Prepare inputs as tuple
            dummy_input_tuple = tuple(dummy_input.values())
            
            torch.onnx.export(
                vision_model,
                dummy_input_tuple,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )
            
            print(f"✓ ONNX model exported to: {output_path}")
            
            # Get file size
            size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"  File size: {size_mb:.2f} MB")
            
            # Verify the exported model
            self._verify_onnx_model(output_path, dummy_input)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during ONNX export: {e}")
            raise
    
    def _create_dummy_input(self, processor, dummy_image_path: Optional[str] = None) -> dict:
        """
        Create dummy input for ONNX export.
        
        Args:
            processor: Model processor
            dummy_image_path: Path to dummy image (creates random if None)
            
        Returns:
            Dictionary of dummy inputs
        """
        from PIL import Image
        
        if dummy_image_path and os.path.exists(dummy_image_path):
            image = Image.open(dummy_image_path)
            print(f"  Using dummy image: {dummy_image_path}")
        else:
            # Create random image
            image = Image.new('RGB', (224, 224), color='red')
            print(f"  Using generated dummy image (224x224)")
        
        # Process image
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _verify_onnx_model(self, model_path: str, dummy_input: dict):
        """
        Verify exported ONNX model.
        
        Args:
            model_path: Path to ONNX model
            dummy_input: Dummy input used for export
        """
        try:
            import onnx
            import onnxruntime as ort
            
            print("\nVerifying ONNX model...")
            
            # Check model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid")
            
            # Test inference
            print("Testing ONNX inference...")
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            
            # Convert inputs to numpy
            numpy_inputs = {k: v.cpu().numpy() for k, v in dummy_input.items()}
            
            # Run inference
            outputs = session.run(None, numpy_inputs)
            print(f"✓ ONNX inference successful")
            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output dtype: {outputs[0].dtype}")
            
        except ImportError as e:
            print(f"⚠️  Could not verify ONNX model (missing dependencies): {e}")
        except Exception as e:
            print(f"⚠️  ONNX verification warning: {e}")
    
    def optimize_onnx_model(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Optimize ONNX model for inference.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save optimized model (overwrites if None)
            
        Returns:
            Path to optimized model
        """
        try:
            from onnxruntime.transformers import optimizer
            
            if output_path is None:
                output_path = model_path.replace('.onnx', '_optimized.onnx')
            
            print(f"\n{'='*70}")
            print("OPTIMIZING ONNX MODEL")
            print(f"{'='*70}")
            print(f"Input: {model_path}")
            print(f"Output: {output_path}")
            
            # Optimize
            optimized_model = optimizer.optimize_model(
                model_path,
                model_type='bert',  # Generic transformer optimization
                num_heads=0,  # Auto-detect
                hidden_size=0  # Auto-detect
            )
            
            optimized_model.save_model_to_file(output_path)
            
            print(f"✓ Optimized model saved to: {output_path}")
            
            # Compare sizes
            original_size = os.path.getsize(model_path) / (1024**2)
            optimized_size = os.path.getsize(output_path) / (1024**2)
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Optimized size: {optimized_size:.2f} MB")
            print(f"  Reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
            
            return output_path
            
        except ImportError:
            print("⚠️  ONNX optimizer not available. Install: pip install onnxruntime-tools")
            return model_path
        except Exception as e:
            print(f"⚠️  Optimization failed: {e}")
            return model_path
    
    def quantize_onnx_model(self, model_path: str, output_path: Optional[str] = None) -> str:
        """
        Quantize ONNX model to INT8.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save quantized model
            
        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if output_path is None:
                output_path = model_path.replace('.onnx', '_quantized.onnx')
            
            print(f"\n{'='*70}")
            print("QUANTIZING ONNX MODEL")
            print(f"{'='*70}")
            print(f"Input: {model_path}")
            print(f"Output: {output_path}")
            
            # Quantize
            quantize_dynamic(
                model_path,
                output_path,
                weight_type=QuantType.QUInt8
            )
            
            print(f"✓ Quantized model saved to: {output_path}")
            
            # Compare sizes
            original_size = os.path.getsize(model_path) / (1024**2)
            quantized_size = os.path.getsize(output_path) / (1024**2)
            print(f"  Original size: {original_size:.2f} MB")
            print(f"  Quantized size: {quantized_size:.2f} MB")
            print(f"  Reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
            
            return output_path
            
        except ImportError:
            print("⚠️  ONNX quantization not available. Install: pip install onnxruntime")
            return model_path
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
            return model_path

