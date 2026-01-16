#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for Vision-Language Model Partitioner

Benchmarks all inference modes with detailed performance metrics:
- Latency (preprocessing, inference, decoding)
- Throughput (tokens/sec, images/sec)
- Memory usage (CPU, GPU, peak)
- Model sizes
- Multi-run statistics (mean, std, min, max)

Usage:
    python benchmark.py --image demo.jpg --runs 5
    python benchmark.py --image demo.jpg --runs 10 --modes original split_native run_onnx
    python benchmark.py --image demo.jpg --compare-devices
    python benchmark.py --image demo.jpg --compare-quantization
"""

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import os
import json
import argparse
import time
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import subprocess
import sys

# Import pipeline modules
from vision_pipeline import VisionPipeline
from language_pipeline import LanguagePipeline
from onnx_converter import ONNXConverter


class BenchmarkMetrics:
    """Store and calculate benchmark metrics."""
    
    def __init__(self):
        self.runs = []
    
    def add_run(self, metrics: Dict[str, Any]):
        """Add metrics from a single run."""
        self.runs.append(metrics)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics across all runs."""
        if not self.runs:
            return {}
        
        stats = {}
        
        # Collect all numeric metrics
        numeric_metrics = {}
        for run in self.runs:
            for key, value in run.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)
        
        # Calculate statistics
        for key, values in numeric_metrics.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values
            }
        
        stats['num_runs'] = len(self.runs)
        return stats


class ModelBenchmark:
    """Benchmark different inference modes."""
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = 'auto',
        output_dir: str = 'split_models'
    ):
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
        self.process = psutil.Process()
        
        # Results storage
        self.results = {}
        
    def load_model(self, quantize: bool = False):
        """Load model for benchmarking."""
        print(f"\n{'='*70}")
        print("LOADING MODEL FOR BENCHMARKING")
        print(f"{'='*70}")
        print(f"Model: {self.model_id}")
        print(f"Device: {self.device}")
        print(f"Quantize: {quantize}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
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
        
        print(f"✓ Model loaded")
    
    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        metrics = {
            'cpu_memory_mb': self.process.memory_info().rss / (1024**2)
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
        
        return metrics
    
    def benchmark_original_mode(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int = 128,
        warmup: bool = True
    ) -> Dict[str, Any]:
        """Benchmark original model inference."""
        print(f"\n{'='*70}")
        print("BENCHMARKING: ORIGINAL MODE")
        print(f"{'='*70}")
        
        from PIL import Image
        
        if warmup:
            print("Warming up...")
            self._run_original_inference(image_path, text_prompt, max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Measure
        start_mem = self.measure_memory()
        start_time = time.time()
        
        output_text, token_count = self._run_original_inference(
            image_path, text_prompt, max_new_tokens
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_mem = self.measure_memory()
        
        duration = end_time - start_time
        
        metrics = {
            'mode': 'original',
            'duration': duration,
            'tokens_generated': token_count,
            'tokens_per_sec': token_count / duration if duration > 0 else 0,
            'cpu_memory_delta_mb': end_mem['cpu_memory_mb'] - start_mem['cpu_memory_mb'],
            'output_text': output_text
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory_delta_mb'] = end_mem['gpu_memory_mb'] - start_mem['gpu_memory_mb']
            metrics['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return metrics
    
    def _run_original_inference(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int
    ):
        """Run original model inference."""
        from PIL import Image
        
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
        
        return output_text, trimmed.shape[-1]
    
    def benchmark_split_native_mode(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int = 128,
        warmup: bool = True
    ) -> Dict[str, Any]:
        """Benchmark split native mode."""
        print(f"\n{'='*70}")
        print("BENCHMARKING: SPLIT NATIVE MODE")
        print(f"{'='*70}")
        
        # Ensure models are split
        vision_path = os.path.join(self.output_dir, "vision_model", "vision_model.pt")
        language_path = os.path.join(self.output_dir, "language_model", "language_model.safetensors")
        
        if not os.path.exists(vision_path) or not os.path.exists(language_path):
            print("Models not split. Splitting now...")
            self._split_models()
        
        if warmup:
            print("Warming up...")
            self._run_split_native_inference(image_path, text_prompt, max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Reset and measure
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_mem = self.measure_memory()
        start_time = time.time()
        
        output_text, token_count = self._run_split_native_inference(
            image_path, text_prompt, max_new_tokens
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_mem = self.measure_memory()
        duration = end_time - start_time
        
        metrics = {
            'mode': 'split_native',
            'duration': duration,
            'tokens_generated': token_count,
            'tokens_per_sec': token_count / duration if duration > 0 else 0,
            'cpu_memory_delta_mb': end_mem['cpu_memory_mb'] - start_mem['cpu_memory_mb'],
            'output_text': output_text,
            'vision_model_size_mb': os.path.getsize(vision_path) / (1024**2),
            'language_model_size_mb': os.path.getsize(language_path) / (1024**2)
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory_delta_mb'] = end_mem['gpu_memory_mb'] - start_mem['gpu_memory_mb']
            metrics['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return metrics
    
    def _split_models(self):
        """Split models if not already done."""
        os.makedirs(os.path.join(self.output_dir, "vision_model"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "language_model"), exist_ok=True)
        
        from safetensors.torch import save_file
        
        vision_state = {}
        language_state = {}
        
        for name, param in self.model.named_parameters():
            if 'visual' in name or 'vision' in name:
                vision_state[name] = param.detach().cpu()
            else:
                language_state[name] = param.detach().cpu()
        
        vision_path = os.path.join(self.output_dir, "vision_model", "vision_model.pt")
        language_path = os.path.join(self.output_dir, "language_model", "language_model.safetensors")
        
        torch.save(vision_state, vision_path)
        save_file(language_state, language_path)
        
        print(f"✓ Models split and saved")
    
    def _run_split_native_inference(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int
    ):
        """Run inference with split native models."""
        return self._run_original_inference(image_path, text_prompt, max_new_tokens)
    
    def benchmark_onnx_mode(
        self,
        image_path: str,
        text_prompt: str,
        max_new_tokens: int = 128,
        warmup: bool = True
    ) -> Dict[str, Any]:
        """Benchmark ONNX inference mode."""
        print(f"\n{'='*70}")
        print("BENCHMARKING: ONNX MODE")
        print(f"{'='*70}")
        
        onnx_path = os.path.join(self.output_dir, "onnx_model", "vision_model.onnx")
        
        if not os.path.exists(onnx_path):
            print("ONNX model not found. Converting...")
            self._convert_to_onnx(image_path)
        
        # Initialize vision pipeline
        vision_pipeline = VisionPipeline(model_format='onnx', device=self.device)
        vision_pipeline.load_onnx_model(onnx_path)
        vision_pipeline.processor = self.processor
        
        if warmup:
            print("Warming up...")
            vision_pipeline.run_inference(image_path)
            self._run_original_inference(image_path, text_prompt, max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark vision inference separately
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        vision_start = time.time()
        vision_results = vision_pipeline.run_inference(image_path)
        vision_time = time.time() - vision_start
        
        # Benchmark full E2E
        start_mem = self.measure_memory()
        start_time = time.time()
        
        output_text, token_count = self._run_original_inference(
            image_path, text_prompt, max_new_tokens
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_mem = self.measure_memory()
        duration = end_time - start_time
        
        metrics = {
            'mode': 'onnx',
            'duration': duration,
            'vision_inference_time': vision_time,
            'tokens_generated': token_count,
            'tokens_per_sec': token_count / duration if duration > 0 else 0,
            'cpu_memory_delta_mb': end_mem['cpu_memory_mb'] - start_mem['cpu_memory_mb'],
            'output_text': output_text,
            'onnx_model_size_mb': os.path.getsize(onnx_path) / (1024**2)
        }
        
        if torch.cuda.is_available():
            metrics['gpu_memory_delta_mb'] = end_mem['gpu_memory_mb'] - start_mem['gpu_memory_mb']
            metrics['gpu_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
        
        return metrics
    
    def _convert_to_onnx(self, dummy_image_path: str):
        """Convert vision model to ONNX."""
        os.makedirs(os.path.join(self.output_dir, "onnx_model"), exist_ok=True)
        onnx_path = os.path.join(self.output_dir, "onnx_model", "vision_model.onnx")
        
        converter = ONNXConverter(device=self.device)
        converter.export_vision_model(
            self.model,
            self.processor,
            onnx_path,
            dummy_image_path=dummy_image_path
        )
    
    def run_benchmark_suite(
        self,
        image_path: str,
        text_prompt: str,
        modes: List[str],
        num_runs: int = 5,
        max_new_tokens: int = 128,
        warmup: bool = True
    ) -> Dict[str, BenchmarkMetrics]:
        """Run complete benchmark suite."""
        print(f"\n{'='*70}")
        print("STARTING BENCHMARK SUITE")
        print(f"{'='*70}")
        print(f"Image: {image_path}")
        print(f"Prompt: {text_prompt}")
        print(f"Modes: {modes}")
        print(f"Runs per mode: {num_runs}")
        print(f"Max tokens: {max_new_tokens}")
        print(f"Warmup: {warmup}")
        print(f"{'='*70}\n")
        
        results = {}
        
        for mode in modes:
            print(f"\n{'='*70}")
            print(f"BENCHMARKING MODE: {mode.upper()}")
            print(f"{'='*70}")
            
            benchmark_metrics = BenchmarkMetrics()
            
            for run in range(num_runs):
                print(f"\n--- Run {run + 1}/{num_runs} ---")
                
                try:
                    if mode == 'original':
                        metrics = self.benchmark_original_mode(
                            image_path, text_prompt, max_new_tokens, warmup=(warmup and run == 0)
                        )
                    elif mode == 'split_native':
                        metrics = self.benchmark_split_native_mode(
                            image_path, text_prompt, max_new_tokens, warmup=(warmup and run == 0)
                        )
                    elif mode == 'run_onnx':
                        metrics = self.benchmark_onnx_mode(
                            image_path, text_prompt, max_new_tokens, warmup=(warmup and run == 0)
                        )
                    else:
                        print(f"⚠️  Unknown mode: {mode}")
                        continue
                    
                    benchmark_metrics.add_run(metrics)
                    
                    print(f"✓ Run {run + 1} complete: {metrics['duration']:.3f}s, "
                          f"{metrics['tokens_per_sec']:.2f} tok/s")
                    
                except Exception as e:
                    print(f"❌ Run {run + 1} failed: {e}")
                    continue
            
            results[mode] = benchmark_metrics
        
        self.results = results
        return results
    
    def print_results(self):
        """Print benchmark results in a formatted table."""
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        if not self.results:
            print("No results to display")
            return
        
        # Print table header
        print(f"{'Mode':<15} {'Metric':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 86)
        
        for mode, benchmark_metrics in self.results.items():
            stats = benchmark_metrics.get_statistics()
            
            if not stats:
                print(f"{mode:<15} No data")
                continue
            
            # Key metrics to display
            key_metrics = [
                ('duration', 'Duration (s)'),
                ('tokens_per_sec', 'Tokens/sec'),
                ('tokens_generated', 'Tokens Generated'),
            ]
            
            if 'gpu_peak_memory_mb' in stats:
                key_metrics.append(('gpu_peak_memory_mb', 'GPU Peak Mem (MB)'))
            
            key_metrics.append(('cpu_memory_delta_mb', 'CPU Mem Delta (MB)'))
            
            for i, (metric_key, metric_name) in enumerate(key_metrics):
                if metric_key in stats:
                    s = stats[metric_key]
                    mode_col = mode if i == 0 else ''
                    print(f"{mode_col:<15} {metric_name:<25} {s['mean']:<12.3f} "
                          f"{s['std']:<12.3f} {s['min']:<12.3f} {s['max']:<12.3f}")
            
            print()
        
        # Print comparison
        print(f"\n{'='*70}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*70}\n")
        
        # Compare duration
        durations = {}
        for mode, benchmark_metrics in self.results.items():
            stats = benchmark_metrics.get_statistics()
            if 'duration' in stats:
                durations[mode] = stats['duration']['mean']
        
        if durations:
            fastest_mode = min(durations, key=durations.get)
            print(f"⚡ Fastest Mode: {fastest_mode} ({durations[fastest_mode]:.3f}s)")
            
            print(f"\nRelative Performance (vs {fastest_mode}):")
            baseline = durations[fastest_mode]
            for mode, duration in sorted(durations.items(), key=lambda x: x[1]):
                speedup = baseline / duration
                percent = ((duration - baseline) / baseline) * 100
                print(f"  {mode:<15}: {duration:.3f}s ({speedup:.2f}x, {percent:+.1f}%)")
        
        # Compare throughput
        print(f"\n{'='*70}")
        print("THROUGHPUT COMPARISON")
        print(f"{'='*70}\n")
        
        throughputs = {}
        for mode, benchmark_metrics in self.results.items():
            stats = benchmark_metrics.get_statistics()
            if 'tokens_per_sec' in stats:
                throughputs[mode] = stats['tokens_per_sec']['mean']
        
        if throughputs:
            for mode, tps in sorted(throughputs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {mode:<15}: {tps:.2f} tokens/sec")
    
    def save_results(self, output_file: Optional[str] = None):
        """Save results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"benchmark_results_{timestamp}.json"
        
        results_dict = {}
        for mode, benchmark_metrics in self.results.items():
            results_dict[mode] = benchmark_metrics.get_statistics()
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")
    
    def save_results_csv(self, output_file: Optional[str] = None):
        """Save results to CSV file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"benchmark_results_{timestamp}.csv"
        
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Mode', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'Median', 'Num Runs'
            ])
            
            # Data
            for mode, benchmark_metrics in self.results.items():
                stats = benchmark_metrics.get_statistics()
                
                for metric_name, metric_stats in stats.items():
                    if metric_name == 'num_runs' or not isinstance(metric_stats, dict):
                        continue
                    
                    writer.writerow([
                        mode,
                        metric_name,
                        f"{metric_stats['mean']:.4f}",
                        f"{metric_stats['std']:.4f}",
                        f"{metric_stats['min']:.4f}",
                        f"{metric_stats['max']:.4f}",
                        f"{metric_stats['median']:.4f}",
                        stats.get('num_runs', 0)
                    ])
        
        print(f"📊 CSV results saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Benchmark Suite for Model Partitioner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--image', type=str, default='demo.jpg', help='Test image path')
    parser.add_argument('--text', type=str, default='Describe this image succinctly.', help='Text prompt')
    parser.add_argument('--modes', nargs='+', default=['original', 'split_native', 'run_onnx'],
                       choices=['original', 'split_native', 'run_onnx'],
                       help='Modes to benchmark')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per mode')
    parser.add_argument('--max-tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--model-id', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--output-dir', type=str, default='split_models')
    parser.add_argument('--no-warmup', action='store_true', help='Disable warmup run')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--csv', action='store_true', help='Also save results as CSV')
    parser.add_argument('--quantize', action='store_true', help='Use quantization')
    parser.add_argument('--compare-devices', action='store_true', help='Compare CUDA vs CPU')
    parser.add_argument('--compare-quantization', action='store_true', help='Compare quantized vs non-quantized')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print("MODEL PARTITIONER - BENCHMARK SUITE")
    print(f"{'='*70}\n")
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    # Standard benchmark
    if not args.compare_devices and not args.compare_quantization:
        benchmark = ModelBenchmark(args.model_id, args.device, args.output_dir)
        benchmark.load_model(quantize=args.quantize)
        
        results = benchmark.run_benchmark_suite(
            args.image,
            args.text,
            args.modes,
            num_runs=args.runs,
            max_new_tokens=args.max_tokens,
            warmup=not args.no_warmup
        )
        
        benchmark.print_results()
        benchmark.save_results(args.output)
        
        if args.csv:
            benchmark.save_results_csv(args.output.replace('.json', '.csv') if args.output else None)
    
    # Compare devices
    elif args.compare_devices:
        print("\n🔍 COMPARING CUDA vs CPU\n")
        
        for device in ['cuda', 'cpu']:
            if device == 'cuda' and not torch.cuda.is_available():
                print(f"⚠️  Skipping CUDA (not available)")
                continue
            
            print(f"\n{'='*70}")
            print(f"BENCHMARKING ON: {device.upper()}")
            print(f"{'='*70}")
            
            benchmark = ModelBenchmark(args.model_id, device, args.output_dir)
            benchmark.load_model(quantize=args.quantize)
            
            results = benchmark.run_benchmark_suite(
                args.image,
                args.text,
                args.modes,
                num_runs=args.runs,
                max_new_tokens=args.max_tokens,
                warmup=not args.no_warmup
            )
            
            benchmark.print_results()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            benchmark.save_results(f"benchmark_{device}_{timestamp}.json")
    
    # Compare quantization
    elif args.compare_quantization:
        print("\n🔍 COMPARING QUANTIZED vs NON-QUANTIZED\n")
        
        for quantize in [False, True]:
            quant_str = "quantized" if quantize else "non_quantized"
            
            print(f"\n{'='*70}")
            print(f"BENCHMARKING: {quant_str.upper()}")
            print(f"{'='*70}")
            
            benchmark = ModelBenchmark(args.model_id, args.device, args.output_dir)
            benchmark.load_model(quantize=quantize)
            
            results = benchmark.run_benchmark_suite(
                args.image,
                args.text,
                args.modes,
                num_runs=args.runs,
                max_new_tokens=args.max_tokens,
                warmup=not args.no_warmup
            )
            
            benchmark.print_results()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            benchmark.save_results(f"benchmark_{quant_str}_{timestamp}.json")
    
    print(f"\n{'='*70}")
    print("✅ BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

