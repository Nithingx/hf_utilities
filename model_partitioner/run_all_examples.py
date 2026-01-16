#!/usr/bin/env python3
"""
Comprehensive example script demonstrating all modes of the Model Partitioner.
This script runs all modes and provides a comparison of results and performance.
"""

import subprocess
import sys
import os
import json
from datetime import datetime


class ExampleRunner:
    """Run and compare all inference modes."""
    
    def __init__(self, image_path: str = "demo.jpg", text_prompt: str = "Describe this image succinctly."):
        self.image_path = image_path
        self.text_prompt = text_prompt
        self.results = {}
        
    def run_mode(self, mode: str, description: str):
        """Run a specific mode and capture results."""
        print(f"\n{'='*70}")
        print(f"Running Mode: {mode}")
        print(f"Description: {description}")
        print(f"{'='*70}\n")
        
        cmd = [
            sys.executable,
            "model_partitioner_v2.py",
            "--mode", mode,
            "--image", self.image_path,
            "--text", self.text_prompt,
            "--max-tokens", "128"
        ]
        
        try:
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            self.results[mode] = {
                'success': True,
                'duration': duration,
                'output': result.stdout,
                'error': result.stderr,
                'timestamp': start_time.isoformat()
            }
            
            print(f"✓ Mode '{mode}' completed successfully in {duration:.2f}s")
            
            # Print relevant output
            if result.stdout:
                print("\nOutput preview:")
                lines = result.stdout.split('\n')
                for line in lines[-20:]:  # Last 20 lines
                    if line.strip():
                        print(f"  {line}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Mode '{mode}' failed")
            print(f"Error: {e.stderr}")
            self.results[mode] = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_all_modes(self):
        """Run all inference modes."""
        print(f"\n{'='*70}")
        print("MODEL PARTITIONER - COMPREHENSIVE EXAMPLE")
        print(f"{'='*70}")
        print(f"Image: {self.image_path}")
        print(f"Prompt: {self.text_prompt}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Check if image exists
        if not os.path.exists(self.image_path):
            print(f"⚠️  Warning: Image '{self.image_path}' not found!")
            print("Please provide a valid image or create a demo.jpg file.")
            return
        
        # Mode 1: Original
        self.run_mode(
            "original",
            "Run original model as-is (baseline)"
        )
        
        # Mode 2: Split Native
        self.run_mode(
            "split_native",
            "Split into PyTorch vision + SafeTensor language"
        )
        
        # Mode 3: Convert ONNX
        self.run_mode(
            "convert_onnx",
            "Convert vision model to ONNX format"
        )
        
        # Mode 4: Run ONNX
        self.run_mode(
            "run_onnx",
            "Run with ONNX vision + SafeTensor language"
        )
        
        # Mode 5: Save Standalone
        self.run_mode(
            "save_standalone",
            "Export models with standalone inference scripts"
        )
        
        # Generate summary
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print summary of all runs."""
        print(f"\n{'='*70}")
        print("SUMMARY OF ALL MODES")
        print(f"{'='*70}\n")
        
        if not self.results:
            print("No results to display")
            return
        
        # Success/Failure summary
        successful = sum(1 for r in self.results.values() if r['success'])
        total = len(self.results)
        
        print(f"Completed: {successful}/{total} modes successful\n")
        
        # Duration comparison
        print("Performance Comparison:")
        print(f"{'Mode':<20} {'Status':<12} {'Duration':<12}")
        print("-" * 44)
        
        for mode, result in self.results.items():
            if result['success']:
                status = "✓ Success"
                duration = f"{result['duration']:.2f}s"
            else:
                status = "✗ Failed"
                duration = "N/A"
            
            print(f"{mode:<20} {status:<12} {duration:<12}")
        
        # Fastest mode
        successful_modes = {m: r for m, r in self.results.items() if r['success']}
        if successful_modes:
            fastest = min(successful_modes.items(), key=lambda x: x[1]['duration'])
            print(f"\n⚡ Fastest mode: {fastest[0]} ({fastest[1]['duration']:.2f}s)")
    
    def save_results(self):
        """Save results to JSON file."""
        output_file = f"example_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {output_file}")
        print(f"{'='*70}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive examples of all Model Partitioner modes"
    )
    parser.add_argument(
        '--image',
        type=str,
        default='demo.jpg',
        help='Path to test image'
    )
    parser.add_argument(
        '--text',
        type=str,
        default='Describe this image succinctly.',
        help='Text prompt for inference'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'original', 'split_native', 'convert_onnx', 'run_onnx', 'save_standalone'],
        default='all',
        help='Specific mode to run (default: all)'
    )
    
    args = parser.parse_args()
    
    runner = ExampleRunner(args.image, args.text)
    
    if args.mode == 'all':
        runner.run_all_modes()
    else:
        # Run single mode
        descriptions = {
            'original': 'Run original model as-is',
            'split_native': 'Split into PyTorch + SafeTensor',
            'convert_onnx': 'Convert to ONNX',
            'run_onnx': 'Run with ONNX',
            'save_standalone': 'Export standalone scripts'
        }
        runner.run_mode(args.mode, descriptions.get(args.mode, ''))
        runner.print_summary()
        runner.save_results()


if __name__ == "__main__":
    main()

