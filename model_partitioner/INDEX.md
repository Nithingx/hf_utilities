# Model Partitioner - Complete Index

Welcome to the Vision-Language Model Partitioner toolkit! This index will help you navigate all the components.

## 📚 Documentation

| File | Purpose | Audience |
|------|---------|----------|
| [README.md](README.md) | Complete documentation with all features and options | All users |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute getting started guide | New users |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Technical implementation details | Developers |
| [INDEX.md](INDEX.md) | This file - navigation guide | All users |

## 🔧 Core Implementation Files

### Main Orchestrator
- **`model_partitioner_v2.py`** (447 lines)
  - Main entry point for all operations
  - Implements all 5 run modes
  - Command-line interface
  - Performance tracking

### Pipeline Components
- **`vision_pipeline.py`** (184 lines)
  - Vision model inference (PyTorch, ONNX)
  - VitisAI EP support (ready)
  - Image preprocessing
  - Feature extraction

- **`language_pipeline.py`** (199 lines)
  - Language model inference (PyTorch, SafeTensors)
  - AWQ support (ready)
  - Text generation
  - Token management

### Conversion Utilities
- **`onnx_converter.py`** (276 lines)
  - PyTorch to ONNX conversion
  - Model validation
  - Optimization tools
  - Quantization support

### Legacy/Original
- **`model_partitioner.py`** (original version)
  - First implementation
  - Preserved for reference
  - Feature-complete with basic modes

## 🎯 Run Modes

### Mode 1: Original Model
**Command:** `python model_partitioner_v2.py --mode original --image demo.jpg`

**What it does:**
- Runs unmodified VL model
- Provides baseline metrics
- No splitting or conversion

**Use case:** Baseline performance measurement

---

### Mode 2: Split Native
**Command:** `python model_partitioner_v2.py --mode split_native --image demo.jpg`

**What it does:**
- Splits vision → PyTorch (.pt)
- Splits language → SafeTensors (.safetensors)
- Runs with separated pipeline

**Use case:** Native format separation, debugging

---

### Mode 3: Convert ONNX
**Command:** `python model_partitioner_v2.py --mode convert_onnx --image demo.jpg`

**What it does:**
- Exports vision → ONNX
- Keeps language → SafeTensors
- Validates conversion

**Use case:** ONNX export, optimization preparation

---

### Mode 4: Run ONNX
**Command:** `python model_partitioner_v2.py --mode run_onnx --image demo.jpg`

**What it does:**
- Uses ONNX vision model
- Uses SafeTensor language model
- Runs end-to-end inference

**Use case:** Production deployment, optimized inference

---

### Mode 5: Save Standalone
**Command:** `python model_partitioner_v2.py --mode save_standalone --image demo.jpg`

**What it does:**
- Creates standalone directories
- Copies models with inference scripts
- Enables independent testing

**Use case:** Component debugging, independent deployment

---

### Mode 6: All
**Command:** `python model_partitioner_v2.py --mode all --image demo.jpg`

**What it does:**
- Runs all modes sequentially
- Compares performance
- Generates comprehensive report

**Use case:** Complete testing, performance comparison

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Your First Test
```bash
python model_partitioner_v2.py --mode original --image demo.jpg
```

### 3. Run All Modes
```bash
python run_all_examples.py --image demo.jpg
```

### 4. Check Results
```bash
# Results saved in split_models/ directory
ls -la split_models/
```

## 📁 File Organization

```
model_partitioner/
│
├── 📘 Documentation
│   ├── README.md                      # Full documentation
│   ├── QUICKSTART.md                 # Getting started
│   ├── IMPLEMENTATION_SUMMARY.md     # Technical details
│   └── INDEX.md                      # This file
│
├── 🔧 Core Code
│   ├── model_partitioner_v2.py       # Main orchestrator ⭐
│   ├── vision_pipeline.py             # Vision inference
│   ├── language_pipeline.py           # Language inference
│   ├── onnx_converter.py             # ONNX conversion
│   └── model_partitioner.py          # Original version
│
├── 🎓 Examples
│   ├── run_all_examples.py           # Example runner
│   └── requirements.txt              # Dependencies
│
└── 📦 Generated (runtime)
    └── split_models/
        ├── vision_model/              # .pt weights
        ├── language_model/            # .safetensors weights
        ├── onnx_model/               # ONNX models
        ├── standalone/               # Standalone scripts
        └── model_config.json         # Metadata
```

## 🎯 Common Use Cases

### Use Case 1: Quick Test
```bash
python model_partitioner_v2.py --mode original --image demo.jpg
```

### Use Case 2: Production Deployment
```bash
# Convert to optimized formats
python model_partitioner_v2.py --mode convert_onnx --image demo.jpg

# Run with optimized formats
python model_partitioner_v2.py --mode run_onnx --image demo.jpg
```

### Use Case 3: Debug Components
```bash
# Create standalone scripts
python model_partitioner_v2.py --mode save_standalone --image demo.jpg

# Test vision independently
cd split_models/standalone/vision
python vision_inference.py --image ../../../demo.jpg

# Test language independently
cd ../language
python language_inference.py --text "Your prompt"
```

### Use Case 4: Performance Comparison
```bash
# Run all modes and compare
python run_all_examples.py --image demo.jpg

# Check results
cat example_results_*.json
```

### Use Case 5: Memory-Constrained Environment
```bash
# Use quantization
python model_partitioner_v2.py --mode original --quantize --image demo.jpg
```

## 🔍 Finding What You Need

### "I want to understand the project"
→ Start with [README.md](README.md)

### "I want to get started quickly"
→ Follow [QUICKSTART.md](QUICKSTART.md)

### "I want to understand the implementation"
→ Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

### "I want to modify the vision pipeline"
→ Edit `vision_pipeline.py`

### "I want to modify the language pipeline"
→ Edit `language_pipeline.py`

### "I want to add ONNX optimizations"
→ Edit `onnx_converter.py`

### "I want to add new modes"
→ Edit `model_partitioner_v2.py`

### "I want to integrate VitisAI"
→ Update `vision_pipeline.py:_get_onnx_providers()`

### "I want to integrate AWQ"
→ Update `language_pipeline.py:load_awq_model()`

## 📊 Output Files Reference

### After Running Any Mode:

**Configuration:**
- `split_models/model_config.json` - Model metadata

**Vision Models:**
- `split_models/vision_model/vision_model.pt` - PyTorch format
- `split_models/onnx_model/vision_model.onnx` - ONNX format

**Language Models:**
- `split_models/language_model/language_model.safetensors` - SafeTensors format

**Standalone:**
- `split_models/standalone/vision/` - Vision-only inference
- `split_models/standalone/language/` - Language-only inference

**Results:**
- `example_results_*.json` - Performance comparison (from run_all_examples.py)

## 🎓 Learning Path

### Beginner
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `--mode original`
3. Run `--mode split_native`
4. Explore output files

### Intermediate
1. Read [README.md](README.md) sections
2. Try all modes
3. Compare performance
4. Modify parameters

### Advanced
1. Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. Understand pipeline architecture
3. Modify pipeline code
4. Integrate VitisAI/AWQ

## 🔧 Command-Line Reference

### Basic Options
```bash
--mode {original|split_native|convert_onnx|run_onnx|save_standalone|all}
--image PATH           # Input image
--text TEXT            # Text prompt
--device {auto|cuda|cpu}
--max-tokens N         # Generation length
```

### Advanced Options
```bash
--model-id MODEL       # Different HF model
--quantize            # Enable 4-bit quantization
--output-dir DIR      # Output directory
```

## 🐛 Troubleshooting Guide

### Problem: CUDA out of memory
**Solution:** Use `--quantize` or `--device cpu`

### Problem: Image not found
**Solution:** Provide valid image with `--image path/to/image.jpg`

### Problem: ONNX conversion fails
**Solution:** Ensure onnx and onnxruntime are installed

### Problem: Module not found
**Solution:** Run `pip install -r requirements.txt`

### Problem: Standalone scripts don't work
**Solution:** Run `--mode save_standalone` first

## 📞 Getting Help

1. Check this INDEX for navigation
2. Read relevant documentation file
3. Run example scripts
4. Check error messages carefully
5. Verify dependencies installed

## 🎯 Next Steps

### Ready to Start?
1. Install dependencies: `pip install -r requirements.txt`
2. Follow [QUICKSTART.md](QUICKSTART.md)
3. Run your first example

### Ready to Deploy?
1. Read [README.md](README.md) deployment section
2. Convert models with `--mode convert_onnx`
3. Test with `--mode run_onnx`
4. Use standalone scripts for production

### Ready to Extend?
1. Read [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
2. Understand pipeline architecture
3. Fork and modify
4. Integrate your optimizations

## 🌟 Key Features Highlight

✅ 5 complete run modes  
✅ Modular pipeline architecture  
✅ ONNX conversion with validation  
✅ SafeTensors support  
✅ Performance tracking  
✅ Standalone deployment  
✅ VitisAI infrastructure ready  
✅ AWQ infrastructure ready  
✅ Comprehensive documentation  
✅ Example scripts included  

## 📝 Version Info

- **Current Version:** 2.0
- **Python:** 3.8+
- **PyTorch:** 2.0+
- **Transformers:** 4.35+

---

**Happy Model Partitioning! 🚀**

For questions or issues, refer to the specific documentation files linked above.

