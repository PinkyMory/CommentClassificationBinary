#!/bin/bash
# AutoDL / any Linux GPU instance one-click environment setup
set -e

# Create conda environment if not present
if ! conda env list 2>/dev/null | grep -q "^llm "; then
    echo "Creating conda environment 'llm' (Python 3.12)..."
    conda create -n llm python=3.12 -y
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm

# Install Python dependencies
pip install -r requirements.txt

# Verify CUDA
echo "Checking CUDA..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'CUDA OK: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Environment setup complete."
echo "Activate with: conda activate llm"
echo ""
echo "Next steps:"
echo "  1. Place raw data in data/raw/"
echo "  2. Run: python scripts/01_sampling.py"
echo "  3. Run: python scripts/02_train_traditional.py"
echo "  4. Run: python scripts/03_train_dl.py --wv-path data/embeddings/<your-word-vectors>"
echo "  5. Run: python scripts/04_train_pretrained.py"
echo "  6. Run: python scripts/05_evaluate_all.py"
echo "  7. Run: python app/server.py (desktop web demo) or python app/demo.py (Gradio)"
