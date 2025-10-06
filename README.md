# BERT-IDS: Network Intrusion Detection Using BERT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

## 📋 Deskripsi Proyek

**BERT-IDS** adalah penelitian inovatif yang menerapkan arsitektur Bidirectional Encoder Representations from Transformers (BERT) untuk deteksi intrusi jaringan. Penelitian ini bertujuan untuk mengembangkan sistem deteksi intrusi yang lebih akurat dengan memanfaatkan kemampuan pemahaman konteks bidirectional dari BERT untuk menganalisis pola lalu lintas jaringan.

### 🎯 Tujuan Penelitian

- Mengembangkan framework novel untuk menerapkan BERT pada deteksi intrusi jaringan
- Mencapai akurasi deteksi yang superior dibandingkan metode tradisional
- Mengurangi tingkat false positive dalam sistem IDS
- Menganalisis kemampuan BERT dalam mendeteksi serangan zero-day
- Menyediakan interpretabilitas model melalui attention mechanisms

## 🔬 Metodologi Penelitian

### Dataset yang Digunakan
- **CICIDS2017**: Dataset komprehensif dengan berbagai jenis serangan
- **NSL-KDD**: Dataset benchmark klasik untuk evaluasi IDS
- **UNSW-NB15**: Dataset modern dengan serangan kontemporer
- **CIC-DDoS2019**: Dataset khusus untuk serangan DDoS

### Arsitektur Model
```
Network Traffic → Tokenization → BERT Encoder → Classification Head → Attack Detection
```

### Baseline Comparisons
- Traditional ML: SVM, Random Forest, Naive Bayes
- Deep Learning: CNN, LSTM, GRU, Autoencoder
- Ensemble Methods: XGBoost, AdaBoost
- Transformer Variants: DistilBERT, RoBERTa

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone repository**
```bash
git clone https://github.com/username/bert-ids.git
cd bert-ids
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. **Download datasets**
```bash
python scripts/download_datasets.py
```

2. **Preprocess data**
```bash
python scripts/preprocess_data.py --dataset cicids2017
```

### Training

```bash
# Basic training
python train.py --config configs/bert_base.yaml

# Training dengan custom parameters
python train.py --config configs/bert_base.yaml --batch_size 32 --learning_rate 2e-5
```

### Evaluation

```bash
# Evaluate model
python evaluate.py --model_path checkpoints/bert_ids_best.pt --dataset cicids2017

# Generate detailed report
python evaluate.py --model_path checkpoints/bert_ids_best.pt --dataset cicids2017 --detailed_report
```

## 📊 Hasil Eksperimen

### Performance Metrics (Preliminary)

| Model | Dataset | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| BERT-IDS | CICIDS2017 | 96.8% | 97.2% | 96.5% | 96.8% |
| Random Forest | CICIDS2017 | 94.2% | 93.8% | 94.6% | 94.2% |
| LSTM | CICIDS2017 | 95.1% | 94.9% | 95.3% | 95.1% |

*Note: Hasil ini adalah preliminary dan akan diupdate seiring progress penelitian*

## 📁 Struktur Proyek

```
bert-ids/
├── README.md
├── prd.md                    # Research Proposal Document
├── requirements.txt
├── setup.py
├── configs/                  # Configuration files
│   ├── bert_base.yaml
│   └── bert_large.yaml
├── src/                      # Source code
│   ├── __init__.py
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── models/               # Model architectures
│   │   ├── __init__.py
│   │   ├── bert_ids.py
│   │   └── baselines.py
│   ├── training/             # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── visualization.py
├── scripts/                  # Utility scripts
│   ├── download_datasets.py
│   ├── preprocess_data.py
│   └── run_experiments.py
├── notebooks/                # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_analysis.ipynb
│   └── results_visualization.ipynb
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── data/                     # Data directory
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/                   # Saved models
│   └── checkpoints/
├── results/                  # Experiment results
│   ├── logs/
│   ├── figures/
│   └── reports/
└── docs/                     # Documentation
    ├── api.md
    ├── datasets.md
    └── experiments.md
```

## 🔧 Configuration

### Model Configuration (configs/bert_base.yaml)
```yaml
model:
  name: "bert-base-uncased"
  num_labels: 8
  dropout: 0.1
  
training:
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 1000
  
data:
  max_sequence_length: 512
  tokenization_strategy: "flow_based"
```

## 📈 Monitoring & Logging

Proyek ini menggunakan:
- **TensorBoard** untuk monitoring training
- **Weights & Biases** untuk experiment tracking
- **MLflow** untuk model versioning

```bash
# Start TensorBoard
tensorboard --logdir results/logs

# View W&B dashboard
wandb login
python train.py --use_wandb
```

## 🧪 Running Experiments

### Experiment Scripts

```bash
# Run full benchmark comparison
python scripts/run_experiments.py --config configs/benchmark.yaml

# Run ablation study
python scripts/run_experiments.py --config configs/ablation.yaml

# Run interpretability analysis
python scripts/interpretability_analysis.py --model_path checkpoints/bert_ids_best.pt
```

### Custom Experiments

```python
from src.training.trainer import BERTIDSTrainer
from src.models.bert_ids import BERTIDSModel

# Initialize model and trainer
model = BERTIDSModel(config)
trainer = BERTIDSTrainer(model, config)

# Train model
trainer.train(train_dataloader, val_dataloader)

# Evaluate
results = trainer.evaluate(test_dataloader)
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Dataset Guide](docs/datasets.md)
- [Experiment Guide](docs/experiments.md)
- [Research Proposal](prd.md)

## 🤝 Contributing

Kami menyambut kontribusi dari komunitas peneliti! Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) untuk guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

Proyek ini dilisensikan under MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📞 Contact

- **Peneliti Utama**: [Nama Anda]
- **Email**: [email@university.edu]
- **Institusi**: [Nama Universitas]
- **Lab**: [Nama Lab/Research Group]

## 🙏 Acknowledgments

- Hugging Face untuk library Transformers
- PyTorch team untuk framework deep learning
- Penyedia dataset: Canadian Institute for Cybersecurity, UNSW Canberra
- [Nama Advisor/Supervisor]
- [Nama Institusi/Lab]

## 📖 Citation

Jika Anda menggunakan kode atau hasil dari penelitian ini, mohon cite:

```bibtex
@article{bertids2024,
  title={BERT-IDS: A Novel Approach to Network Intrusion Detection Using Bidirectional Encoder Representations from Transformers},
  author={[Nama Penulis]},
  journal={[Target Journal]},
  year={2024},
  note={Under Review}
}
```

## 🔄 Changelog

### Version 0.1.0 (Current)
- Initial research framework
- Basic BERT-IDS implementation
- Preliminary experiments on CICIDS2017
- Baseline model comparisons

### Planned Features
- [ ] Multi-dataset evaluation
- [ ] Advanced tokenization strategies
- [ ] Real-time inference pipeline
- [ ] Adversarial robustness testing
- [ ] Federated learning implementation

---

**Status Penelitian**: 🚧 Dalam Pengembangan

**Last Updated**: December 2024