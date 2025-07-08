# INFO7375_FineTuningLLM

Historical Text Modernizer
A fine-tuned language model that transforms archaic and historical English text into modern, accessible language while preserving meaning and cultural context.

🎯 Overview
This project fine-tunes GPT-2 Medium using LoRA (Low-Rank Adaptation) on a comprehensive dataset of historical text pairs, enabling automatic modernization of Shakespeare, legal documents, religious texts, and historical speeches.

📊 Key Results
Training Success: 61% loss reduction (10.0 → 3.9) over 3 epochs
Baseline Improvement: 45.2% performance gain over pre-trained models
Evaluation Metrics: 89.74% similarity, 91.96% modernization success
Dataset: 304 examples across 9 historical text types

🔧 Technical Implementation
Model Architecture
Base Model: GPT-2 Medium (355M parameters)
Fine-tuning: LoRA with 4.3M trainable parameters (1.20% of total)
Training: 3 epochs, 5e-5 learning rate, Tesla T4 GPU
Optimization: Comprehensive hyperparameter search (3 configurations)

Dataset Composition
Source TypeExamplesDomainShakespeare155Literary/PoeticLegal Documents31Formal/LegalReligious Texts19Biblical/ArchaicHistorical Speeches10Political/OratoryVariations94Augmented Examples
🚀 Quick Start
Installation
bashpip install torch transformers peft accelerate datasets
Basic Usage
pythonfrom modernizer import HistoricalTextModernizer

# Initialize modernizer
modernizer = HistoricalTextModernizer()

# Modernize text
result = modernizer.modernize("Thou art a noble friend")
print(result)  # Output: "You are a noble friend"
Interactive Mode
python# Run interactive modernization
modernizer.interactive_mode()


historical-text-modernizer/
├── 📁 data/
│   ├── 📄 README.md                          # Dataset documentation
│   ├── 📁 raw/
│   │   ├── shakespeare_sources.txt           # Original Shakespeare sources
│   │   ├── legal_sources.txt                 # Legal document sources
│   │   ├── historical_sources.txt            # Historical speech sources
│   │   └── biblical_sources.txt              # Religious text sources
│   ├── 📁 processed/
│   │   ├── train_data_expanded.json          # Training data (212 examples)
│   │   ├── val_data_expanded.json            # Validation data (45 examples)
│   │   └── test_data_expanded.json           # Test data (47 examples)
│   └── validate_dataset.py                   # Dataset validation script
├── 📁 src/
│   ├── __init__.py
│   ├── dataset_creation.py                   # Step 4: Dataset creation
│   ├── model_training.py                     # Step 5: LoRA fine-tuning
│   ├── hyperparameter_optimization.py       # Step 6: Hyperparameter experiments
│   ├── baseline_comparison.py               # Step 6b: Baseline evaluation
│   ├── inference_pipeline.py                # Step 7: Inference system
│   ├── custom_metrics.py                    # Step 8: Evaluation metrics
│   ├── enhanced_training.py                 # Step 9: Training with metrics
│   └── modernizer.py                        # Main modernizer class
├── 📁 models/
│   ├── .gitkeep                             # Keep empty directory
│   ├── historical-modernizer-final/         # Final trained model
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── checkpoints/                         # Training checkpoints
├── 📁 notebooks/
│   ├── complete_pipeline.ipynb              # Full assignment notebook
│   ├── data_exploration.ipynb               # Dataset analysis
│   ├── model_evaluation.ipynb               # Results analysis
│   └── hyperparameter_analysis.ipynb       # Optimization analysis
├── 📁 scripts/
│   ├── setup_environment.py                 # Environment setup
│   ├── download_models.py                   # Model download utility
│   ├── run_training.py                      # Training script
│   └── evaluate_model.py                    # Evaluation script
├── 📁 tests/
│   ├── __init__.py
│   ├── test_dataset.py                      # Dataset tests
│   ├── test_modernizer.py                   # Modernizer tests
│   └── test_metrics.py                      # Metrics tests
├── 📁 docs/
│   ├── assignment_documentation.md          # Complete assignment write-up
│   ├── methodology.md                       # Technical methodology
│   ├── results_analysis.md                  # Results documentation
│   └── video_script.md                      # Video presentation script
├── 📁 results/
│   ├── training_logs/                       # Training output logs
│   ├── evaluation_results/                  # Evaluation outputs
│   ├── visualizations/                      # Charts and graphs
│   └── comparison_tables/                   # Performance comparisons
├── 📁 examples/
│   ├── basic_usage.py                       # Simple usage examples
│   ├── batch_processing.py                  # Batch processing demo
│   └── interactive_demo.py                  # Interactive demonstration
├── 📄 README.md                             # Main project documentation
├── 📄 requirements.txt                      # Python dependencies
├── 📄 requirements-dev.txt                  # Development dependencies
├── 📄 setup.py                              # Package installation
├── 📄 pyproject.toml                        # Modern Python configuration
├── 📄 .gitignore                            # Git ignore rules
├── 📄 .gitattributes                        # Git attributes (for LFS if needed)
├── 📄 LICENSE                               # Project license
├── 📄 CONTRIBUTING.md                       # Contribution guidelines
├── 📄 CHANGELOG.md                          # Version history
└── 📄 video_script.md                       # 10-minute video script

📈 Performance Analysis
Hyperparameter Optimization
ConfigurationLearning RateLoRA RankVal LossPerformanceConservative1e-0589.576BaselineBalanced5e-05167.987ModerateAggressive1e-04320.633Optimal
Baseline Comparison

Pre-trained GPT-2: 40.0% accuracy (generates irrelevant content)
Fine-tuned Model: 85.2% accuracy (successful modernization)
Improvement: +45.2% performance gain

🎯 Example Transformations
Historical TextModern Text"Thou art a villain and thy words are false""You are a villain and your words are false""Wherefore dost thou weep?""Why do you weep?""Four score and seven years ago""Eighty-seven years ago""We hold these truths to be self-evident""We believe these facts are obvious"
🔬 Evaluation Metrics
Custom Metrics Suite

BLEU Score: 79.33% (translation quality)
Semantic Similarity: 86.27% (meaning preservation)
Modernization Success: 91.96% (transformation accuracy)
Valid Output Rate: 100% (generation reliability)

Quality Control
Intelligent fallback to rule-based modernization
Multi-criteria output validation
Systematic error detection and correction

🛠️ Advanced Features
Inference Pipeline

Single Text Processing: Direct modernization
Batch Processing: Efficient multiple text handling
Interactive Mode: Real-time testing interface
Quality Assurance: Automatic output validation

Error Analysis & Improvements
Generation Control: Optimized parameters for focused output
Fallback Mechanisms: Rule-based reliability guarantee
Quality Assessment: Multi-dimensional evaluation framework

📚 Applications
Digital Humanities: Making historical texts accessible
Education: Interactive historical document exploration
Research: Automated analysis of archaic language patterns
Cultural Preservation: Bridging historical and contemporary language

🎓 Academic Contribution
Novel Aspects
Domain-Specific Dataset: First comprehensive historical text modernization corpus
Hybrid Approach: Model + rule-based reliability system
Multi-Domain Coverage: Literature, legal, religious, and political texts
Systematic Methodology: Reproducible fine-tuning pipeline

Research Impact
Specialized NLP: Historical text processing advancement
Evaluation Framework: Custom metrics for transformation quality
Production Pipeline: End-to-end modernization system
Educational Value: Comprehensive ML methodology demonstration

🔄 Future Enhancements
Web Interface: Browser-based modernization tool
API Development: RESTful service for integration
Multi-Language Support: Extension to other historical languages
Performance Optimization: Model compression and acceleration

📄 Citation
@software{historical_text_modernizer,
  title={Historical Text Modernizer: Fine-tuned Language Model for Archaic Text Transformation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/historical-text-modernizer}
}
