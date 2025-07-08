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

📁 Project Structure
historical-text-modernizer/
├── src/
│   ├── dataset_creation.py     # Dataset preparation
│   ├── model_training.py       # LoRA fine-tuning
│   ├── evaluation.py           # Custom metrics
│   ├── inference.py            # Modernization pipeline
│   └── improvements.py         # Quality optimizations
├── data/
│   ├── train_data_expanded.json
│   ├── val_data_expanded.json
│   └── test_data_expanded.json
├── models/
│   └── historical-modernizer-final/
├── notebooks/
│   └── complete_pipeline.ipynb
└── README.md

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
