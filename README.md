# EMR-AGENT Repository

A comprehensive repository containing AI-powered tools for Electronic Medical Record (EMR) analysis, including automated cohort generation and clinical feature extraction.

## 🏗️ Repository Structure

This repository contains two main projects:

```
EMR-AGENT/
├── EMR-Agent/          # Main EMR Agent system
└── PreCISE-EMR/        # PreCISE-EMR preprocessing toolkit  
```

## 📁 Projects Overview

### 🤖 EMR-Agent
An advanced AI agent system for automated patient cohort generation and clinical feature mapping from EMR databases.

**Key Features:**
- **CMA (Cohort generation with Main Agent)**: Automated patient cohort extraction
- **CFSA (Clinical Feature Selection and mapping Agent)**: Clinical feature mapping and selection
- Support for multiple EMR databases (eICU, MIMIC-III, SICdb)
- Ablation study modules for comprehensive evaluation

**📖 For detailed information, please refer to: [`EMR-Agent/README.md`](EMR-Agent/README.md)**

### 🔧 PreCISE-EMR  
A preprocessing and dataset construction toolkit for EMR data analysis.

**Key Features:**
- EMR data preprocessing pipelines
- Dataset construction utilities
- Data quality assessment tools
- Feature engineering capabilities

**📖 For detailed information, please refer to: [`PreCISE-EMR/README.md`](PreCISE-EMR/README.md)**

## 🚀 Quick Start

### Prerequisites
```bash
# Clone the repository
git clone <repository-url>
cd EMR-AGENT

# Install common dependencies
pip install -r requirements.txt
```

### Getting Started

1. **For EMR-Agent**: Navigate to the `EMR-Agent/` directory and follow the instructions in [`EMR-Agent/README.md`](EMR-Agent/README.md)

2. **For PreCISE-EMR**: Navigate to the `PreCISE-EMR/` directory and follow the instructions in [`PreCISE-EMR/README.md`](PreCISE-EMR/README.md)

## 📚 Documentation

Each project contains its own comprehensive documentation:

- **[EMR-Agent Documentation](EMR-Agent/README.md)** - Complete guide for the EMR Agent system
- **[PreCISE-EMR Documentation](PreCISE-EMR/README.md)** - Complete guide for the PreCISE-EMR toolkit

## 🎯 Use Cases

- **Healthcare Researchers**: Automated cohort identification and feature extraction
- **Data Scientists**: EMR data preprocessing and analysis
- **Clinical Researchers**: Patient population studies and clinical feature analysis
- **Healthcare Analytics**: Large-scale EMR data processing

## 📄 Citation

If you use this repository in your research, please cite our work:

```bibtex
@article{emr-agent,
  title={EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases},
  author={[Authors]},
  journal={[Journal]},
  year={[Year]}
}
```

## 🤝 Contributing

We welcome contributions! Please see the individual project READMEs for specific contribution guidelines.

## 📞 Support

For questions and support:
- Check the relevant project's README for specific issues
- Open an issue in the GitHub repository
- Contact the development team

## 📝 License

This project is licensed under [License] - see the LICENSE file for details.

---

**🔗 Quick Links:**
- [EMR-Agent Setup Guide](EMR-Agent/README.md#quick-start)
- [PreCISE-EMR Setup Guide](PreCISE-EMR/README.md)
- [EMR-Agent Scripts Documentation](EMR-Agent/README.md#script-description)

💡 **Tip**: Start with the project that best fits your use case - EMR-Agent for automated analysis or PreCISE-EMR for data preprocessing.
