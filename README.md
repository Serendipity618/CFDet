# CFDet: Achieving Counterfactual Explanation for Sequence Anomaly Detection

CFDet (**Counterfactual Detection**) is a framework for **counterfactual explanation in sequence anomaly detection**. It explains the predictions of **Deep SVDD-based anomaly detection models** by highlighting key **anomalous entries** responsible for an anomaly.

This repository provides:

- **Deep SVDD for sequence anomaly detection** 📈
- **CFDet for counterfactual explanations** 🔍
- **Preprocessing, training, and evaluation scripts** ⚙️

---

## 📖 Introduction

Anomaly detection on discrete sequential data is crucial for detecting novel attacks or abnormal system behaviors, particularly from log messages. While existing methods perform well at detecting anomalies, providing **explanations** for these detections is still a challenge due to the **discrete nature of sequential data**.

This project introduces **CFDet (Counterfactual Detection)**, a framework that provides **counterfactual explanations** for detected anomalies by highlighting the **anomalous entries** that contribute to the anomaly classification.

---

## 📄 Reference Paper

This project is based on the following research:

**Title:** [Achieving Counterfactual Explanation for Sequence Anomaly Detection (Springer)](https://link.springer.com/chapter/10.1007/978-3-031-70371-3_2)  
**ArXiv Version:** [arXiv:2210.04145](https://arxiv.org/abs/2210.04145)  
**Authors:** He Cheng, Depeng Xu, Shuhan Yuan, Xintao Wu  
**Publication Date:** August 22, 2024  
**Conference:** Joint European Conference on Machine Learning and Knowledge Discovery in Databases  
**Pages:** 19-35  
**Publisher:** Springer Nature Switzerland  

### Abstract

> Anomaly detection on discrete sequential data has been investigated for a long time because of its potential in various applications, such as detecting novel attacks or abnormal system behaviors from log messages. Although many approaches can achieve good performance on anomalous sequence detection, how to explain the detection results is still challenging due to the discrete nature of sequential data. Specifically, given a sequence that is detected as anomalous, the explanation is to highlight those anomalous entries in the sequence leading to the anomalous outcome. To this end, we propose a novel framework, called CFDet, that can explain the detection results of one-class sequence anomaly detection models by highlighting the anomalous entries in the sequences based on the idea of counterfactual explanation. Experimental results on three datasets show that CFDet can provide explanations by correctly detecting anomalous entries.

---

## 📖 Citation

If you use CFDet in your research, please cite the following paper:

```bibtex
@inproceedings{cheng2024cfdet,
  author    = {He Cheng and Depeng Xu and Shuhan Yuan and Xintao Wu},
  title     = {Achieving Counterfactual Explanation for Sequence Anomaly Detection},
  booktitle = {Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  pages     = {19--35},
  doi       = {10.1007/978-3-031-70371-3_2}
}
```

---

## ⚙️ Installation

Ensure you have **Python 3.8+** installed. Install dependencies using:

```bash
pip install -r requirements.txt
```

### Required Libraries

- `torch`
- `numpy`
- `pandas`
- `scikit-learn`

---

## 🚀 Usage

To train and evaluate the CFDet model, run:

```bash
python main.py --dataset_path ../data/BGL.log_structured_v1.csv
```

### Workflow

1. **Data Processing**: Loads and preprocesses log data.
2. **Anomaly Detection (Deep SVDD)**:
   - Initializes Deep SVDD model.
   - Trains and evaluates on log sequences.
3. **Counterfactual Explanation (CFDet)**:
   - Finds baseline sequences for explanations.
   - Trains and evaluates CFDet to generate counterfactuals.

---

## 📂 Project Structure

```
├── data/                       # Folder for dataset files (e.g., BGL.log_structured_v1.csv)
├── output/                     # Output directory for trained models and evaluation results
│   ├── model/                  # Trained model states
│   ├── ad_result/              # Results from anomaly detection evaluation
│   ├── explain_result/         # Results from counterfactual explanation evaluation
├── CFDet/                      # Jupyter Notebook implementations
├── src/                        # Source code directory
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # Main script to run training & evaluation
│   ├── preprocessing.py        # Data processing utilities
│   ├── encoding.py             # Encoding log events
│   ├── ad_model.py             # Deep SVDD anomaly detection model
│   ├── ad_trainer.py           # Training for anomaly detection
│   ├── cfdet.py                # Counterfactual explanation model (CFDet)
│   ├── cfdet_trainer.py        # Training for CFDet explanation model
│   ├── utils.py                # Helper functions for training & evaluation
├── requirements.txt            # Required dependencies
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

For any inquiries, please contact **He Cheng** or refer to the corresponding paper.

---

## 🎓 Acknowledgments

This project is based on **"Achieving Counterfactual Explanation for Sequence Anomaly Detection"**, presented at the **Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2024**.
