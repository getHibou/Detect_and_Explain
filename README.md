# 🧠 Fine-tuning Generative LLMs to Detect and Explain Suicidal Ideation in Brazilian Portuguese Texts

This repository provides a complete pipeline for fine-tuning **LLMs (Large Language Models)** to classify suicidal ideation using **5-fold cross-validation**, **LoRA optimization**, and **ROC curve analysis**. The models are trained with **Unsloth** for efficient memory usage and fine-tuning.

## 📌 Overview

This pipeline:

✔️ Uses **Low-Rank Adaptation (LoRA) + RSLora** for memory-efficient fine-tuning.  
✔️ Trains and evaluates **multiple LLMs**, including **LLaMA 3**, **Gemma**, and **Phi-3**.  
✔️ Implements **5-fold Stratified Cross-Validation** for better model generalization.  
✔️ **Optimizes the classification threshold** using the **ROC-AUC curve**.  
✔️ **Generates confusion matrices and performance reports** automatically.  

## 🚀 Technologies Used

The project relies on the following frameworks:

- **Transformers** 🤗 - NLP models and tokenizers  
- **TRL** 🚀 - Fine-tuning with Reinforcement Learning  
- **Unsloth** ⚡ - Optimized fine-tuning framework for LLaMA  
- **PEFT** 🛠️ - Efficient parameter fine-tuning (LoRA, QLoRA)  
- **scikit-learn** 📊 - Metrics, stratified cross-validation  
- **Matplotlib & Seaborn** 📉 - Model visualization (ROC curve, confusion matrix)  
- **PyTorch** 🔥 - Deep learning framework  
- **Datasets** 📚 - Hugging Face dataset management  
- **Pandas & NumPy** 🏗️ - Data manipulation  

---

## 📂 Project Structure

### **1️⃣ Model Setup & Fine-Tuning**
- **Loads & configures LLMs** (LLaMA 3, Phi-3, Gemma, Qwen)
- **Modifies `lm_head` weights** for classification tuning
- **Uses LoRA with RSLora** to efficiently fine-tune models

### **2️⃣ Data Processing**
- **Loads labeled dataset** from CSV  
- **Splits dataset into training & validation** (Stratified Sampling)  
- **Visualizes data distribution** using Matplotlib  

### **3️⃣ Training with Stratified Cross-Validation**
- Uses **5-fold Stratified K-Fold**  
- Fine-tunes the model separately for each fold  

### **4️⃣ ROC Curve & Optimal Threshold Calculation**
- **Generates ROC curve & AUC score**  
- **Finds the optimal classification threshold**  
- **Saves metrics & confusion matrix**  

### **5️⃣ Model Evaluation & Visualization**
- Computes **Accuracy, Precision, Recall, F1-score**  
- Saves & plots **Confusion Matrix**  
- Plots **ROC Curve** to assess model performance  

---

## 🛠 Installation & Setup

### **1️⃣ Install Required Dependencies**
Make sure you have Python installed, then run:

```bash
pip install torch transformers trl unsloth datasets peft pandas numpy scikit-learn matplotlib seaborn tqdm
