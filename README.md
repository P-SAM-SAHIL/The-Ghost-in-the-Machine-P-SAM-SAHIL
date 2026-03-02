# The-Ghost-in-the-Machine-P-SAM-SAHIL


---


This project shows a simple idea:

Instead of using embeddings, we detect AI text using **interpretable sparse neuron features** from a large language model.

We extract activations from Gemma-2-2B, convert them into SAE features, and train an XGBoost classifier on top.



---

## What this does (in simple terms)

Pipeline:

```
Text → Gemma Residual Stream → SAE Sparse Features → XGBoost → Prediction
```

Each feature is actually a **real neuron direction** you can inspect on Neuronpedia.

So this is not a black box.

---

## Results

* 500 samples → 98% accuracy
* 2000 samples → **99% accuracy**



---

## How to run

### 1. Install deps

```
pip install sae-lens transformer-lens
pip install "numpy<2.0.0"
pip install xgboost kagglehub shap
```

---

### 2. Add your tokens

Put your HuggingFace token inside the script:

```
HF_TOKEN = "your_token"
```

Make sure Kaggle API is configured.

---

### 3. Run

Just run the script or notebook.

It will:

• Load dataset
• Extract SAE features
• Train classifier
• Print top predictive neurons

---

## Why this is cool

Most AI detectors use embeddings.

This uses **actual interpretable neurons** inside the model.

We can literally trace which features detect AI writing.

---

## Credits

* Gemma-2-2B
* SAE-Lens
* TransformerLens
* DAIGT V2 dataset

