# BART-Based Scientific Paper Summarization

This project implements a BART-based abstractive summarization model trained on a subset of the CNN/DailyMail dataset to generate concise summaries of scientific papers. It follows a modular design with clean separation between exploration, training, evaluation, and deployment.

## Project Structure

```
summarisation-bart/
│
├── app/                   # Hugging Face Space application
│   └── app.py
│
├── data/                  # Disk-saved datasets (train/val/test)
│
├── outputs/               # Saved model
│   └── model/
│
├── src/                   # Source scripts
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/             # Jupyter notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_train.ipynb
│   └── 03_evaluation.ipynb
│
├── requirements.txt
└── README.md
```
The outputs folder contains compressed files as the files were too large to upload through GitHub or GitLFS

## Model

The model is built on top of `facebook/bart-base` and fine-tuned for text summarization. BART is a denoising autoencoder for pretraining sequence-to-sequence models and has shown state-of-the-art performance on summarization tasks.

Key configurations:
- Base architecture: `facebook/bart-base`
- Tokenizer: BART tokenizer
- Evaluation strategy: ROUGE and BERTScore
- Device: CPU or GPU (via PyTorch)

## Dataset

A preprocessed subset of the CNN/DailyMail dataset is used. The dataset is split into:
- Training: 7,000 samples
- Validation: 1,500 samples
- Testing: 1,500 samples

All data is saved to disk in Hugging Face's Arrow format and loaded via `datasets.load_from_disk()`.

## Training

The model is trained using Hugging Face’s `Trainer` API with a custom training loop for flexibility. Key training parameters:
- Learning Rate: 5e-5
- Batch Size: 4 (gradient accumulation enabled)
- Epochs: 3–4
- Generation: Beam search (num_beams = 4)

Progress bars are enabled via `tqdm` and metrics are logged for each evaluation step. Models are saved to `outputs/model/`.

## Evaluation

The performance of the summarization model was evaluated using both lexical and semantic metrics:

- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation):  
  ROUGE-1, ROUGE-2, and ROUGE-L metrics were computed to measure lexical overlap between generated summaries and reference summaries. These metrics respectively capture unigram, bigram, and longest common subsequence matches.

- **BERTScore**:  
  BERTScore (F1) was calculated using contextual embeddings from a pre-trained transformer model to assess the semantic similarity between predictions and references. This helps evaluate summary quality beyond surface-level text matches.

- **Length Metrics**:  
  The average number of words in predicted summaries and reference summaries was measured to observe consistency in output length across different training checkpoints.

All metrics were computed on the test set across three training checkpoints: `checkpoint-3500`, `checkpoint-7000`, and `checkpoint-10500`. The results were compiled into a comparison table and visualized using bar plots to highlight performance trends over training progress.

### Evaluation Results

| Checkpoint       | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE Sum | BERTScore (F1) | Pred Len | Ref Len |
|------------------|---------|---------|---------|------------|----------------|----------|---------|
| checkpoint-3500  | 35.03   | 14.42   | 22.72   | 32.62      | 87.21          | 99.01    | 51.71   |
| checkpoint-7000  | 35.58   | 14.89   | 23.15   | 33.11      | 87.31          | 98.12    | 51.71   |
| checkpoint-10500 | 36.14   | 15.33   | 23.71   | 33.72      | 87.30          | 94.11    | 51.71   |

### Visual Insights

The following plots were generated to analyze trends across checkpoints:

- **Grouped Bar Chart for ROUGE and BERTScore**  
  A grouped bar chart compares ROUGE-1, ROUGE-2, ROUGE-L, and BERTScore (F1) scores across all evaluated checkpoints. The visualization highlights progressive improvement in performance as training advances.

- **Summary Length Comparison**  
  A bar chart displays the average predicted summary length versus the reference summary length. This comparison shows that while semantic accuracy improves, predicted summaries remain significantly longer than the reference targets.

These metrics and plots were used to identify the most effective checkpoint and gain insight into model behavior over time.

## Demo

The deployed app is hosted on Hugging Face Spaces and allows users to input text and generate summaries directly through the web interface.

**Try it here**: [https://huggingface.co/spaces/your-username/your-repo-name](https://huggingface.co/spaces/your-username/your-repo-name)

## Credits

This project relies on the following tools, datasets, and open-source libraries:

- **Hugging Face Transformers**:  
  For providing the `facebook/bart-base` model and Trainer API used for training and evaluation.  
  https://github.com/huggingface/transformers

- **CNN/DailyMail Dataset**:  
  Used as the source dataset for training and evaluation. Hosted on Hugging Face Datasets Hub.  
  https://huggingface.co/datasets/cnn_dailymail

- **ROUGE Metric**:  
  For evaluating summarization quality based on lexical overlap (ROUGE-1, ROUGE-2, ROUGE-L).  
  https://aclanthology.org/W04-1013/

- **BERTScore**:  
  Used for evaluating semantic similarity between generated and reference summaries.  
  https://github.com/Tiiiger/bert_score

- **TQDM**:  
  For progress bars during training and evaluation.  
  https://github.com/tqdm/tqdm

- **Matplotlib** and **Seaborn**:  
  For generating visualizations of model performance.  
  https://matplotlib.org/  
  https://seaborn.pydata.org/

All components were used under their respective open-source licenses.
