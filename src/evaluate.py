from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import evaluate
from transformers import EvalPrediction

rouge = evaluate.load("rouge")

def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred: EvalPrediction):
        predictions, labels = eval_pred

        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        return {k: round(v * 100, 2) for k, v in result.items()}

    return compute_metrics


def load_model_and_tokenizer(model_path="../outputs/model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model, tokenizer


def generate_summaries(model, tokenizer, dataset, max_input_len=512, max_target_len=128):
    model.eval()
    summaries = []
    for item in tqdm(dataset):
        input_ids = tokenizer(item["input_text"], return_tensors="pt", truncation=True, max_length=max_input_len).input_ids
        output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_target_len)[0]
        summary = tokenizer.decode(output_ids, skip_special_tokens=True)
        summaries.append(summary)
    return summaries


def compute_rouge(predictions, references):
    return rouge.compute(predictions=predictions, references=references, use_stemmer=True)


def evaluate(model_path="../outputs/model", test_data_path="../data/test"):
    model, tokenizer = load_model_and_tokenizer(model_path)
    dataset = load_from_disk(test_data_path)

    input_texts = tokenizer.batch_decode(dataset["input_ids"], skip_special_tokens=True)
    reference_summaries = tokenizer.batch_decode(dataset["labels"], skip_special_tokens=True)

    generated_summaries = generate_summaries(model, tokenizer, [{"input_text": t} for t in input_texts])

    scores = compute_rouge(generated_summaries, reference_summaries)

    print("\nROUGE Scores:")
    for k, v in scores.items():
        print(f"{k}: {v:.2f}")

    return scores