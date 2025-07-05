from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def load_and_split_dataset(model_name='facebook/bart-base', max_input_len=512, max_target_len=128):
    # Load CNN/DailyMail dataset
    train = load_dataset("cnn_dailymail", "3.0.0", split="train")
    val = load_dataset("cnn_dailymail", "3.0.0", split="validation")
    test = load_dataset("cnn_dailymail", "3.0.0", split="test")
    
    # Combine and shuffle
    full_dataset = concatenate_datasets([train, val, test]).shuffle(seed=42)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)

    train_dataset = full_dataset.select(range(train_size))
    val_dataset = full_dataset.select(range(train_size, train_size + val_size))
    test_dataset = full_dataset.select(range(train_size + val_size, total_size))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["article"],
            max_length=max_input_len,
            padding="max_length",
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["highlights"],
                max_length=max_target_len,
                padding="max_length",
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

    return train_dataset, val_dataset, test_dataset

