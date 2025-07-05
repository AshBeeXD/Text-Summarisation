from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

def get_training_args(output_dir="../outputs/model"):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="../outputs/logs",
        logging_strategy="epoch",
        predict_with_generate=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        report_to="none",
        fp16=False
    )

def train_model(model, tokenizer, args, train_dataset, val_dataset, compute_metrics=None):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    trainer.train()
    return trainer

