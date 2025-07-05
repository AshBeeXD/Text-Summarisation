from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_tokenizer(model_name="facebook/bart-base"):
    return AutoTokenizer.from_pretrained(model_name)

def get_model(model_name="facebook/bart-base"):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)

