import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer from local checkpoint
model_path = "../model/checkpoint-10500"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def summarize(text, max_input_len=512, max_target_len=128, num_beams=4):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len)
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=max_target_len,
        num_beams=num_beams,
        early_stopping=True,
    )[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Gradio UI
demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Input Text", lines=10, placeholder="Paste an article here..."),
        gr.Slider(128, 1024, value=512, label="Max Input Length"),
        gr.Slider(32, 256, value=128, label="Max Summary Length"),
        gr.Slider(1, 8, value=4, step=1, label="Beam Width"),
    ],
    outputs="text",
    title="Neural Text Summarizer (BART)",
    description="A summarization model trained on CNN/DailyMail using BART-base.",
)

demo.launch()

