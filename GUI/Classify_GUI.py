import tkinter as tk
from tkinter import scrolledtext, font
from tkinter import messagebox

import torch
from transformers import BertTokenizer, BertModel
from Model.Trans_Classifier import TransformerClassifier, load_model

model = load_model("/Users/cuilili/Documents/Exeter_DS/Dissertation/Presentation/best.pth")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").eval()
device = torch.device("cpu")

def classify(text):
    # Tokenize and encode the input text
    encoding = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=512,
                                     return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get BERT embeddings
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.squeeze(0)

    # Make prediction using the loaded model
    with torch.no_grad():
        prediction = model(embeddings.unsqueeze(0)).squeeze().item()

    # Return classification result
    return "Spam" if prediction >= 0.5 else "Ham"


def on_button_click():
    input_text = text_area.get("1.0", tk.END).strip()
    if input_text:
        result = classify(input_text)
        result_label.config(text=f"Classification Result: {result}")
    else:
        result_label.config(text="Please enter some text.")

# Create the main window
root = tk.Tk()
root.title("Email Classification")

# Set the window size
root.geometry("600x400")  # Width x Height

# Create a scrolled text widget for multi-line input
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20)
text_area.pack(padx=15, pady=10)

# Create a button to submit the text
submit_button = tk.Button(root, text="Submit", command=on_button_click, width=15, height=2)
submit_button.pack(pady=10)

# Create a label to display the result
result_font = font.Font(size=14, weight="bold")
result_label = tk.Label(root, text="Classification Result:", wraplength=500, font=result_font)
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
