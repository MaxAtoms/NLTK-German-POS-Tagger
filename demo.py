import os
import tkinter as tk
from tkinter import ttk
import nltk
from nltk import word_tokenize
from nltk.data import path
from german_tagger import PerceptronTagger

def ui():
    def on_tag_button_click():
        text = text_field.get().strip()
        tokens = word_tokenize(text)
        toggle_tagdict = use_tagdict_var.get()
        tagged_text = tagger.tag(tokens, use_tagdict=toggle_tagdict)

        for widget in result_frame.winfo_children():
            widget.destroy()

        for i, (token, tag) in enumerate(tagged_text):
            token_label = tk.Label(result_frame, text=token, font=("Arial", 12, "bold"))
            tag_label = tk.Label(result_frame, text=tag, font=("Arial", 10, "italic"))
            token_label.grid(row=0, column=i, padx=5)
            tag_label.grid(row=1, column=i, padx=5)

    root = tk.Tk()
    root.title("German POS Tagger")

    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)

    text_field = tk.Entry(input_frame, width=50)
    text_field.grid(row=0, column=0, padx=(0, 5))

    tag_button = ttk.Button(input_frame, text="Tag", command=on_tag_button_click)
    tag_button.grid(row=0, column=1, padx=(0, 5))

    use_tagdict_var = tk.BooleanVar(value=True)
    use_tagdict = ttk.Checkbutton(input_frame, text="Use tag dictionary", variable=use_tagdict_var)
    use_tagdict.grid(row=0, column=2)

    result_frame = tk.Frame(root)
    result_frame.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    nltk.download('punkt')
    path.append(os.getcwd())

    tagger = PerceptronTagger(1, "tiger_train90_val10", load=True, lang='deu')
    ui()
