
import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
from infer import chat_with_bot  # Make sure infer.py is in the same directory

# Function to handle user input and bot response
def handle_chat():
    question = user_input.get()
    if question.strip() == "":
        return
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, f"You: {question}\n", "user")
    chat_log.insert(tk.END, "Bot: Thinking...\n", "bot")
    chat_log.config(state=tk.DISABLED)
    chat_log.see(tk.END)
    user_input.delete(0, tk.END)
    
    def generate_and_display():
        answer = chat_with_bot(question)
        chat_log.config(state=tk.NORMAL)
        chat_log.delete("end-2l", "end-1l")  # Remove "Bot: Thinking..."
        chat_log.insert(tk.END, f"Bot: {answer}\n\n", "bot")
        chat_log.config(state=tk.DISABLED)
        chat_log.see(tk.END)

    Thread(target=generate_and_display).start()

# Setup window
root = tk.Tk()
root.title("FreshieBot")
root.geometry("600x600")
root.configure(bg="#1e1e1e")

# Chat log
chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="#2d2d2d", fg="#ffffff", font=("Segoe UI", 11))
chat_log.config(state=tk.DISABLED)
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Custom tags for color
chat_log.tag_config("user", foreground="#00ffff")
chat_log.tag_config("bot", foreground="#90ee90")

# User input
input_frame = tk.Frame(root, bg="#1e1e1e")
input_frame.pack(pady=10, fill=tk.X)

user_input = tk.Entry(input_frame, font=("Segoe UI", 12), bg="#2d2d2d", fg="#ffffff", insertbackground="#ffffff")
user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5), ipady=6)
user_input.bind("<Return>", lambda event: handle_chat())

send_button = tk.Button(input_frame, text="Send", command=handle_chat, bg="#007acc", fg="white", font=("Segoe UI", 11, "bold"))
send_button.pack(side=tk.RIGHT, padx=(5, 10))

# Run the app
root.mainloop()
