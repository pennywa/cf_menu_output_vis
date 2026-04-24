# AI Waiter Chatbot (RAG Demo)

This project shows two outputs for every user question:

1. **Without RAG**: a generic chatbot-style response that is not grounded in the menu data.
2. **With RAG**: a retrieval-grounded response using `cheesecake_factory_menu.txt`.

## Local CLI Run

```bash
python ai_waiter_chatbot.py
```

Then ask questions such as:

- `What pasta items do you have?`
- `Do you have cheesecake?`
- `What are the cheapest appetizers?`

Type `exit` to quit.

## Deploy to Hugging Face Spaces

This repo now includes:

- `app.py` (Gradio Space entrypoint)
- `requirements.txt` (Space dependencies)
- `embedding_vis.py` (interactive embedding node visualization)
- `cheesecake_factory_menu.txt` (RAG data source)

### Steps

1. Create a new Space on Hugging Face:
   - Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
   - Choose **Gradio** as the SDK
2. Upload these files to the Space:
   - `app.py`
   - `ai_waiter_chatbot.py`
   - `cheesecake_factory_menu.txt`
   - `requirements.txt`
3. Hugging Face will automatically build and launch the app.
4. Open your Space URL and ask menu questions.

The UI will show:

- `Response 1: Without RAG`
- `Response 2: With RAG`
- `Embedding Nodes (Interactive)` with highlighted retrieved menu nodes and query position
