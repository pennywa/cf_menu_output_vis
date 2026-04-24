# AI Waiter Chatbot (RAG Demo) 🍰

The UI will show:

- `Response 1: Without RAG`
- `Response 2: With RAG`
- `Embedding Nodes (Interactive)` with highlighted retrieved menu nodes and query position

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
