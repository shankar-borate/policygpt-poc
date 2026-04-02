# Policy GPT

This project exposes the Policy GPT retrieval pipeline through FastAPI and serves a built-in chat UI.

## Run

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Set environment variables:

```powershell
$env:OPENAI_API_KEY="your-key"
```

3. Start the app:

```powershell
python app.py
```

4. Open `http://127.0.0.1:8000`

## Notes

- The document folder is fixed in code to `D:\policy-mgmt\policies`.
- Supported policy files are `.html`, `.htm`, and `.txt`.
- The server starts immediately and indexes documents in the background.
- Use `http://127.0.0.1:8000`, not `https://127.0.0.1:8000`. TLS is not configured in `app.py`.
- While indexing is running, `/api/health` returns `status: "in_progress"` plus file progress details.
- Thread state is stored in memory and resets when the server restarts.
- Answers are returned in concise Markdown and end with `Reference: <policy-file-name>`.
- The original CLI still works through `python policy_gpt_poc.py`.
