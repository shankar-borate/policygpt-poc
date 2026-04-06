# Policy GPT

This project exposes the Policy GPT retrieval pipeline through FastAPI and serves a built-in chat UI.

## Run

1. Install dependencies:

```powershell
pip install -r requirements.txt
```

2. Set config in [policygpt/config.py](/d:/policy-mgmt/policygpt-poc/policygpt/config.py).

Change this one value:

```python
ai_profile = "openai"
```

Available values:

```python
ai_profile = "openai"
ai_profile = "bedrock-20b"
ai_profile = "bedrock-120b"
ai_profile = "bedrock-claude-sonnet-4-6"
ai_profile = "bedrock-claude-opus-4-6"
```

Then choose the shared cost/accuracy preset for any model:

```python
accuracy_profile = "high"
accuracy_profile = "medium"
accuracy_profile = "low"
```

Secrets still come from your environment:

OpenAI:

```powershell
$env:OPENAI_API_KEY="your-key"
```

Amazon Bedrock:

Use your normal AWS credentials plus optional region override:

```powershell
$env:AWS_BEDROCK_REGION="ap-south-1"
```

Optional accuracy override:

```powershell
$env:POLICY_GPT_ACCURACY_PROFILE="medium"
```

3. Start the app.

Foreground:

```powershell
python app.py
```

Background:

```powershell
.\run.bat
```

```bash
./run.sh
```

Both scripts write to `app.log` and store the process id in `app.pid`.

4. Open `http://127.0.0.1:8010`

## Notes

- The document folder is fixed in code to `D:\policy-mgmt\vcx_policies`.
- Supported policy files are `.html`, `.htm`, `.txt`, and text-based `.pdf`.
- The server starts immediately and indexes documents in the background.
- Use `http://127.0.0.1:8010`, not `https://127.0.0.1:8010`. TLS is not configured in `app.py`.
- While indexing is running, `/api/health` returns `status: "in_progress"` plus file progress details.
- Thread state is stored in memory and resets when the server restarts.
- Answers are returned in concise Markdown and end with `Reference: <policy-file-name>`.
- When `Config.debug` is `True`, each LLM call writes a request file to `<debug_log_dir>\llm\requests` and a matching response file to `<debug_log_dir>\llm\responses`.
- Override the LLM debug log location with `POLICY_GPT_DEBUG_LOG_DIR` and disable debug logging with `POLICY_GPT_DEBUG=0`.
- The original CLI still works through `python policy_gpt_poc.py`.
- `Config.ai_profile` is the model-selection switch.
- `Config.accuracy_profile` is the shared cost/quality switch for all models.
- `openai` uses `gpt-4.1` with `text-embedding-3-large`.
- `bedrock-20b` uses `openai.gpt-oss-20b-1:0` with `amazon.titan-embed-text-v2:0`.
- `bedrock-120b` uses `openai.gpt-oss-120b-1:0` with `amazon.titan-embed-text-v2:0`.
- `bedrock-claude-sonnet-4-6` uses the Bedrock global inference profile `global.anthropic.claude-sonnet-4-6` with `amazon.titan-embed-text-v2:0`.
- `bedrock-claude-opus-4-6` uses the Bedrock global inference profile `global.anthropic.claude-opus-4-6-v1` with `amazon.titan-embed-text-v2:0`.
- Claude Bedrock profiles are sent through the Bedrock `Converse` API, while the existing GPT-OSS Bedrock profiles continue using the OpenAI-compatible `InvokeModel` payload.
- `accuracy_profile = "high"` keeps the widest retrieval/prompt budgets, `medium` trims them, and `low` is the most cost-conscious.
