"""Centralized model initialization.

The notebooks all import `model` from here, so swapping providers only requires
editing this file. Pick exactly one `MODEL_SPEC` / `API_KEY_ENV` pair below;
`model` is built from whichever is uncommented.
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env", override=True)

from langchain.chat_models import init_chat_model

# --- OpenAI, direct ---
# MODEL_SPEC = {"model": "openai:gpt-5.6-luna", "use_responses_api": True}
# API_KEY_ENV = "OPENAI_API_KEY"

# --- OpenAI via the LangSmith LLM Gateway ---
# Routes every model call through the Gateway so workspace policies
# (PII / secrets / allow-lists / cost caps) are enforced.
# MODEL_SPEC = {
#     "model": "openai:gpt-5.6-luna",
#     "base_url": "https://gateway.smith.langchain.com/openai",
#     "use_responses_api": True,
# }
# API_KEY_ENV = "LANGSMITH_API_KEY_GATEWAY"

# --- Anthropic (default) ---
# Haiku 5.5: fast and inexpensive, which suits a workshop. Swap in
# "anthropic:claude-sonnet-5-5" or "anthropic:claude-opus-5-5" for harder tasks.
MODEL_SPEC = {"model": "anthropic:claude-haiku-5-5"}
API_KEY_ENV = "ANTHROPIC_API_KEY"

# --- Azure OpenAI ---
# MODEL_SPEC = {"model": "azure_openai:gpt-5.6-terra", "azure_deployment": "gpt-5.6-terra"}
# API_KEY_ENV = "AZURE_OPENAI_API_KEY"

# --- AWS Bedrock (matches the Instruqt sandbox) ---
# Bedrock authenticates via the AWS credential chain (AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY, ~/.aws, or an IAM role), not a single key. The `us.`
# model ID is a cross-region inference profile, which current Claude models need.
# Note: the web_research tool (utils/search.py) still calls the Anthropic API.
# MODEL_SPEC = {
#     "model": "bedrock_converse:us.anthropic.claude-sonnet-4-6",
#     "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
# }
# API_KEY_ENV = None

model = init_chat_model(**MODEL_SPEC, **({"api_key": os.environ[API_KEY_ENV]} if API_KEY_ENV else {}))
