import asyncio, sys
from hca.core.ollama_client import OllamaClient

async def test():
    c = OllamaClient(base_url='http://ollama:11434', default_model='qwen3:14b', timeout=60)
    try:
        print("Sending chat request...", flush=True)
        r = await c.chat([{"role": "user", "content": "Say hello in one word"}], max_tokens=20)
        print(f"OK: {r}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
    await c._client.aclose()

asyncio.run(test())
