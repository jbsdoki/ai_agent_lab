"""Verify local Ollama responds via the localhost API."""

from ollama import Client

OLLAMA_HOST = "http://localhost:11434"
MODEL = "llama3.2:latest"
PROMPT = "What is the weather like in Los Angeles?"


def create_client(host: str) -> Client:
    return Client(host=host)


def ask_weather(client: Client, model: str, prompt: str) -> str:
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


if __name__ == "__main__":
    client = create_client(OLLAMA_HOST)
    reply = ask_weather(client, MODEL, PROMPT)
    print(reply)
