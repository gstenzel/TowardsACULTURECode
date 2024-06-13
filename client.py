import openai
import typing
import dotenv
import warnings
import random
import tenacity

__all__ = ["chat", "get_embedding"]

try:
    openai_config = dotenv.dotenv_values(".env")
except FileNotFoundError:
    raise FileNotFoundError(
        "No .env file found, please create on with OPENAI_API_USERNAME and ((OPENAI_API_USERNAME and OPENAI_API_PASSWORD) or OPENAI_API_KEY) environment variables"
    )


assert "OPENAI_API_URL" in openai_config, "Please set the API_URL environment variable"


if (
    "OPENAI_API_USERNAME" not in openai_config
    or "OPENAI_API_PASSWORD" not in openai_config
):
    assert (
        "OPENAI_API_KEY" in openai_config
    ), "Please set the OPENAI_API_KEY environment variable or both API_USERNAME and API_PASSWORD"
else:
    import httpx

    openai.OpenAI.custom_auth = httpx.BasicAuth(
        openai_config["OPENAI_API_USERNAME"], openai_config["OPENAI_API_PASSWORD"]
    )

if "OPENAI_API_KEY" not in openai_config:
    openai_config["OPENAI_API_KEY"] = "sk-dummykey"


client = openai.OpenAI(
    base_url=openai_config["OPENAI_API_URL"],
    api_key=openai_config["OPENAI_API_KEY"],
)


@tenacity.retry(
    stop=(
        tenacity.stop_after_attempt(3)
        #   | tenacity.stop_after_delay(60)
    ),
)
def chat(model: str, messages: typing.List[typing.Dict[str, str]], **options):
    import chatroom

    seed = options.pop("seed", random.randint(0, 1e10))
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        timeout=30,
        **options,
    )
    content = completion.choices[0].message.content
    return chatroom.Message(content=content, model=model, seed=seed)


def get_embedding(model: str, prompt: str, **options):
    import numpy as np

    try:
        return np.array(
            client.embeddings.create(model=model, input=prompt, **options)
            .data[0]
            .embedding
        )
    except openai.NotFoundError:
        warnings.warn(f"Embedding model {model} not found")
        return np.zeros(512)


if __name__ == "__main__":
    print(
        chat(model="llama3:8b", messages="whats the square root of 4?", temperature=0.5)
    )
