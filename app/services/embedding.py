from openai import OpenAI
from app.core.config import get_settings

settings = get_settings()

client = OpenAI(
    api_key=settings.azure_openai_api_key,
    base_url=f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1/",
)

EMBEDDING_MODEL = settings.azure_openai_embedding_deployment


def generate_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


if __name__ == "__main__":
    text = "Artificial intelligence is transforming healthcare."

    vector = generate_embedding(text)

    print("Dimensions:", len(vector))
    print("First 5 values:", vector[:5])
