from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1",
)

resp = client.chat.completions.create(
    model="claude-opus-4-6",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=200,
)

print(resp.choices[0].message.content)