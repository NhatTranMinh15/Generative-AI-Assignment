from dotenv import load_dotenv
import openai, os
from openai import AzureOpenAI

load_dotenv(verbose=True, override=True)
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
# models = openai.models.list()
# for m in models:
#     print(m.model_dump_json())

response = client.chat.completions.create(
    model="gpt-4o",  # model = "deployment_name".
    messages=[
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {"role": "user", "content": "What is the Sun?"},
    ],
    stream=False,
)

# print(response)
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)

# try:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",  # model = "deployment_name".
#         messages=[
#             {
#                 "role": "system",
#                 "content": "Assistant is a large language model trained by OpenAI.",
#             },
#             {"role": "user", "content": "What is the Sun?"},
#         ],
#         stream=False,
#     )

#     # print(response)
#     print(response.model_dump_json(indent=2))
#     print(response.choices[0].message.content)
# except Exception as e:
#     print(f"An error occurred with model : {e}")
