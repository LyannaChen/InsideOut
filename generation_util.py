from openai import OpenAI
from ratelimiter import RateLimiter
from retrying import retry
import urllib
import base64
from google.genai import types
from google import genai

### --- ### 
# WARNING: Change the API setting according to your account
openai.api_key = YOUR_API_KEY
openai.organization = YOUR_ORGANIZATION
GEMINI_API_KEY = YOUR_GEMINI_KEY
### --- ### 


client = OpenAI(api_key=OPENAI_API_KEY, organization=ORGANIZATION)
google_client = genai.Client(api_key=GEMINI_API_KEY)

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=1200, period=60)
# Generating text outputs
def generate_chatgpt_original(utt, model='gpt4o'):
    if model == 'o4-mini':
        response = client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "user",
                    "content": utt
                }
            ]
        )
        output = response.output_text
    else:
        if model == 'gpt4o':
            model = "gpt-4o-2024-05-13"
        elif model =='gpt3.5':
            model = "gpt-3.5-turbo-0125"
        elif model == 'gpt4.1-mini':
            model = "gpt-4.1-mini-2025-04-14"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": utt}
            ]
        )
        output = response.choices[0].message.content
    return output

def generate_chatgpt_original_with_system(system_prompt, utt, model='gpt4o'):
    if model == 'o4-mini':
        response = client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": utt
                }
            ]
        )
        output = response.output_text
    else:
        if model == 'gpt4o':
            model = "gpt-4o-2024-05-13"
        elif model =='gpt3.5':
            model = "gpt-3.5-turbo-0125"
        elif model == 'gpt4.1-mini':
            model = "gpt-4.1-mini-2025-04-14"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt}, #  "You are a helpful assistant."},
                {"role": "user", "content": utt}
            ]
        )
        output = response.choices[0].message.content
    return output


# @retry(stop_max_attempt_number=10)
# @RateLimiter(max_calls=1200, period=60)
# # Generating json output
# def generate_chatgpt_json(utt):
#     response = client.chat.completions.create(
#         # model="gpt-3.5-turbo-0125",
#         model="gpt-4o-2024-05-13",
#         response_format={ "type": "json_object" },
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#             {"role": "user", "content": utt}
#         ]
#     )
#     print(response.choices[0].message.content)
#     return response.choices[0].message.content

def save_img_from_url(url, fname):
    urllib.request.urlretrieve(url, fname)
    return

# getting Dalle-3's generation
@retry(stop_max_delay=3000, wait_fixed=1000)
@RateLimiter(max_calls=600, period=60)
def get_dalle_response(prompt, quality="standard", n=1):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality=quality,
        n=n,
    )
    return response.data[0].url


def get_gemini_response(utt, system_prompt=None):
    contents = [utt]

    if system_prompt is not None:
        response = google_client.models.generate_content(
            # model="gemini-2.0-flash", # run1
            model="gemini-2.5-flash-preview-04-17",
            config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=1024)),
            contents=contents
        )
    else:
        response = google_client.models.generate_content(
            # model="gemini-2.0-flash", # run1
            model="gemini-2.5-flash-preview-04-17",
            config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024)),
            contents=contents
        )

    # print(response.text)
    return response.text