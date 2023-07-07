import requests


def get_rest_request(text, model_name="my_model"):
    url = f"http://localhost:8501/v1/models/{model_name}:predict"

    payload = {"instances": [text]}
    response = requests.post(url=url, json=payload)
    return response


rs_rest = get_rest_request(text="classify my text")
rs_rest.json()
