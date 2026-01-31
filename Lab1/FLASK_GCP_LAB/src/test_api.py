import requests

url = 'http://127.0.0.1:8080/predict'

with open("data/test.png", "rb") as img:
    response = requests.post(url, files={"image": img})

if response.status_code == 200:
    prediction = response.json()['prediction']
    print('Predicted species:', prediction)
else:
    print('Error:', response.status_code)