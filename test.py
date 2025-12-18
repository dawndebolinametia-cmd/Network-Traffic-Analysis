import requests

url = "http://localhost:3000/api/session"

payload = {"username": "Debbir", "password": "Anime#210305"}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())

