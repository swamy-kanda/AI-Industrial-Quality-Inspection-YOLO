import requests

BOT_TOKEN = "9092126711:AAGtwPYSAvRg2GLd8SAhXzeRG9ZtiUcVylk"

CHAT_ID = "1178667077"


def send_telegram_image(image_path,caption="Defect detected!"):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(image_path,"rb") as image:
        files = {
            "photo" : image
        }

        data = {
            "chat_id" : CHAT_ID,
            "caption" : caption
        }

        response = requests.post(
            url,
            data = data,
            files = files
        )

    if response.status_code == 200:
        print("Telegram image send successfully")
    else:
        print("Telegram sending failed: ",response.text)