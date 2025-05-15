from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import tensorflow as tf
from PIL import Image
import io
from bs4 import BeautifulSoup
import re
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, use ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for text processing
MAX_WORDS = 5000
MAX_LEN = 128

# Load models and tokenizer
try:
    cnn_model = tf.keras.models.load_model('ResNet50_defaced_clf.h5')
    lstm_model = tf.keras.models.load_model('BiLSTM.h5')
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error loading models or tokenizer: {e}")

class WebsiteInput(BaseModel):
    url: str

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text

def setup_selenium():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def process_url(driver, url, screenshot_path):
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        html_content = driver.page_source
        extracted_text = extract_text_from_html(html_content)
        return extracted_text
    except (TimeoutException, WebDriverException) as e:
        print(f"Error processing {url}: {str(e)}")
        return None

def capture_screenshot(url: str):
    driver = setup_selenium()
    try:
        if not url.startswith("http"):
            url = "http://" + url
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        screenshot = driver.get_screenshot_as_png()
        html_content = driver.page_source
        extracted_text = extract_text_from_html(html_content)
        return screenshot, extracted_text
    except (TimeoutException, WebDriverException) as e:
        raise HTTPException(status_code=500, detail=f"Error processing website: {str(e)}")
    finally:
        driver.quit()

def preprocess_image(screenshot):
    img = Image.open(io.BytesIO(screenshot)).convert("RGB")
    img = img.resize((256, 144))  # CHUẨN KHỚP VỚI (144 height, 256 width)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_text(text):
    if isinstance(text, str):
        text = [text]
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded_sequences

@app.get("/")
def read_root():
    return {"message": "Website Defacement Classification API"}

@app.post("/analyze")
async def analyze_website(website: WebsiteInput):
    try:
        screenshot, html_content = capture_screenshot(website.url)
        screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')

        processed_image = preprocess_image(screenshot)
        processed_text = preprocess_text(html_content)

        image_pred = cnn_model.predict(processed_image)
        text_pred = lstm_model.predict(processed_text)

        final_pred = (image_pred + text_pred) / 2
        prediction = 1 if final_pred > 0.5 else 0

        return {
            "url": website.url,
            "prediction": prediction,
            "label": "defaced" if prediction == 1 else "cleaned",
            "confidence": float(final_pred),
            "screenshot_base64": screenshot_base64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
