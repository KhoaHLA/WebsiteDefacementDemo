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
from connect_dtb import conn, get_connection
from typing import List
from fastapi.responses import JSONResponse
from fastapi import Query
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for text processing
MAX_WORDS = 5000
MAX_LEN = 128

# Load models and tokenizer
try:
    cnn_model = tf.keras.models.load_model('models/ResNet50_defaced_clf.h5')
    lstm_model = tf.keras.models.load_model('models/BiLSTM.h5')
    with open("models/tokenizer.pickle", "rb") as handle:
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
            "confidence": format(float(final_pred), ".4f"),
            "screenshot_base64": screenshot_base64
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class DefaceItem(BaseModel):
    link_url: str
    image_url: str | None = None
    confidence: float   

@app.post("/create")
def create_deface(item: DefaceItem):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO deface (link_url, image_url, confidence)
            VALUES (?, ?, ?)
        """, item.link_url, item.image_url, item.confidence)
        conn.commit()
        return {"message": "Them du lieu thanh cong"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/all")
def get_all_deface():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, link_url, image_url, confidence, created_date FROM deface")
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "id": row.id,
                "link_url": row.link_url,
                "image_url": row.image_url,
                "confidence": row.confidence,
                "created_date": row.created_date.strftime('%Y-%m-%d %H:%M:%S') if row.created_date else None
            })

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_monthly_stats(year: int = Query(..., description="Year to fetch stats for")):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                MONTH(created_date) AS month,
                SUM(CASE WHEN confidence >= 0.5 THEN 1 ELSE 0 END) AS defaced,
                SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) AS cleaned
            FROM deface
            WHERE YEAR(created_date) = ?
            GROUP BY MONTH(created_date)
            ORDER BY month;
        """, year)
        rows = cursor.fetchall()

        result = {
            "defaced": [0] * 12,
            "cleaned": [0] * 12
        }

        for row in rows:
            month_index = row.month - 1  # tháng 1 là index 0
            result["defaced"][month_index] = row.defaced
            result["cleaned"][month_index] = row.cleaned

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/growth-rate")
def get_growth_rate(year: int = Query(..., description="Năm hiện tại")):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Đếm số lượng dòng cho năm hiện tại và năm trước
        cursor.execute("""
            SELECT
                SUM(CASE WHEN YEAR(created_date) = ? THEN 1 ELSE 0 END) AS current_year,
                SUM(CASE WHEN YEAR(created_date) = ? THEN 1 ELSE 0 END) AS last_year
            FROM deface
        """, year, year - 1)

        row = cursor.fetchone()
        current = row.current_year or 0
        last = row.last_year or 0

        # Tính phần trăm tăng/giảm
        if last == 0:
            percent_change = 100.0 if current > 0 else 0.0
        else:
            percent_change = ((current - last) / last) * 100

        return JSONResponse(content={"percent_change": round(percent_change, 2)})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/count-by-year")
def get_counts_by_year(year: int = Query(..., description="Năm cần thống kê")):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                SUM(CASE WHEN confidence >= 0.5 THEN 1 ELSE 0 END) AS defaced_count,
                SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) AS cleaned_count
            FROM deface
            WHERE YEAR(created_date) = ?
        """, year)

        row = cursor.fetchone()

        return JSONResponse(content={
            "year": year,
            "defaced": row.defaced_count or 0,
            "cleaned": row.cleaned_count or 0
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))