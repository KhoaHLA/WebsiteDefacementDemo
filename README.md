# Website Defacement Detection

This is a demo web application of Website Defacement Detection. This is the main part of our project: Website Defacement Detection Using Deep Learning - IE105 
## Project Structure
```
project/
├── backend/         # FastAPI backend        
└── README.md
```

## Setup Instructions

### Backend Setup
1. Create and activate virtual environment:
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backend server:
```bash
uvicorn main:app --reload --host localhost  --port 8080
```

### Frontend Setup
1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Run the frontend development server:
```bash
npm run dev
```

## Usage
1. Frontend will be available at: http://localhost:3000
2. Backend Swagger API Docs will be available at http://localhost:8000/docs
3. Open the frontend URL in your browser
4. Enter a website URL to analyze
5. View the defacement detection results

## API Endpoints
- POST `/analyze`: Analyze a website for defacement
  - Input: `{ "url": "example.com" }`
  - Output: `{ "url": "example.com", "prediction": 0, "label": "cleaned", "confidence": 0.95 }` 
