# Website Defacement Detection

A web application that detects website defacement using machine learning models.

## Project Structure
```
project/
├── backend/         # FastAPI backend
├── frontend/        # Next.js frontend
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
uvicorn main:app --reload --port 3000
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
1. Frontend will be available at: http://localhost:3001
2. Open the frontend URL in your browser
3. Enter a website URL to analyze
4. View the defacement detection results

## API Endpoints
- POST `/analyze`: Analyze a website for defacement
  - Input: `{ "url": "example.com" }`
  - Output: `{ "url": "example.com", "prediction": 0, "label": "cleaned", "confidence": 0.95 }` 