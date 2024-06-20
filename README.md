# Chatbot Project

## Setup Instructions

### Backend

1. Ensure you have Python 3 installed. Check with:

   ```bash
   python3 --version
   ```

2. Navigate to the backend directory:

   ```bash
   cd backend
   ```

3. Create a virtual environment and activate it using Python 3:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Create a .env file by copying the example file and set your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

6. Create the Chroma database:

   ```bash
   python create_chroma_db.py
   ```

7. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install the dependencies:

   ```bash
   npm install
   ```

3. Run the Vite development server:

   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:5173`

### Interacting with the Chatbot

- Use the input field to type your query and hit 'Ask'.
- The chatbot will respond, and the chat history will be displayed below.
