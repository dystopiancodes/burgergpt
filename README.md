# Burger King RAG

## Setup Instructions

### Backend

1. Create a .env file by copying the example file and set your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

2. Ensure you have Python 3 installed. Check with:

   ```bash
   python3 --version
   ```

3. Navigate to the backend directory:

   ```bash
   cd backend


   ```

4. Create a virtual environment and activate it using Python 3:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   mkdir -p backend/venv/hf_home
   mkdir -p backend/models

   ```

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

huggingface-cli login

transformers-cli download mistralai/Mistral-7B-Instruct-v0.1 --cache-dir ../backend/m
odels

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
