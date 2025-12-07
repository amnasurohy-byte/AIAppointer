#!/bin/bash
# Start script for AI Appointer Assist

echo "Starting AI Appointer Assist..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start Streamlit application
streamlit run src/app.py \
    --server.port ${STREAMLIT_SERVER_PORT:-8501} \
    --server.address ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --server.headless ${STREAMLIT_SERVER_HEADLESS:-true}
