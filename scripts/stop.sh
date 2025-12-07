#!/bin/bash
# Stop script for AI Appointer Assist

echo "Stopping AI Appointer Assist..."

# Find and kill Streamlit processes
pkill -f "streamlit run src/app.py" || echo "No running instances found"

echo "Application stopped."
