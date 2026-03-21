#!/bin/bash
# Launch the NeuroRehab Streamlit dashboard

cd "$(dirname "$0")/.."
echo "🧠 NeuroRehab — Starting Dashboard"
echo "Open your browser at: http://localhost:8501"
echo ""
streamlit run ui/dashboard.py \
    --server.headless true \
    --server.port 8501 \
    --theme.base dark \
    --theme.primaryColor "#00c8aa" \
    --theme.backgroundColor "#080d1a" \
    --theme.secondaryBackgroundColor "#0f1729" \
    --theme.textColor "#e0e8ff"
