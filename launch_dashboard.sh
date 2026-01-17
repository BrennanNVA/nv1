#!/bin/bash
# Launch Nova Aetus Dashboard

set -e

echo "🚀 Launching Nova Aetus Dashboard..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing dependencies..."
    pip install -q streamlit plotly
fi

# Check if config exists
if [ ! -f "config.toml" ]; then
    echo "⚠️  Warning: config.toml not found. Dashboard may not work correctly."
fi

# Launch dashboard
echo "📊 Starting Streamlit dashboard..."
echo "   Dashboard will open at: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the dashboard"
echo ""

cd "$(dirname "$0")"
streamlit run src/nova/dashboard/app.py --server.port 8501 --server.address localhost
