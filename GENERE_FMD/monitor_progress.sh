#!/bin/bash
# Monitor the progress of the full dataset processing

echo "========================================="
echo "🎵 MUSIC GENRE CLUSTERING - PROGRESS MONITOR"
echo "========================================="
echo ""

# Check if process is running
if pgrep -f "main.py" > /dev/null; then
    echo "✅ Pipeline is RUNNING"
    echo ""
else
    echo "❌ Pipeline is NOT running"
    echo ""
fi

# Show last 30 lines of log
if [ -f "run_log_full_dataset.txt" ]; then
    echo "📋 Latest Log Output:"
    echo "========================================="
    tail -n 30 run_log_full_dataset.txt
    echo ""
fi

# Count processed files
if [ -f "results/extracted_features.csv" ]; then
    lines=$(wc -l < results/extracted_features.csv)
    processed=$((lines - 1))
    percentage=$(awk "BEGIN {printf \"%.2f\", ($processed/8000)*100}")
    echo "========================================="
    echo "📊 Feature Extraction Progress:"
    echo "   Processed: $processed / 8000 files"
    echo "   Progress: $percentage%"
    echo "========================================="
fi

echo ""
echo "💡 To run this monitor continuously, use:"
echo "   watch -n 30 ./monitor_progress.sh"
