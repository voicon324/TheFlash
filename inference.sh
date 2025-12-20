#!/bin/bash
# VNPT AI - The Builder Track 2
# Inference Pipeline Entry Script

set -e

echo "=========================================="
echo "VNPT AI - The Builder Track 2"
echo "Starting Inference Pipeline..."
echo "=========================================="

# Change to code directory
cd /code

# Run the prediction pipeline
python predict.py

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "Output files:"
echo "  - /code/output/submission.csv"
echo "  - /code/output/submission_time.csv"
echo "=========================================="
