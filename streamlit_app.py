import streamlit as st
import simulator
import data_scraper
import os

# Streamlit Cloud Entry Point
# This file is a simple wrapper for dashboard.py for easier cloud deployment

if __name__ == "__main__":
    # Import and run the dashboard logic
    from dashboard import main
    main()
