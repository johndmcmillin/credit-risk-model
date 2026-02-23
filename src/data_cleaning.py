# src/data_cleaning.py
# Utility functions for data cleaning — extracted from notebooks for reuse
# Populate as you refine your pipeline

def load_and_filter_terminal_loans(filepath):
    """Load Lending Club data and filter to terminal loan outcomes only."""
    import pandas as pd
    
    df = pd.read_csv(filepath, low_memory=False)
    
    terminal_statuses = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    
    df = df[df['loan_status'].isin(terminal_statuses.keys())].copy()
    df['default'] = df['loan_status'].map(terminal_statuses)
    
    return df
