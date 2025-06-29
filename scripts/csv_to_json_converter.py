#!/usr/bin/env python3
"""
CSV to JSON Converter for Conversation Simulator

This script converts conversation data from CSV files to JSON format,
partitioning by company and following the target format.
"""

import pandas as pd
import json
from datetime import datetime
import os
from typing import Dict, List, Any


def load_data(conversations_path: str, opportunities_path: str) -> tuple:
    """
    Load both CSV files and return as pandas DataFrames
    
    Args:
        conversations_path: Path to conversations CSV file
        opportunities_path: Path to opportunities CSV file
        
    Returns:
        tuple: (conversations_df, opportunities_df)
    """
    conversations_df = pd.read_csv(conversations_path)
    opportunities_df = pd.read_csv(opportunities_path)
    return conversations_df, opportunities_df


def create_opportunity_mappings(opportunities_df: pd.DataFrame) -> tuple:
    """
    Create mapping dictionaries from opportunity_id to company and status
    
    Args:
        opportunities_df: DataFrame containing opportunity data
        
    Returns:
        tuple: (opportunity_to_company, opportunity_to_status)
    """
    opportunity_to_company = dict(zip(opportunities_df['opportunity_id'], opportunities_df['company']))
    opportunity_to_status = dict(zip(opportunities_df['opportunity_id'], opportunities_df['status']))
    return opportunity_to_company, opportunity_to_status


def format_timestamp(timestamp_str: str) -> str:
    """
    Convert timestamp from CSV format to ISO 8601 format
    
    Args:
        timestamp_str: Timestamp string in original format
        
    Returns:
        str: Formatted timestamp in ISO 8601 format
    """
    try:
        dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        # Handle potential different formats or return original if parsing fails
        return timestamp_str


def map_status_to_outcome(status: str) -> Dict[str, str]:
    """
    Map status value to outcome dictionary
    
    Args:
        status: Status value from opportunities_df
        
    Returns:
        dict: Outcome dictionary with name and description
    """
    status_mapping = {
        "ClosedWon": {"name": "won", "description": "Sold successfully"},
        "ClosedLost": {"name": "lost", "description": "Not sold"}
    }
    
    # Handle NULL or any other status not in mapping
    if status in status_mapping:
        return status_mapping[status]
    else:
        return {"name": "unknown", "description": "Outcome unknown"}


def process_conversations(conversations_df: pd.DataFrame, 
                         opportunity_to_company: Dict[str, str],
                         opportunity_to_status: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process conversations and organize them by company
    
    Args:
        conversations_df: DataFrame containing conversation messages
        opportunity_to_company: Mapping from opportunity_id to company
        opportunity_to_status: Mapping from opportunity_id to status
        
    Returns:
        dict: Dictionary with company as key and list of conversation objects as value
    """
    # Group messages by opportunity_id
    conversations_by_opportunity = {}
    for _, row in conversations_df.iterrows():
        opportunity_id = row['opportunity_id']
        
        # Skip if opportunity_id is not in our mappings
        if opportunity_id not in opportunity_to_company:
            continue
            
        if opportunity_id not in conversations_by_opportunity:
            conversations_by_opportunity[opportunity_id] = []
            
        # Map message type to sender
        sender = "customer" if row['type'] == 'inbound' else "agent"
        
        # Create message object
        message = {
            "sender": sender,
            "content": row['text'],
            "timestamp": format_timestamp(row['timestamp'])
        }
        
        conversations_by_opportunity[opportunity_id].append(message)
    
    # Organize conversations by company
    conversations_by_company = {}
    for opportunity_id, messages in conversations_by_opportunity.items():
        company = opportunity_to_company.get(opportunity_id)
        status = opportunity_to_status.get(opportunity_id, "NULL")
        
        if company not in conversations_by_company:
            conversations_by_company[company] = []
            
        # Create conversation object with messages and outcome
        conversation = {
            "messages": messages,
            "outcome": map_status_to_outcome(status)
        }
        
        conversations_by_company[company].append(conversation)
    
    return conversations_by_company


def save_json_by_company(conversations_by_company: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
    """
    Save conversations to JSON files by company
    
    Args:
        conversations_by_company: Dictionary with company as key and conversations as value
        output_dir: Directory to save output JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON file for each company
    for company, conversations in conversations_by_company.items():
        # Clean company name for filename
        company_name = company.replace(" ", "_").lower()
        output_path = os.path.join(output_dir, f"{company_name}_conversations.json")
        
        # Create final JSON structure
        output_data = {
            "comments": f"Conversation data for {company}",
            "conversations": conversations
        }
        
        # Write to file with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Created JSON file for {company} at: {output_path}")


def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conversations_path = os.path.join(base_dir, "data", "conversations.csv")
    opportunities_path = os.path.join(base_dir, "data", "opportunities_df.csv")
    output_dir = os.path.join(base_dir, "output")
    
    # Load data
    print("Loading CSV data...")
    conversations_df, opportunities_df = load_data(conversations_path, opportunities_path)
    print(f"Loaded {len(conversations_df)} messages and {len(opportunities_df)} opportunities")
    
    # Create mappings
    print("Creating opportunity mappings...")
    opportunity_to_company, opportunity_to_status = create_opportunity_mappings(opportunities_df)
    
    # Process conversations
    print("Processing conversations...")
    conversations_by_company = process_conversations(
        conversations_df, 
        opportunity_to_company,
        opportunity_to_status
    )
    
    # Save JSON files
    print("Saving JSON files by company...")
    save_json_by_company(conversations_by_company, output_dir)
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()
