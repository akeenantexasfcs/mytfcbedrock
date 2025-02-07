#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import logging
import os
import streamlit as st
import uuid
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Basic configuration
BEDROCK_AGENT_ID = "VLZFRY26GV"
BEDROCK_AGENT_ALIAS_ID = "QT0I0B0VIG"
UI_TITLE = "SOUTHERN AG AGENT"

# Page setup
st.set_page_config(page_title=UI_TITLE, layout="wide")
st.title(UI_TITLE)

# Debug mode toggle
with st.sidebar:
    debug_mode = st.checkbox("Enable Debug Mode", value=False)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# AWS Configuration
try:
    credentials = {
        "aws_access_key_id": st.secrets["aws"]["access_key_id"].strip(),
        "aws_secret_access_key": st.secrets["aws"]["secret_access_key"].strip(),
        "region_name": st.secrets["aws"]["region"].strip()
    }
    
    session = boto3.Session(**credentials)
    bedrock_client = session.client("bedrock-agent-runtime")
    st.sidebar.success(f"Connected to AWS Region: {credentials['region_name']}")
except Exception as e:
    st.error("Failed to initialize AWS. Check your credentials.")
    st.stop()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input():
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process response
    with st.chat_message("assistant"):
        try:
            # Get response from Bedrock
            response = bedrock_client.invoke_agent(
                agentId=BEDROCK_AGENT_ID,
                agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                sessionId=st.session_state.session_id,
                inputText=prompt
            )
            
            if debug_mode:
                st.write("Raw Response:", response)
            
            # Extract completion text - this varies based on response format
            try:
                # Method 1: Direct text extraction
                text = response.get('completion', {}).get('promptOutput', {}).get('text', '')
                
                # Method 2: Extract from chunks if streaming
                if not text and hasattr(response, '__iter__'):
                    chunks = []
                    for chunk in response:
                        if debug_mode:
                            st.write("Chunk:", chunk)
                        # Try different ways to get text from chunk
                        if hasattr(chunk, 'get_text'):
                            chunks.append(chunk.get_text())
                        elif hasattr(chunk, 'get'):
                            text_data = chunk.get('chunk', {}).get('bytes', b'').decode('utf-8')
                            chunks.append(text_data)
                    text = ''.join(chunks)
                
                # If still no text, try observation data
                if not text and 'observation' in response:
                    observations = response['observation']
                    for obs in observations:
                        if 'finalResponse' in obs:
                            text = obs['finalResponse'].get('text', '')
                
                if not text:
                    text = "No response text found"
                
            except Exception as e:
                if debug_mode:
                    st.error(f"Error extracting text: {str(e)}")
                text = "Error processing response"
            
            # Add response to history and display
            st.session_state.messages.append({"role": "assistant", "content": text})
            st.markdown(text)
            
            if debug_mode:
                st.write("Processing complete")
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            if debug_mode:
                st.write("Full error:", e)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

