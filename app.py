#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import logging
import os
import re
import streamlit as st
import uuid
import boto3
import botocore
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Bedrock Agent Configuration
BEDROCK_AGENT_ID = "VLZFRY26GV"
BEDROCK_AGENT_ALIAS_ID = "QT0I0B0VIG"
UI_TITLE = "SOUTHERN AG AGENT"

def init_session_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

# Page setup
st.set_page_config(page_title=UI_TITLE, layout="wide")
st.title(UI_TITLE)

# Debug mode toggle
with st.sidebar:
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    if debug_mode:
        st.info("Debug mode enabled")

# Initialize session state if needed
if len(st.session_state.items()) == 0:
    init_session_state()

# Set up AWS credentials
try:
    credentials = {
        "aws_access_key_id": st.secrets["aws"]["access_key_id"].strip(),
        "aws_secret_access_key": st.secrets["aws"]["secret_access_key"].strip(),
        "region_name": st.secrets["aws"]["region"].strip(),
    }
    
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = credentials["region_name"]
    
except Exception as e:
    st.error("Failed to configure AWS credentials.")
    st.stop()

# Initialize Bedrock client
try:
    session = boto3.Session(**credentials)
    bedrock_client = session.client("bedrock-agent-runtime")
except Exception as e:
    st.error("Failed to initialize Bedrock client.")
    st.stop()

# Display AWS configuration status
st.sidebar.success(f"Connected to AWS Region: {credentials['region_name']}")

# Sidebar button to reset session
with st.sidebar:
    if st.button("Reset Session"):
        init_session_state()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input and response handling
if prompt := st.chat_input():
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Process response
    with st.chat_message("assistant"):
        try:
            # Invoke agent
            response = bedrock_client.invoke_agent(
                agentId=BEDROCK_AGENT_ID,
                agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                sessionId=st.session_state.session_id,
                inputText=prompt
            )
            
            # Process event stream
            output_text = []
            # Before processing starts
            if debug_mode:
                st.write("Response type:", type(response))
                st.write("Response dir:", dir(response))
                st.write("Is iterable:", hasattr(response, '__iter__'))
                st.write("Starting event processing...")
                
            event_count = 0
            for event in response:
                event_count += 1
                if debug_mode:
                    st.write(f"\nProcessing event {event_count}:")
                    st.write("Event type:", type(event))
                    st.write("Event dir:", dir(event))
                    if hasattr(event, '__dict__'):
                        st.write("Event dict:", event.__dict__)
                try:
                    if debug_mode:
                        st.write("Event:", event)
                        st.write("Event type:", type(event))
                        st.write("Event dir:", dir(event))
                    
                    # Handle completion event
                    if hasattr(event, 'completion'):
                        completion = event.completion
                        if hasattr(completion, 'promptOutput') and hasattr(completion.promptOutput, 'text'):
                            output_text.append(completion.promptOutput.text)
                            if debug_mode:
                                st.write("Found completion text:", completion.promptOutput.text)
                    
                    # Try different ways to access content
                    elif hasattr(event, 'text'):
                        output_text.append(event.text)
                    elif hasattr(event, 'content'):
                        output_text.append(event.content)
                    elif isinstance(event, dict):
                        # Try to find text in dictionary structure
                        if 'completion' in event:
                            completion = event['completion']
                            if isinstance(completion, dict) and 'promptOutput' in completion:
                                text = completion['promptOutput'].get('text', '')
                                if text:
                                    output_text.append(text)
                            
                except Exception as e:
                    if debug_mode:
                        st.error(f"Error processing event: {e}")
                        st.write("Failed event:", event)
                    continue
            
            # Combine all chunks
            final_response = "".join(output_text) if output_text else "No response generated"
            
            # Add response to history
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            st.markdown(final_response)
            
        except Exception as e:
            if debug_mode:
                st.error(f"Error: {str(e)}")
            st.error("Failed to process response")

