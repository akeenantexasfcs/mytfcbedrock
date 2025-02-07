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

# Initialize session state
def init_session_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}

# Page setup
st.set_page_config(page_title=UI_TITLE, layout="wide")
st.title(UI_TITLE)

# Debug mode toggle
with st.sidebar:
    debug_mode = st.checkbox("Enable Debug Mode", value=False)
    if debug_mode:
        st.info("Debug mode enabled")

# Helper function for logging in debug mode
def debug_log(message):
    if debug_mode:
        st.info(f"🔍 Debug: {message}")
    logger.info(message)

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

    logger.info(f"AWS Region: {credentials['region_name']}")
    logger.info(f"Access Key ID length: {len(credentials['aws_access_key_id'])}")
    logger.info(f"Secret Key length: {len(credentials['aws_secret_access_key'])}")

    os.environ["AWS_ACCESS_KEY_ID"] = credentials["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = credentials["region_name"]

except Exception as e:
    logger.error(f"Error setting up AWS credentials: {str(e)}")
    st.error("Failed to configure AWS credentials. Please check your secrets configuration.")
    st.stop()

# Initialize Bedrock client
try:
    session = boto3.Session(**credentials)
    bedrock_client = session.client("bedrock-agent-runtime")
    logger.info("Bedrock client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Bedrock client: {str(e)}")
    st.error("Failed to initialize Bedrock client. Please check your AWS configuration.")
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

# Function to process streaming EventStream responses
def process_event_stream(response_stream):
    """Processes streaming responses from Bedrock."""
    logger.info("Processing EventStream response from Bedrock agent.")
    output_text = []

    for event in response_stream:
        if debug_mode:
            st.write("Event received:", event)
        logger.info(f"Received event: {event}")

        # Extract text from EventStream chunk
        try:
            if isinstance(event, dict) and "chunk" in event:
                chunk_data = event["chunk"]
                if isinstance(chunk_data, dict) and "bytes" in chunk_data:
                    chunk_text = chunk_data["bytes"].decode("utf-8")
                    output_text.append(chunk_text)
        except Exception as e:
            logger.error(f"Error processing event stream chunk: {str(e)}")
            continue

    final_text = "".join(output_text)
    logger.info(f"Final extracted text length: {len(final_text)}")
    return final_text if final_text else "No response generated"

# Function to process dictionary responses
def process_dict_response(response):
    """Processes dictionary responses from Bedrock."""
    try:
        completion = response.get("completion", {})
        if not isinstance(completion, dict):
            return "Error: Invalid completion format"

        prompt_output = completion.get("promptOutput", {})
        if not isinstance(prompt_output, dict):
            return "Error: Invalid promptOutput format"

        text = prompt_output.get("text", "No response generated")
        logger.info(f"Successfully extracted text of length: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Error processing dictionary response: {str(e)}", exc_info=True)
        return "Error processing response"

# Chat input and response handling
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.empty():
            try:
                logger.info(f"Invoking Bedrock agent with prompt: {prompt}")
                debug_log("Starting agent invocation...")

                with st.spinner("Processing your request..."):
                    # Log agent configuration
                    logger.info(f"Agent ID: {BEDROCK_AGENT_ID}")
                    logger.info(f"Agent Alias ID: {BEDROCK_AGENT_ALIAS_ID}")
                    logger.info(f"Session ID: {st.session_state.session_id}")

                    # Invoke Bedrock agent
                    response = bedrock_client.invoke_agent(
                        agentId=BEDROCK_AGENT_ID,
                        agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                        sessionId=st.session_state.session_id,
                        inputText=prompt,
                    )

                    # Initialize default values
                    output_text = "No response generated"
                    citations = []
                    trace = {}

                    # FIRST check if it's an EventStream
                    if isinstance(response, (botocore.eventstream.EventStream, EventStream)):
                        output_text = process_event_stream(response)

                    # If response is a dictionary, process it normally
                    elif isinstance(response, dict):
                        output_text = process_dict_response(response)
                        citations = response.get("citations", [])
                        trace = response.get("trace", {})

                    else:
                        logger.error(f"Unexpected response type: {type(response)}")
                        output_text = "Error: Unexpected response format from Bedrock."

                    if not output_text:
                        output_text = "No response generated"

                    logger.info(f"Extracted output text: {output_text}")
                    logger.info(f"Citations found: {len(citations)}")
                    logger.info(f"Trace data present: {'Yes' if trace else 'No'}")

                    if debug_mode:
                        st.json(
                            {
                                "Response Type": str(type(response)),
                                "Output Length": len(output_text),
                                "Citations Count": len(citations),
                                "Has Trace": bool(trace),
                            }
                        )

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                error_message = e.response["Error"]["Message"]
                logger.error(f"AWS Error: {error_code} - {error_message}")
                st.error(f"AWS Error: {error_message}")
                output_text = "Sorry, there was an error processing your request."

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred: {str(e)}")
                output_text = "Sorry, there was an error processing your request."

            # Update session state and display response
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            st.markdown(output_text, unsafe_allow_html=True)

            if debug_mode:
                st.write("Response processing complete")

