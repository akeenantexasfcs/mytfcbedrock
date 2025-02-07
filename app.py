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

def debug_log(message):
    """Log to Streamlit if debug_mode is on, and always log at INFO level."""
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

st.sidebar.success(f"Connected to AWS Region: {credentials['region_name']}")

# Sidebar button to reset session
with st.sidebar:
    if st.button("Reset Session"):
        init_session_state()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

def process_event_stream(response_stream):
    """Processes streaming responses from Bedrock."""
    logger.info("Processing EventStream response from Bedrock agent.")
    output_text = []

    for event in response_stream:
        if debug_mode:
            st.write("Event received:", event)
        logger.info(f"Received event: {event}")

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

def process_dict_response(response):
    """
    Processes dictionary responses from Bedrock (Agent or Model).
    Detailed debug logs show exactly what keys exist and attempts
    to parse the JSON in rawResponse -> content if it exists.
    """
    # Log the entire dictionary for debugging:
    debug_log("Raw dictionary response from Bedrock:")
    try:
        # Dump the entire response for inspection
        debug_log(json.dumps(response, indent=2))
    except Exception as e:
        debug_log(f"Failed to JSON-dump the response: {str(e)}")

    try:
        # Show top-level keys
        debug_log(f"Top-level keys in response: {list(response.keys())}")

        # --- 1) Check if this looks like a Bedrock Agent response
        if isinstance(response, dict) and "modelInvocationOutput" in response:
            debug_log("Detected 'modelInvocationOutput' key -> Attempting to parse agent response")
            agent_output = response["modelInvocationOutput"]
            raw_resp = agent_output.get("rawResponse", {})
            raw_json_str = raw_resp.get("content", "")

            debug_log(f"raw_json_str length: {len(raw_json_str)}")

            if raw_json_str:
                try:
                    parsed_json = json.loads(raw_json_str)
                    # Log keys inside parsed JSON
                    debug_log(f"Keys in parsed_json: {list(parsed_json.keys())}")

                    # If there's an 'observation' array at the top level of 'response'
                    if "observation" in response and isinstance(response["observation"], list):
                        debug_log(f"Found 'observation' array with length {len(response['observation'])}")
                        # Check last observation for finalResponse
                        final_obs = response["observation"][-1]
                        if isinstance(final_obs, dict) and "finalResponse" in final_obs:
                            final_resp = final_obs["finalResponse"].get("text", "")
                            if final_resp:
                                debug_log(f"Extracted finalResponse.text with length: {len(final_resp)}")
                                return final_resp

                    # Otherwise, parse the 'content' array in the parsed JSON
                    # which is where Claude/LLM might store chunks
                    if "content" in parsed_json and isinstance(parsed_json["content"], list):
                        text_chunks = []
                        for item in parsed_json["content"]:
                            if "text" in item and item["text"]:
                                text_chunks.append(item["text"])
                        joined_text = "\n".join(text_chunks).strip()
                        if joined_text:
                            debug_log(f"Extracted joined_text with length: {len(joined_text)}")
                            return joined_text

                    # If we get here, we didn't find text anywhere
                    debug_log("No text found in 'observation' or 'content'.")
                    return "No response generated from Bedrock Agent."

                except json.JSONDecodeError as jerr:
                    debug_log(f"JSONDecodeError parsing Agent 'rawResponse': {str(jerr)}")
                    return "Error: Failed to parse the Agent rawResponse JSON."

        # --- 2) Check for a standard model-based response (Completion API)
        debug_log("Checking for standard 'completion' structure (model style).")
        completion = response.get("completion", {})
        if isinstance(completion, dict):
            prompt_output = completion.get("promptOutput", {})
            debug_log(f"prompt_output keys: {list(prompt_output.keys())} if dict")
            if isinstance(prompt_output, dict):
                text = prompt_output.get("text", "")
                if text:
                    debug_log(f"Extracted text from promptOutput with length: {len(text)}")
                    return text
                else:
                    debug_log("No text in promptOutput.")

        # If everything fails, we log out the fallback
        debug_log("No recognized response format found. Returning error.")
        return "Error: Invalid response format"

    except Exception as e:
        logger.error(f"Error processing dictionary response: {str(e)}", exc_info=True)
        return f"Error processing response: {str(e)}"

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

                    output_text = "No response generated"
                    citations = []
                    trace = {}

                    # 1) Check if it's an EventStream
                    if isinstance(response, (botocore.eventstream.EventStream, EventStream)):
                        logger.info("Processing EventStream response from Bedrock agent.")
                        output_text = process_event_stream(response)

                    # 2) Otherwise treat it like a dictionary
                    elif isinstance(response, dict):
                        logger.info("Processing dictionary response from Bedrock agent.")
                        output_text = process_dict_response(response)

                    else:
                        logger.error(f"Unexpected response type: {type(response)}")
                        output_text = "Error: Unexpected response format from Bedrock."

                    if not output_text:
                        output_text = "No response generated"

                    # For demonstration, logging citations/trace for your reference:
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

