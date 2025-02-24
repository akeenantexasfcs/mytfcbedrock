#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import json
import logging
import os
import re
import time
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

def init_session_state():
    """Ensures all necessary session state variables are defined."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citations" not in st.session_state:
        st.session_state.citations = []
    if "trace" not in st.session_state:
        st.session_state.trace = {}

# Page setup
st.set_page_config(
    page_title=st.secrets.get("UI_TITLE", "thecapitalmarketsagent"),
    layout="wide"
)
st.title(st.secrets.get("UI_TITLE", "thecapitalmarketsagent"))

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
else:
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

# Get Bedrock agent configuration from secrets
try:
    BEDROCK_AGENT_ID = st.secrets["bedrock"]["agent_id"]
    BEDROCK_AGENT_ALIAS_ID = st.secrets["bedrock"]["agent_alias_id"]
    logger.info(f"Loaded Bedrock agent configuration from secrets")
except Exception as e:
    logger.error(f"Error loading Bedrock agent configuration: {str(e)}")
    st.error("Failed to load Bedrock agent configuration from secrets. Please check your secrets configuration.")
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

# Button to reset session
with st.sidebar:
    if st.button("Reset Session"):
        init_session_state()
        # We'll stop so that the next user action re-runs the script
        st.stop()

#####################
# Chat / Agent Logic
#####################

def process_event_stream(response_stream):
    """Processes streaming responses from Bedrock agent."""
    logger.info("Processing EventStream response from Bedrock agent.")
    output_text = []

    for event in response_stream:
        if debug_mode:
            st.write("Event received:", event)
        logger.info(f"Received event: {event}")

        # Check for immediate throttling error
        if isinstance(event, dict) and "error" in event:
            err = event["error"].get("message", "")
            if "Your request rate is too high" in err:
                logger.error("Throttling encountered in event stream chunk.")
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ThrottlingException",
                            "Message": "Your request rate is too high (event-stream)."
                        }
                    },
                    operation_name="invoke_agent"
                )

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

def parse_usage_info(response):
    """Extract usage stats (e.g. tokens used) from the response if available."""
    usage_data = {}
    try:
        if isinstance(response, dict) and "modelInvocationOutput" in response:
            model_inv_output = response["modelInvocationOutput"]
            metadata = model_inv_output.get("metadata", {})
            usage_data = metadata.get("usage", {})
    except Exception as e:
        logger.warning(f"Could not parse usage info: {str(e)}", exc_info=True)
    return usage_data

def debug_log_dict(response_dict, message="Raw dictionary response from Bedrock:"):
    debug_log(message)
    try:
        debug_log(json.dumps(response_dict, indent=2, default=str))
    except Exception as e:
        debug_log(f"Failed to JSON-dump the response: {str(e)}")

def process_dict_response(response):
    """Processes dictionary responses from Bedrock agent."""
    debug_log_dict(response)

    top_keys = list(response.keys())
    debug_log(f"Top-level keys in response: {top_keys}")

    # If we have an EventStream in 'completion'
    if "completion" in response and isinstance(response["completion"], (EventStream, botocore.eventstream.EventStream)):
        debug_log("'completion' is an EventStream -> using process_event_stream()")
        return process_event_stream(response["completion"])

    # Otherwise, check standard agent response
    if "modelInvocationOutput" in response:
        debug_log("Detected 'modelInvocationOutput' -> Attempting to parse agent response")
        agent_output = response["modelInvocationOutput"]
        raw_resp = agent_output.get("rawResponse", {})
        raw_json_str = raw_resp.get("content", "")

        debug_log(f"raw_json_str length: {len(raw_json_str)}")

        if raw_json_str:
            try:
                parsed_json = json.loads(raw_json_str)
                debug_log(f"Keys in parsed_json: {list(parsed_json.keys())}")

                # Sometimes finalResponse is in response["observation"][-1]
                if "observation" in response and isinstance(response["observation"], list):
                    debug_log(f"Found 'observation' array with length {len(response['observation'])}")
                    final_obs = response["observation"][-1]
                    if isinstance(final_obs, dict) and "finalResponse" in final_obs:
                        final_resp = final_obs["finalResponse"].get("text", "")
                        if final_resp:
                            debug_log(f"Extracted finalResponse.text with length: {len(final_resp)}")
                            return final_resp

                # Or parse 'content' array
                if "content" in parsed_json and isinstance(parsed_json["content"], list):
                    text_chunks = []
                    for item in parsed_json["content"]:
                        if "text" in item and item["text"]:
                            text_chunks.append(item["text"])
                    joined_text = "\n".join(text_chunks).strip()
                    if joined_text:
                        debug_log(f"Extracted joined_text with length: {len(joined_text)}")
                        return joined_text

                debug_log("No text found in 'observation' or 'content' for Agent response.")
                return "No response generated from Bedrock Agent."

            except json.JSONDecodeError as jerr:
                debug_log(f"JSONDecodeError parsing rawResponse: {str(jerr)}")
                return "Error: Failed to parse the Agent rawResponse JSON."

    # Or a "model" style 'completion'
    if "completion" in response:
        debug_log("Detected top-level 'completion' -> Attempting to parse model style")
        completion = response["completion"]
        debug_log(f"completion: {completion}")

        if not isinstance(completion, dict):
            debug_log("completion is not a dict.")
            return "Error: Invalid 'completion' format"

        if "promptOutput" in completion:
            prompt_output = completion["promptOutput"]
            debug_log(f"Found 'promptOutput'. Keys: {list(prompt_output.keys()) if isinstance(prompt_output, dict) else 'not dict'}")
            if isinstance(prompt_output, dict):
                text = prompt_output.get("text", "")
                if text:
                    debug_log(f"Extracted text from promptOutput. length: {len(text)}")
                    return text
                else:
                    debug_log("No text in promptOutput.")

        if "text" in completion:
            text_val = completion["text"]
            if text_val:
                debug_log(f"Extracted text from completion['text']. length: {len(text_val)}")
                return text_val
            else:
                debug_log("completion['text'] is empty.")

        debug_log("No recognized sub-structure in 'completion'. Returning error.")
        return "Error: Invalid response format"

    debug_log("No recognized response format found. Returning error.")
    return "Error: Invalid response format"

def invoke_agent_with_backoff(prompt, max_retries=2, backoff_base=60):
    """
    Attempt to call invoke_agent. If we get throttled, wait & retry up to max_retries times.
    backoff_base is in seconds, e.g. 60 for 1 minute wait.
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Invoking Bedrock agent (attempt {attempt+1} of {max_retries}) with prompt: {prompt}")
            response = bedrock_client.invoke_agent(
                agentId=BEDROCK_AGENT_ID,
                agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                sessionId=st.session_state.session_id,
                inputText=prompt,
            )
            return response
        except ClientError as e:
            err_code = e.response["Error"].get("Code", "")
            err_msg = e.response["Error"].get("Message", "")
            logger.error(f"AWS Error: {err_code} - {err_msg}")

            if err_code in ["ThrottlingException", "TooManyRequestsException"]:
                if attempt < (max_retries - 1):
                    wait_time = backoff_base * (2 ** attempt)
                    st.warning(f"Throttling encountered. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.error("Max retries reached. Still throttled. Please wait or reduce request rate.")
                    raise e
            else:
                raise e
        except Exception as ex:
            logger.error(f"Unexpected error: {str(ex)}", exc_info=True)
            raise ex

#################################
# Main Chat Flow
#################################
prompt = st.chat_input()
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            output_text = "No response generated"
            try:
                debug_log("Starting agent invocation with backoff.")
                response = invoke_agent_with_backoff(prompt)

                if isinstance(response, (botocore.eventstream.EventStream, EventStream)):
                    logger.info("Processing EventStream response from Bedrock agent.")
                    output_text = process_event_stream(response)
                elif isinstance(response, dict):
                    logger.info("Processing dictionary response from Bedrock agent.")
                    output_text = process_dict_response(response)
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
                    output_text = "Error: Unexpected response format from Bedrock."

                if not output_text:
                    output_text = "No response generated"

                usage_info = parse_usage_info(response)
                logger.info(f"Usage info from Bedrock: {usage_info}")

                if debug_mode and usage_info:
                    st.write("**Usage Info**")
                    st.json(usage_info)

            except ClientError as e:
                logger.error(f"ClientError after attempts: {str(e)}", exc_info=True)
                st.error(f"AWS Error: {e.response['Error']['Message']}")
                output_text = "Sorry, there was an error processing your request."
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred: {str(e)}")
                output_text = "Sorry, there was an error processing your request."

            st.session_state.messages.append({"role": "assistant", "content": output_text})

# Finally, re-render the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

