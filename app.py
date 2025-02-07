#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import logging
import os
import re
import time
import requests
import streamlit as st
import uuid
import boto3
import botocore
from botocore.eventstream import EventStream
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
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

#####################################
# Initialize session state
#####################################
def init_session_state():
    """
    Ensures all necessary session state variables are defined.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citations" not in st.session_state:
        st.session_state.citations = []
    if "trace" not in st.session_state:
        st.session_state.trace = {}
    if "knowledge_bases" not in st.session_state:
        # This is the fix to ensure knowledge_bases always exists
        st.session_state.knowledge_bases = None

#####################################
# Page setup
#####################################
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
else:
    # If the script re-ran, still ensure everything is set
    init_session_state()

#####################################
# Set up AWS credentials
#####################################
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

#####################################
# Initialize Bedrock client
#####################################
try:
    session = boto3.Session(**credentials)
    bedrock_client = session.client("bedrock-agent-runtime")
    logger.info("Bedrock client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Bedrock client: {str(e)}")
    st.error("Failed to initialize Bedrock client. Please check your AWS configuration.")
    st.stop()

st.sidebar.success(f"Connected to AWS Region: {credentials['region_name']}")

#####################################
# Knowledge Base Listing Code
#####################################
def list_knowledge_bases(max_results=10, next_token=None):
    """
    Calls the /knowledgebases/ endpoint directly using SigV4 signing.
    POST /knowledgebases/
    { "maxResults": number, "nextToken": "string" }
    """
    region = credentials["region_name"]
    service = "bedrock"
    host = f"bedrock.{region}.amazonaws.com"
    endpoint = f"https://{host}/knowledgebases/"
    
    payload = {"maxResults": max_results}
    if next_token:
        payload["nextToken"] = next_token
    
    data = json.dumps(payload)
    
    req = AWSRequest(
        method="POST",
        url=endpoint,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    botocore_creds = session.get_credentials()
    req.context["client_region"] = region
    req.context["has_streaming_input"] = False

    SigV4Auth(botocore_creds, service, region).add_auth(req)

    prepared_request = requests.Request(
        method=req.method,
        url=req.url,
        headers=dict(req.headers),
        data=req.data
    ).prepare()

    with requests.Session() as s:
        response = s.send(prepared_request)
    
    if debug_mode:
        st.write("ListKnowledgeBases status code:", response.status_code)
        try:
            st.json(response.json())
        except Exception as ex:
            st.write("Could not parse JSON from response:", str(ex))

    if response.status_code != 200:
        raise Exception(f"ListKnowledgeBases failed: {response.status_code} - {response.text}")
    
    return response.json()

#####################################
# Sidebar buttons
#####################################
with st.sidebar:
    if st.button("Reset Session"):
        init_session_state()
        st.experimental_rerun()

    # Button to list knowledge bases
    if st.button("List Knowledge Bases"):
        try:
            kb_data = list_knowledge_bases()
            st.session_state.knowledge_bases = kb_data
            st.success("Successfully listed knowledge bases.")
        except Exception as e:
            st.error(f"Could not list KBs: {str(e)}")

# If we've listed knowledge bases, show them
if st.session_state.knowledge_bases:
    st.sidebar.write("Knowledge Bases Found:")
    for kb in st.session_state.knowledge_bases.get("knowledgeBaseSummaries", []):
        kb_name = kb.get("name", "UnknownName")
        kb_id = kb.get("knowledgeBaseId", "UnknownID")
        kb_status = kb.get("status", "N/A")
        st.sidebar.write(f"• {kb_name} (ID={kb_id}, Status={kb_status})")

#####################################
# Throttling / EventStream
#####################################
def process_event_stream(response_stream):
    logger.info("Processing EventStream response from Bedrock agent.")
    output_text = []

    for event in response_stream:
        if debug_mode:
            st.write("Event received:", event)
        logger.info(f"Received event: {event}")

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
    debug_log_dict(response)

    top_keys = list(response.keys())
    debug_log(f"Top-level keys in response: {top_keys}")

    if "completion" in response and isinstance(response["completion"], (EventStream, botocore.eventstream.EventStream)):
        debug_log("'completion' is an EventStream -> using process_event_stream()")
        return process_event_stream(response["completion"])

    if "modelInvocationOutput" in response:
        debug_log("Detected 'modelInvocationOutput' key -> Attempting to parse agent response")
        agent_output = response["modelInvocationOutput"]
        raw_resp = agent_output.get("rawResponse", {})
        raw_json_str = raw_resp.get("content", "")

        debug_log(f"raw_json_str length: {len(raw_json_str)}")

        if raw_json_str:
            try:
                parsed_json = json.loads(raw_json_str)
                debug_log(f"Keys in parsed_json: {list(parsed_json.keys())}")

                if "observation" in response and isinstance(response["observation"], list):
                    debug_log(f"Found 'observation' array with length {len(response['observation'])}")
                    final_obs = response["observation"][-1]
                    if isinstance(final_obs, dict) and "finalResponse" in final_obs:
                        final_resp = final_obs["finalResponse"].get("text", "")
                        if final_resp:
                            debug_log(f"Extracted finalResponse.text with length: {len(final_resp)}")
                            return final_resp

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
                debug_log(f"JSONDecodeError parsing Agent 'rawResponse': {str(jerr)}")
                return "Error: Failed to parse the Agent rawResponse JSON."

    if "completion" in response:
        debug_log("Detected top-level 'completion' key -> Attempting to parse model style response")
        completion = response["completion"]
        debug_log(f"completion: {completion}")

        if not isinstance(completion, dict):
            debug_log("completion is not a dict, can't parse further.")
            return "Error: Invalid 'completion' format"

        if "promptOutput" in completion:
            prompt_output = completion["promptOutput"]
            debug_log(f"Found 'promptOutput' in completion. Keys: {list(prompt_output.keys()) if isinstance(prompt_output, dict) else 'not dict'}")
            if isinstance(prompt_output, dict):
                text = prompt_output.get("text", "")
                if text:
                    debug_log(f"Extracted text from completion.promptOutput with length: {len(text)}")
                    return text
                else:
                    debug_log("No text in promptOutput.")

        if "text" in completion:
            text_val = completion["text"]
            if text_val:
                debug_log(f"Extracted text from completion['text'] with length: {len(text_val)}")
                return text_val
            else:
                debug_log("completion['text'] is empty.")

        debug_log("No recognized sub-structure in 'completion'. Returning error.")
        return "Error: Invalid response format"

    debug_log("No recognized response format found. Returning error.")
    return "Error: Invalid response format"

def invoke_agent_with_backoff(prompt, max_retries=2, backoff_base=60):
    """
    Attempt to call invoke_agent. If we get throttled, wait and retry up to max_retries times.
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
            return response  # success, return
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
                    raise e  # re-raise so we display an error
            else:
                raise e
        except Exception as ex:
            logger.error(f"Unexpected error: {str(ex)}", exc_info=True)
            raise ex

#####################################
# Chat Input Logic
#####################################
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

#####################################
# Conversation Rerender
#####################################
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

