#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import logging
import logging.config
import os
import re
import boto3
import streamlit as st
import uuid
import yaml
from botocore.exceptions import ClientError

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
if os.path.exists("logging.yaml"):
    with open("logging.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
else:
    log_level = logging.getLevelNamesMapping()[st.secrets.get("LOG_LEVEL", "INFO")]
    logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Secrets / Configuration
# -----------------------------------------------------------------------------
agent_id = st.secrets["BEDROCK_AGENT_ID"]
agent_alias_id = st.secrets["BEDROCK_AGENT_ALIAS_ID"]
ui_title = st.secrets["BEDROCK_AGENT_TEST_UI_TITLE"]
ui_icon = st.secrets["BEDROCK_AGENT_TEST_UI_ICON"]

# -----------------------------------------------------------------------------
# Boto3 Client for Bedrock
# -----------------------------------------------------------------------------
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    """Initialize Streamlit session state variables."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}

# -----------------------------------------------------------------------------
# Helpers: Process Event Stream and JSON Responses
# -----------------------------------------------------------------------------
def process_bedrock_event_stream(event_stream):
    """
    Process a streaming response from Bedrock Agent Runtime.
    Returns (full_text, citations, trace).
    """
    full_text = ""
    citations = []
    trace = {}

    try:
        for event in event_stream:
            if "chunk" in event:
                chunk_data = event["chunk"]["bytes"].decode("utf-8")
                try:
                    chunk_json = json.loads(chunk_data)
                    if "completion" in chunk_json:
                        full_text += chunk_json["completion"]
                    if "citations" in chunk_json:
                        citations.extend(chunk_json["citations"])
                    if "trace" in chunk_json:
                        trace.update(chunk_json["trace"])
                except json.JSONDecodeError:
                    # If chunk isn't valid JSON, treat it as raw text
                    full_text += chunk_data
    except Exception as e:
        logger.error(f"Error processing event stream: {str(e)}")
        return f"Error processing response: {str(e)}", [], {}

    return full_text, citations, trace

def process_response(response):
    """
    Process both streaming and non-streaming responses from Bedrock Agent Runtime.
    Returns (full_text, citations, trace).
    """
    content_type = response.get("contentType")
    
    # Streaming scenario
    if content_type == "text/event-stream" and "body" in response:
        return process_bedrock_event_stream(response["body"])
        
    # Non-streaming JSON scenario
    elif "completion" in response:
        return response.get("completion", ""), response.get("citations", []), response.get("trace", {})
        
    else:
        logger.error(
            "Bedrock agent response has unexpected format. "
            f"Response keys: {list(response.keys())}"
        )
        return (
            "No valid response returned by the agent. Please try again.",
            [],
            {}
        )

# -----------------------------------------------------------------------------
# Streamlit App: Page / Layout Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)

# If we have a new session, initialize
if len(st.session_state.items()) == 0:
    init_session_state()

# Reset Session in sidebar
with st.sidebar:
    if st.button("Reset Session"):
        init_session_state()

# -----------------------------------------------------------------------------
# Display Existing Messages
# -----------------------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Chat Input & Invoke Agent
# -----------------------------------------------------------------------------
prompt = st.chat_input()
if prompt:
    # 1) User sends message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2) Prepare container for assistant response
    with st.chat_message("assistant"):
        with st.empty():
            with st.spinner("Processing..."):
                # Invoke the agent
                try:
                    logger.debug(
                        f"Invoking agent with agentId={agent_id}, "
                        f"agentAliasId={agent_alias_id}, "
                        f"sessionId={st.session_state.session_id}"
                    )
                    response = bedrock_agent_runtime.invoke_agent(
                        agentId=agent_id,
                        agentAliasId=agent_alias_id,
                        sessionId=st.session_state.session_id,
                        inputText=prompt
                    )

                    logger.debug(f"Agent response keys: {list(response.keys())}")
                    
                    # Process the response using our handler
                    output_text, citations, trace = process_response(response)

                except ClientError as e:
                    error_code = e.response["Error"]["Code"]
                    error_message = e.response["Error"]["Message"]
                    logger.error(f"AWS API Error: {error_code} - {error_message}")
                    output_text = (
                        "I encountered an AWS service error. Please try again later. "
                        f"Error: {error_code}"
                    )
                    citations = []
                    trace = {}

                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    output_text = (
                        "I encountered an unexpected error. Please try again."
                    )
                    citations = []
                    trace = {}

            # 3) Add citations to the output text
            if citations:
                citation_num = 1
                # Convert placeholders %[#]% to superscript references
                output_text = re.sub(r"%\[(\d+)\]%", r"<sup>[\1]</sup>", output_text)
                citation_locs = ""
                for citation_item in citations:
                    for retrieved_ref in citation_item.get("retrievedReferences", []):
                        citation_marker = f"[{citation_num}]"
                        match retrieved_ref["location"]["type"]:
                            case "CONFLUENCE":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['confluenceLocation']['url']}"
                            case "CUSTOM":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['customDocumentLocation']['id']}"
                            case "KENDRA":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['kendraDocumentLocation']['uri']}"
                            case "S3":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['s3Location']['uri']}"
                            case "SALESFORCE":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['salesforceLocation']['url']}"
                            case "SHAREPOINT":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['sharePointLocation']['url']}"
                            case "SQL":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['sqlLocation']['query']}"
                            case "WEB":
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['webLocation']['url']}"
                            case _:
                                logger.warning(
                                    f"Unknown location type: {retrieved_ref['location']['type']}"
                                )
                        citation_num += 1
                output_text += f"\n{citation_locs}"

            # 4) Update session state with the assistant's response
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            st.session_state.citations = citations
            st.session_state.trace = trace

            # 5) Display the assistant's response
            st.markdown(output_text, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: Trace & Citations
# -----------------------------------------------------------------------------
trace_types_map = {
    "Pre-Processing": ["preGuardrailTrace", "preProcessingTrace"],
    "Orchestration": ["orchestrationTrace"],
    "Post-Processing": ["postProcessingTrace", "postGuardrailTrace"]
}

trace_info_types_map = {
    "preProcessingTrace": ["modelInvocationInput", "modelInvocationOutput"],
    "orchestrationTrace": ["invocationInput", "modelInvocationInput", "modelInvocationOutput", "observation", "rationale"],
    "postProcessingTrace": ["modelInvocationInput", "modelInvocationOutput", "observation"]
}

with st.sidebar:
    st.title("Trace")

    step_num = 1
    for trace_type_header, trace_keys in trace_types_map.items():
        st.subheader(trace_type_header)
        has_trace = False

        for trace_type in trace_keys:
            if trace_type in st.session_state.trace:
                has_trace = True
                trace_steps = {}

                # Group trace events by traceId
                for trace_event in st.session_state.trace[trace_type]:
                    if trace_type in trace_info_types_map:
                        for info_type in trace_info_types_map[trace_type]:
                            if info_type in trace_event:
                                trace_id = trace_event[info_type]["traceId"]
                                trace_steps.setdefault(trace_id, []).append(trace_event)
                                break
                    else:
                        trace_id = trace_event.get("traceId", f"unknown_id_{step_num}")
                        trace_steps.setdefault(trace_id, []).append(trace_event)

                # Display each grouped step
                for trace_id in trace_steps:
                    with st.expander(f"Trace Step {step_num}", expanded=False):
                        for entry in trace_steps[trace_id]:
                            st.code(json.dumps(entry, indent=2), language="json", line_numbers=True)
                    step_num += 1

        if not has_trace:
            st.text("None")

    # Citations
    st.subheader("Citations")
    if st.session_state.citations:
        citation_num = 1
        for citation_item in st.session_state.citations:
            for retrieved_ref_num, retrieved_ref in enumerate(citation_item.get("retrievedReferences", [])):
                with st.expander(f"Citation [{citation_num}]", expanded=False):
                    citation_str = json.dumps(
                        {
                            "generatedResponsePart": citation_item.get("generatedResponsePart"),
                            "retrievedReference": citation_item["retrievedReferences"][retrieved_ref_num]
                        },
                        indent=2
                    )
                    st.code(citation_str, language="json", line_numbers=True)
                citation_num += 1
    else:
        st.text("None")

