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

# Configure logging using YAML
if os.path.exists("logging.yaml"):
    with open("logging.yaml", "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)
else:
    log_level = logging.getLevelNamesMapping()[st.secrets.get("LOG_LEVEL", "INFO")]
    logging.basicConfig(level=log_level)

logger = logging.getLogger(__name__)

# Get config from Streamlit secrets
agent_id = st.secrets["BEDROCK_AGENT_ID"]
agent_alias_id = st.secrets["BEDROCK_AGENT_ALIAS_ID"]
ui_title = st.secrets["BEDROCK_AGENT_TEST_UI_TITLE"]
ui_icon = st.secrets["BEDROCK_AGENT_TEST_UI_ICON"]

# Initialize the Bedrock Runtime client
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

def init_session_state():
    """Initialize Streamlit session state variables."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}

def process_bedrock_event_stream(event_stream):
    """
    Given a botocore.eventstream.EventStream object, parse the chunks
    and build the full completion text, citations, and trace.
    Returns (full_text, citations, trace).
    """
    full_text = ""
    citations = []
    trace = {}

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

            except json.JSONDecodeError as e:
                # If the chunk isn't valid JSON, log a warning and append raw text
                logger.warning(f"Chunk not valid JSON: {e}, chunk={chunk_data}")
                full_text += chunk_data

    return full_text, citations, trace

def process_bedrock_json_response(response):
    """
    Process a synchronous JSON response from Bedrock Agent Runtime.
    Returns (full_text, citations, trace).
    """
    full_text = response.get("completion", "")
    citations = response.get("citations", [])
    trace = response.get("trace", {})
    return full_text, citations, trace

# General page configuration and initialization
st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)

# Initialize session state if empty
if len(st.session_state.items()) == 0:
    init_session_state()

# Sidebar button to reset session
with st.sidebar:
    if st.button("Reset Session"):
        init_session_state()

# Display any existing messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input that invokes the agent
prompt = st.chat_input()
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.empty():
            output_text = ""
            citations = []
            trace = {}

            with st.spinner("Processing..."):
                try:
                    logger.debug(
                        f"Invoking agent with parameters: agentId={agent_id}, "
                        f"agentAliasId={agent_alias_id}, sessionId={st.session_state.session_id}"
                    )
                    response = bedrock_agent_runtime.invoke_agent(
                        agentId=agent_id,
                        agentAliasId=agent_alias_id,
                        sessionId=st.session_state.session_id,
                        inputText=prompt
                    )

                    # Examine what we got back
                    logger.debug(f"Response keys: {list(response.keys())}")
                    content_type = response.get("contentType")

                    # Handle streaming vs. non-streaming
                    if content_type == "text/event-stream" and "body" in response:
                        output_text, citations, trace = process_bedrock_event_stream(response["body"])
                    elif "completion" in response:
                        output_text, citations, trace = process_bedrock_json_response(response)
                    else:
                        logger.error(
                            "Bedrock agent response has no 'body' or 'completion' attribute. "
                            f"Response keys: {list(response.keys())}"
                        )
                        output_text = (
                            "No valid event stream or completion returned by the agent. "
                            "Please try again or check logs."
                        )

                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    error_message = e.response['Error']['Message']
                    logger.error(f"AWS API Error: {error_code} - {error_message}")
                    output_text = (
                        "I encountered an AWS service error. Please try again later. "
                        f"Error: {error_code}"
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    output_text = (
                        "I had trouble processing the JSON response. Please try again."
                    )
                except Exception as e:
                    logger.error(f"Unexpected error: {str(e)}")
                    output_text = (
                        "I apologize, but I encountered an unexpected error. Please try again."
                    )

            # Add citations if available
            if citations:
                citation_num = 1
                # Convert placeholders %[#]% to superscript references
                output_text = re.sub(r"%\[(\d+)\]%", r"<sup>[\1]</sup>", output_text)
                citation_locs = ""
                for citation_item in citations:
                    for retrieved_ref in citation_item.get("retrievedReferences", []):
                        citation_marker = f"[{citation_num}]"
                        match retrieved_ref['location']['type']:
                            case 'CONFLUENCE':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['confluenceLocation']['url']}"
                            case 'CUSTOM':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['customDocumentLocation']['id']}"
                            case 'KENDRA':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['kendraDocumentLocation']['uri']}"
                            case 'S3':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['s3Location']['uri']}"
                            case 'SALESFORCE':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['salesforceLocation']['url']}"
                            case 'SHAREPOINT':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['sharePointLocation']['url']}"
                            case 'SQL':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['sqlLocation']['query']}"
                            case 'WEB':
                                citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['webLocation']['url']}"
                            case _:
                                logger.warning(f"Unknown location type: {retrieved_ref['location']['type']}")
                        citation_num += 1
                output_text += f"\n{citation_locs}"

            # Store output in session
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            st.session_state.citations = citations
            st.session_state.trace = trace

            # Display assistant response
            st.markdown(output_text, unsafe_allow_html=True)

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

# Sidebar Trace Section
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

                # Group traces by traceId
                for trace_event in st.session_state.trace[trace_type]:
                    if trace_type in trace_info_types_map:
                        for info_type in trace_info_types_map[trace_type]:
                            if info_type in trace_event:
                                trace_id = trace_event[info_type]["traceId"]
                                trace_steps.setdefault(trace_id, []).append(trace_event)
                                break
                    else:
                        # Fallback if no known structure
                        trace_id = trace_event.get("traceId", f"no-id-{step_num}")
                        trace_steps.setdefault(trace_id, []).append(trace_event)

                # Display each grouped step
                for trace_id in trace_steps:
                    with st.expander(f"Trace Step {step_num}", expanded=False):
                        for entry in trace_steps[trace_id]:
                            st.code(json.dumps(entry, indent=2), language="json", line_numbers=True)
                    step_num += 1

        if not has_trace:
            st.text("None")

    # Citations Section
    st.subheader("Citations")
    if len(st.session_state.citations) > 0:
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

