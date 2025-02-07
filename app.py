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
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bedrock Agent Configuration
BEDROCK_AGENT_ID = "VLZFRY26GV"
BEDROCK_AGENT_ALIAS_ID = "QT0I0B0VIG"
UI_TITLE = "SOUTHERN AG AGENT"

def init_session_state():
    """Initialize or reset the session state"""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}

def setup_aws_credentials():
    """Set up AWS credentials from Streamlit secrets"""
    try:
        os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["aws"]["access_key_id"]
        os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["aws"]["secret_access_key"]
        os.environ['AWS_DEFAULT_REGION'] = st.secrets["aws"]["region"]
        logger.info(f"AWS credentials configured for region: {st.secrets['aws']['region']}")
    except Exception as e:
        logger.error(f"Error setting up AWS credentials: {str(e)}")
        st.error("Failed to configure AWS credentials. Please check your secrets configuration.")
        return False
    return True

def initialize_bedrock_client():
    """Initialize the Bedrock client"""
    try:
        client = boto3.client("bedrock-agent-runtime")
        logger.info("Bedrock client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error initializing Bedrock client: {str(e)}")
        st.error("Failed to initialize Bedrock client. Please check your AWS configuration.")
        return None

# Page setup
st.set_page_config(page_title=UI_TITLE, layout="wide")
st.title(UI_TITLE)

# Initialize session state if needed
if len(st.session_state.items()) == 0:
    init_session_state()

# Set up AWS credentials
if not setup_aws_credentials():
    st.stop()

# Initialize Bedrock client
bedrock_client = initialize_bedrock_client()
if not bedrock_client:
    st.stop()

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
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Handle assistant response
    with st.chat_message("assistant"):
        with st.empty():
            try:
                logger.info(f"Invoking Bedrock agent with prompt: {prompt}")
                with st.spinner("Processing your request..."):
                    response = bedrock_client.invoke_agent(
                        agentId=BEDROCK_AGENT_ID,
                        agentAliasId=BEDROCK_AGENT_ALIAS_ID,
                        sessionId=st.session_state.session_id,
                        inputText=prompt
                    )
                    
                    # Extract response components
                    completion = response.get('completion', {})
                    output_text = completion.get('promptOutput', {}).get('text', '')
                    citations = completion.get('citations', [])
                    trace = completion.get('trace', {})
                    
                    logger.info("Successfully received response from Bedrock agent")

                    # Process JSON response if applicable
                    try:
                        output_json = json.loads(output_text, strict=False)
                        if "instruction" in output_json and "result" in output_json:
                            output_text = output_json["result"]
                    except json.JSONDecodeError:
                        pass

                    # Process citations
                    if citations:
                        citation_num = 1
                        output_text = re.sub(r"%\[(\d+)\]%", r"<sup>[\1]</sup>", output_text)
                        citation_locs = ""
                        
                        for citation in citations:
                            for retrieved_ref in citation["retrievedReferences"]:
                                citation_marker = f"[{citation_num}]"
                                location_type = retrieved_ref['location']['type']
                                
                                # Handle different citation types
                                try:
                                    match location_type:
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
                                            logger.warning(f"Unknown citation location type: {location_type}")
                                except Exception as e:
                                    logger.error(f"Error processing citation {citation_num}: {str(e)}")
                                
                                citation_num += 1
                        
                        output_text += f"\n{citation_locs}"

            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                logger.error(f"AWS Error: {error_code} - {error_message}")
                st.error(f"AWS Error: {error_message}")
                output_text = "Sorry, there was an error processing your request."
                citations = []
                trace = {}
            
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred: {str(e)}")
                output_text = "Sorry, there was an error processing your request."
                citations = []
                trace = {}

            # Update session state and display response
            st.session_state.messages.append({"role": "assistant", "content": output_text})
            st.session_state.citations = citations
            st.session_state.trace = trace
            st.markdown(output_text, unsafe_allow_html=True)

# Trace configuration
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

# Sidebar trace information
with st.sidebar:
    st.title("Trace")

    # Display trace information
    step_num = 1
    for trace_type_header in trace_types_map:
        st.subheader(trace_type_header)

        has_trace = False
        for trace_type in trace_types_map[trace_type_header]:
            if trace_type in st.session_state.trace:
                has_trace = True
                trace_steps = {}

                for trace in st.session_state.trace[trace_type]:
                    if trace_type in trace_info_types_map:
                        trace_info_types = trace_info_types_map[trace_type]
                        for trace_info_type in trace_info_types:
                            if trace_info_type in trace:
                                trace_id = trace[trace_info_type]["traceId"]
                                if trace_id not in trace_steps:
                                    trace_steps[trace_id] = [trace]
                                else:
                                    trace_steps[trace_id].append(trace)
                                break
                    else:
                        trace_id = trace["traceId"]
                        trace_steps[trace_id] = [
                            {
                                trace_type: trace
                            }
                        ]

                for trace_id in trace_steps.keys():
                    with st.expander(f"Trace Step {str(step_num)}", expanded=False):
                        for trace in trace_steps[trace_id]:
                            trace_str = json.dumps(trace, indent=2)
                            st.code(trace_str, language="json", line_numbers=True)
                    step_num += 1
        if not has_trace:
            st.text("None")

    # Display citations
    st.subheader("Citations")
    if len(st.session_state.citations) > 0:
        citation_num = 1
        for citation in st.session_state.citations:
            for retrieved_ref_num, retrieved_ref in enumerate(citation["retrievedReferences"]):
                with st.expander(f"Citation [{str(citation_num)}]", expanded=False):
                    citation_str = json.dumps(
                        {
                            "generatedResponsePart": citation["generatedResponsePart"],
                            "retrievedReference": citation["retrievedReferences"][retrieved_ref_num]
                        },
                        indent=2
                    )
                    st.code(citation_str, language="json", line_numbers=True)
                citation_num = citation_num + 1
    else:
        st.text("None")

