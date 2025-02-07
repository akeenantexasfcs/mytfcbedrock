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
   st.session_state.session_id = str(uuid.uuid4())
   st.session_state.messages = []
   st.session_state.citations = []
   st.session_state.trace = {}

def process_agent_response(event_stream):
   """Process the event stream from Bedrock Agent Runtime"""
   full_response = ""
   citations = []
   trace = {}
   
   try:
       for event in event_stream:
           if hasattr(event, 'get'):  # Check if event is a dict-like object
               chunk = event.get('chunk', {})
               if chunk and hasattr(chunk.get('bytes', b''), 'decode'):
                   chunk_data = chunk['bytes'].decode('utf-8')
                   try:
                       chunk_json = json.loads(chunk_data)
                       if 'completion' in chunk_json:
                           full_response += chunk_json['completion']
                       if 'citations' in chunk_json:
                           citations.extend(chunk_json['citations'])
                       if 'trace' in chunk_json:
                           trace.update(chunk_json['trace'])
                   except json.JSONDecodeError as e:
                       logger.warning(f"Failed to parse chunk as JSON: {e}")
                       full_response += chunk_data
           else:  # Handle the case where event might be the response itself
               try:
                   if hasattr(event, 'decode'):
                       full_response += event.decode('utf-8')
                   elif isinstance(event, str):
                       full_response += event
               except Exception as e:
                   logger.warning(f"Failed to process event: {e}")
                   
   except Exception as e:
       logger.error(f"Error processing event stream: {e}")
       raise

   return full_response, citations, trace

# General page configuration and initialization
st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)
if len(st.session_state.items()) == 0:
   init_session_state()

# Sidebar button to reset session state
with st.sidebar:
   if st.button("Reset Session"):
       init_session_state()

# Messages in the conversation
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"], unsafe_allow_html=True)

# Chat input that invokes the agent
if prompt := st.chat_input():
   st.session_state.messages.append({"role": "user", "content": prompt})
   with st.chat_message("user"):
       st.write(prompt)

   with st.chat_message("assistant"):
       with st.empty():
           output_text = ""
           citations = []
           trace = {}

           with st.spinner():
               try:
                   # Log the request parameters for debugging
                   logger.debug(f"Invoking agent with parameters: agentId={agent_id}, agentAliasId={agent_alias_id}")
                   
                   response = bedrock_agent_runtime.invoke_agent(
                       agentId=agent_id,
                       agentAliasId=agent_alias_id,
                       sessionId=st.session_state.session_id,
                       inputText=prompt
                   )
                   
                   # Process the response directly
                   output_text, citations, trace = process_agent_response(response)
                   
               except ClientError as e:
                   error_code = e.response['Error']['Code']
                   error_message = e.response['Error']['Message']
                   logger.error(f"AWS API Error: {error_code} - {error_message}")
                   output_text = f"I encountered an AWS service error. Please try again later. Error: {error_code}"
               
               except json.JSONDecodeError as e:
                   logger.error(f"JSON parsing error: {e}")
                   output_text = "I had trouble processing the response. Please try again."
               
               except Exception as e:
                   logger.error(f"Unexpected error: {str(e)}")
                   output_text = "I apologize, but I encountered an unexpected error. Please try again."

           # Add citations if available
           if citations:
               citation_num = 1
               output_text = re.sub(r"%\[(\d+)\]%", r"<sup>[\1]</sup>", output_text)
               citation_locs = ""
               for citation in citations:
                   for retrieved_ref in citation["retrievedReferences"]:
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

           st.session_state.messages.append({"role": "assistant", "content": output_text})
           st.session_state.citations = citations
           st.session_state.trace = trace
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

# Sidebar section for trace
with st.sidebar:
   st.title("Trace")

   # Show each trace type in separate sections
   step_num = 1
   for trace_type_header in trace_types_map:
       st.subheader(trace_type_header)

       # Organize traces by step similar to how it is shown in the Bedrock console
       has_trace = False
       for trace_type in trace_types_map[trace_type_header]:
           if trace_type in st.session_state.trace:
               has_trace = True
               trace_steps = {}

               for trace in st.session_state.trace[trace_type]:
                   # Each trace type and step may have different information for the end-to-end flow
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

               # Show trace steps in JSON similar to the Bedrock console
               for trace_id in trace_steps.keys():
                   with st.expander(f"Trace Step {str(step_num)}", expanded=False):
                       for trace in trace_steps[trace_id]:
                           trace_str = json.dumps(trace, indent=2)
                           st.code(trace_str, language="json", line_numbers=True, wrap_lines=True)
                   step_num += 1
       if not has_trace:
           st.text("None")

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
                   st.code(citation_str, language="json", line_numbers=True, wrap_lines=True)
               citation_num = citation_num + 1
   else:
       st.text("None")

