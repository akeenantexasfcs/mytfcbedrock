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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
        credentials = {
            "aws_access_key_id": st.secrets["aws"]["access_key_id"].strip(),
            "aws_secret_access_key": st.secrets["aws"]["secret_access_key"].strip(),
            "region_name": st.secrets["aws"]["region"].strip(),
        }

        logger.info(f"AWS Region: {credentials['region_name']}")
        logger.info(f"Access Key ID length: {len(credentials['aws_access_key_id'])}")
        logger.info(f"Secret Key length: {len(credentials['aws_secret_access_key'])}")

        # Set environment variables
        os.environ["AWS_ACCESS_KEY_ID"] = credentials["aws_access_key_id"]
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["aws_secret_access_key"]
        os.environ["AWS_DEFAULT_REGION"] = credentials["region_name"]

        return credentials
    except Exception as e:
        logger.error(f"Error setting up AWS credentials: {str(e)}")
        st.error("Failed to configure AWS credentials. Please check your secrets configuration.")
        return None

def initialize_bedrock_clients(credentials):
    """Initialize Bedrock clients"""
    try:
        session = boto3.Session(
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            region_name=credentials["region_name"],
        )

        # Bedrock Runtime Client
        agent_runtime_client = session.client("bedrock-agent-runtime")

        # Bedrock Management Client (for listing agent aliases)
        bedrock_client = session.client("bedrock")

        logger.info("Bedrock clients initialized successfully")

        # Test the bedrock management client with a valid API call
        try:
            response = bedrock_client.list_agent_aliases(agentId=BEDROCK_AGENT_ID)
            logger.info("Successfully tested Bedrock agent alias retrieval")
        except ClientError as e:
            logger.error(f"Failed to list agent aliases: {e.response['Error']['Message']}")
            raise

        return agent_runtime_client  # Use runtime client for invoking the agent

    except Exception as e:
        logger.error(f"Error initializing Bedrock clients: {str(e)}")
        st.error("Failed to initialize Bedrock clients. Please check your AWS configuration.")
        return None

# Page setup
st.set_page_config(page_title=UI_TITLE, layout="wide")
st.title(UI_TITLE)

# Initialize session state if needed
if "session_id" not in st.session_state:
    init_session_state()

# Set up AWS credentials
credentials = setup_aws_credentials()
if not credentials:
    st.stop()

# Initialize Bedrock client
bedrock_client = initialize_bedrock_clients(credentials)
if not bedrock_client:
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
                        inputText=prompt,
                    )

                    # Extract response components
                    completion = response.get("completion", {})
                    output_text = completion.get("promptOutput", {}).get("text", "")
                    citations = completion.get("citations", [])
                    trace = completion.get("trace", {})

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
                                location_type = retrieved_ref["location"]["type"]

                                try:
                                    match location_type:
                                        case "CONFLUENCE":
                                            citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['confluenceLocation']['url']}"
                                        case "CUSTOM":
                                            citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['customDocumentLocation']['id']}"
                                        case "KENDRA":
                                            citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['kendraDocumentLocation']['uri']}"
                                        case "S3":
                                            citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['s3Location']['uri']}"
                                        case "WEB":
                                            citation_locs += f"\n<br>{citation_marker} {retrieved_ref['location']['webLocation']['url']}"
                                        case _:
                                            logger.warning(f"Unknown citation location type: {location_type}")
                                except Exception as e:
                                    logger.error(f"Error processing citation {citation_num}: {str(e)}")

                                citation_num += 1

            except ClientError as e:
                logger.error(f"AWS Error: {e.response['Error']['Message']}")
                st.error(f"AWS Error: {e.response['Error']['Message']}")
                output_text = "Sorry, there was an error processing your request."

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred: {str(e)}")
                output_text = "Sorry, there was an error processing your request."

            st.session_state.messages.append({"role": "assistant", "content": output_text})
            st.markdown(output_text, unsafe_allow_html=True)

