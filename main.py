import os
import re
import time
import traceback
import requests
import socks
import socket
from datetime import datetime
from dotenv import load_dotenv
from arcadepy import Arcade
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import json
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import DataType
from weaviate.classes.config import Property
from weaviate.classes.init import AdditionalConfig, Timeout

# Load environment variables
load_dotenv()

# Proxy configuration
PROXY_ENABLED = os.environ.get("PROXY_ENABLED", "false").lower() == "true"  # Disabled by default for local use
PROXY_TYPE = os.environ.get("PROXY_TYPE", "tor")  # tor, rotating, or direct
PROXY_URL = os.environ.get("PROXY_URL")
TOR_PROXY = "socks5://127.0.0.1:9050"
PROXY_ROTATION_INTERVAL = int(os.environ.get("PROXY_ROTATION_INTERVAL", "120"))  # seconds
last_proxy_rotation = time.time()

# User configuration from environment variables
USER_ID = os.environ.get("USER_ID", "")
CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "")
TARGET_USERNAME = os.environ.get("TARGET_USERNAME", "")

def setup_proxy():
    """Configure proxy based on environment settings"""
    global last_proxy_rotation
    
    if not PROXY_ENABLED:
        print("Proxy disabled. Using direct connection.")
        return False
    
    try:
        if PROXY_TYPE.lower() == "tor":
            print("Setting up Tor proxy")
            proxy_parts = TOR_PROXY.replace("://", ":").split(":")
            if len(proxy_parts) >= 3:
                proxy_type = proxy_parts[0]
                proxy_host = proxy_parts[1]
                proxy_port = int(proxy_parts[2])
                
                # Configure proxy
                if proxy_type.lower() == "socks5":
                    socks.set_default_proxy(socks.SOCKS5, proxy_host, proxy_port)
                    socket.socket = socks.socksocket
                    print(f"SOCKS5 proxy configured: {proxy_host}:{proxy_port}")
                    
                    # Request a new Tor circuit
                    try:
                        requests.get("https://check.torproject.org/", timeout=5)
                        print("Successfully connected through Tor")
                    except Exception as e:
                        print(f"Error testing Tor connection: {e}")
                        
                    last_proxy_rotation = time.time()
                    return True
        
        elif PROXY_TYPE.lower() == "rotating" and PROXY_URL:
            print(f"Setting up rotating proxy: {PROXY_URL}")
            proxy_parts = PROXY_URL.replace("://", ":").split(":")
            if len(proxy_parts) >= 3:
                proxy_type = proxy_parts[0]
                proxy_host = proxy_parts[1]
                proxy_port = int(proxy_parts[2])
                
                # Configure proxy
                if proxy_type.lower() == "socks5":
                    socks.set_default_proxy(socks.SOCKS5, proxy_host, proxy_port)
                    socket.socket = socks.socksocket
                    print(f"SOCKS5 proxy configured: {proxy_host}:{proxy_port}")
                elif proxy_type.lower() == "socks4":
                    socks.set_default_proxy(socks.SOCKS4, proxy_host, proxy_port)
                    socket.socket = socks.socksocket
                    print(f"SOCKS4 proxy configured: {proxy_host}:{proxy_port}")
                elif proxy_type.lower() in ["http", "https"]:
                    os.environ["HTTP_PROXY"] = PROXY_URL
                    os.environ["HTTPS_PROXY"] = PROXY_URL
                    print(f"HTTP/HTTPS proxy configured: {PROXY_URL}")
                else:
                    print(f"Unsupported proxy type: {proxy_type}")
                    return False
                
                last_proxy_rotation = time.time()
                return True
        
        return False
    except Exception as e:
        print(f"Error setting up proxy: {e}")
        traceback.print_exc()
        return False

def rotate_proxy_if_needed():
    """Rotate proxy if the rotation interval has passed"""
    global last_proxy_rotation
    
    if not PROXY_ENABLED:
        return
    
    current_time = time.time()
    if current_time - last_proxy_rotation > PROXY_ROTATION_INTERVAL:
        print("Rotating proxy connection...")
        
        # Reset socket to default
        socket.socket = socket._socketobject
        
        # Set up proxy again
        setup_proxy()
        
        # Update rotation timestamp
        last_proxy_rotation = current_time

# Initialize proxy
if PROXY_ENABLED:
    setup_proxy()

# Initialize the Arcade client
client = Arcade(
    api_key=os.environ.get("ARCADE_API_KEY")
)

# Initialize OpenAI client
openai_client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Initialize Weaviate client if credentials are provided
weaviate_enabled = False
weaviate_client = None

if os.environ.get("WEAVIATE_URL") and os.environ.get("WEAVIATE_API_KEY"):
    try:
        print("Initializing Weaviate client...")
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.environ.get("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.environ.get("WEAVIATE_API_KEY")),
            headers={
                "X-OpenAI-Api-Key": os.environ.get("OPENAI_API_KEY")
            },
            additional_config=AdditionalConfig(
                timeout=Timeout(query=60)  # Increase query timeout to 60 seconds
            ),
            skip_init_checks=True
        )
        weaviate_enabled = True
        print("Weaviate client initialized successfully")
    except Exception as e:
        print(f"Error initializing Weaviate client: {e}")
        traceback.print_exc()
        print("Continuing without Weaviate integration...")
else:
    print("Weaviate credentials not provided. Summaries will not be stored for searching.")

# Define YouTube URL pattern - updated to handle Slack's URL formatting
YOUTUBE_URL_PATTERN = r'<?(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})(&amp;[^>]*)?'

def setup_weaviate_collection():
    """Set up the Weaviate collection for storing YouTube summaries."""
    if not weaviate_enabled:
        print("Weaviate integration is disabled. Skipping collection setup.")
        return False
        
    try:
        # Check if collection already exists
        if not weaviate_client.collections.exists("YouTubeSummary"):
            print("Attempting to create YouTubeSummary collection...")
            # Create collection with appropriate schema
            summaries = weaviate_client.collections.create(
                name="YouTubeSummary",
                properties=[
                    Property(name="video_id", data_type=DataType.TEXT),
                    Property(name="video_url", data_type=DataType.TEXT),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.DATE)
                ],
                vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_openai(),
                generative_config=weaviate.classes.config.Configure.Generative.openai()
            )
            print("Created YouTubeSummary collection in Weaviate")
            return True
        else:
            print("YouTubeSummary collection already exists in Weaviate")
            return True
            
    except Exception as e:
        print(f"Error setting up Weaviate collection: {e}")
        traceback.print_exc()
        print("Continuing without Weaviate integration...")
        return False

def store_summary_in_weaviate(video_id, video_url, summary):
    """Store the video summary in Weaviate."""
    if not weaviate_enabled:
        print("Weaviate integration is disabled. Skipping summary storage.")
        return False
        
    try:
        # Get the collection
        summaries = weaviate_client.collections.get("YouTubeSummary")
        
        # Format the date in RFC3339 format (removing microseconds)
        created_at = datetime.now().replace(microsecond=0).isoformat() + "Z"
        
        # Create a data object
        summaries.data.insert({
            "video_id": video_id,
            "video_url": video_url,
            "summary": summary,
            "created_at": created_at
        })
        
        print(f"Stored summary for video {video_id} in Weaviate")
        return True
    except Exception as e:
        print(f"Error storing summary in Weaviate: {e}")
        traceback.print_exc()
        print("Failed to store summary in Weaviate")
        return False

def query_summaries(query_text):
    """Query the stored summaries using Weaviate's Query Agent with enhanced citations.
    
    Provides richer, more readable citations in Slack format with proper linking.
    
    Args:
        query_text (str): The natural language query to search for
        
    Returns:
        str: The formatted answer with properly attributed sources
    """
    if not weaviate_enabled:
        return "Sorry, the Weaviate integration is currently disabled. Summaries are not being stored for searching."
        
    try:
        # Import here to avoid dependency issues if the package is not installed
        from weaviate_agents.query import QueryAgent
        from weaviate_agents.utils import print_query_agent_response
        
        print(f"Querying Weaviate with: '{query_text}'")
        
        # Initialize the query agent with our YouTubeSummary collection
        query_agent = QueryAgent(
            client=weaviate_client,
            collections=["YouTubeSummary"]
        )
        
        # Run the query and get the response
        response = query_agent.run(query_text)
        
        # Print detailed response for debugging
        print("Query Agent Response:")
        print(f"- Original Query: {response.original_query}")
        print(f"- Collections Used: {response.collection_names}")
        print(f"- Total Time: {response.total_time:.2f} seconds")
        print(f"- Token Usage: {response.usage.total_tokens} tokens")
        
        # Add the sources to the final answer
        final_answer = response.final_answer
        
        # Add sources/citations if they exist
        if hasattr(response, 'sources') and response.sources:
            final_answer += "\n\n*Sources:*"
            for i, source in enumerate(response.sources, 1):
                # Get the object_id from the source
                if hasattr(source, 'object_id'):
                    object_id = source.object_id
                    
                    # Try to get the object from Weaviate using the object_id
                    try:
                        summaries = weaviate_client.collections.get("YouTubeSummary")
                        result = summaries.query.fetch_objects(
                            limit=1,
                            filters=weaviate.classes.query.Filter.by_id().equal(object_id)
                        )
                        
                        if result.objects and len(result.objects) > 0:
                            obj = result.objects[0]
                            props = obj.properties
                            video_id = props.get('video_id', 'Unknown')
                            video_url = props.get('video_url', 'Unknown URL')
                            
                            # Try to get video title - might exist in some records
                            video_title = props.get('title', '')
                            
                            # If we don't have a title stored, use the video_id to create a descriptive title
                            if not video_title or video_title == 'Unknown Title':
                                video_title = f"YouTube Video ({video_id})"
                            
                            # Format the citation for Slack with proper linking
                            if video_url and video_url != 'Unknown URL':
                                # Create a well-formatted Slack link with available information
                                final_answer += f"\n{i}. <{video_url}|{video_title}>"
                                
                                # If we have creation date information, add it
                                if 'created_at' in props:
                                    try:
                                        # Format the date nicely if it exists
                                        from datetime import datetime
                                        date_obj = datetime.fromisoformat(props['created_at'].replace('Z', '+00:00'))
                                        date_str = date_obj.strftime('%Y-%m-%d')
                                        final_answer += f" (indexed on {date_str})"
                                    except:
                                        # If date parsing fails, continue without the date
                                        pass
                            else:
                                # Fallback if URL is missing but we have a video ID
                                if video_id and video_id != 'Unknown':
                                    reconstructed_url = f"https://www.youtube.com/watch?v={video_id}"
                                    final_answer += f"\n{i}. <{reconstructed_url}|{video_title}>"
                                else:
                                    final_answer += f"\n{i}. Source ID: {object_id}"
                        else:
                            # Fallback if object not found
                            final_answer += f"\n{i}. Source ID: {object_id}"
                    except Exception as e:
                        print(f"Error retrieving source from Weaviate: {e}")
                        traceback.print_exc()
                        final_answer += f"\n{i}. Source ID: {object_id}"
                else:
                    final_answer += f"\n{i}. Unknown source"
        
        return final_answer
    except ImportError:
        print("weaviate-agents package not installed. Please install with: pip install weaviate-agents")
        return "Error: The weaviate-agents package is not installed."
    except Exception as e:
        print(f"Error querying summaries: {e}")
        traceback.print_exc()
        return f"Sorry, I encountered an error while searching: {str(e)}"

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    match = re.search(YOUTUBE_URL_PATTERN, url)
    if match:
        return match.group(4)
    return None

def format_transcript(transcript_list):
    """Format transcript list into a single text string."""
    return " ".join([item['text'] for item in transcript_list])

def get_youtube_transcript(video_id):
    """Get transcript for a YouTube video."""
    try:
        print(f"Attempting to get transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        print(f"Successfully retrieved transcript with default language")
        return format_transcript(transcript_list)
    except Exception as e:
        print(f"Error getting transcript: {e}")
        traceback.print_exc()
        return None

def summarize_transcript(transcript):
    """Summarize the transcript using OpenAI."""
    try:
        response = openai_client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": """You are an AI expert that creates comprehensive, detailed summaries of AI interview transcripts, emphasizing all technical discussions and insights. Your objective is to extract and elaborate on all significant AI-related topics with technical depth and precision.

Summary Requirements:
- Extract and elaborate on all significant AI-related topics discussed in the interview
- Capture nuances of technical discussions across all mentioned AI areas
- Provide in-depth analysis of all methodologies and concepts mentioned
- Highlight implications, applications, and innovative aspects for each topic
- Include challenges, hurdles, solutions, and future directions in AI implementation
- Reflect both depth and breadth of the conversation across all AI topics
- Serve as a deep dive into the technical realm of AI, covering all points raised
- Provide detailed explanations, not just summaries
- Aim for a comprehensive, lengthy analysis that covers all technical aspects discussed

Metrics Section:
If the transcript is with a founder who provides metrics about revenue levels, total customers, retention rates, or growth rates, include a dedicated metrics section with this data.

Depth of Analysis:
Go deep on each technical point, regardless of the specific AI area. For each topic (e.g., model architectures, learning approaches, agents, agent frameworks, RAG, deployment challenges, voice models, multimodality):
- Explain core concepts and their significance
- Discuss any novel approaches or innovations mentioned
- Elaborate on challenges and why they arise
- Detail proposed solutions and their complexity
- Explore implications and potential future developments

Technical Areas to Consider (but not limited to):
- AI model architectures (transformer and non-transformer)
- Machine learning approaches (supervised, unsupervised, reinforcement learning, etc.)
- Natural Language Processing techniques
- AI Agents and frameworks
- Data trends, challenges, processing and transformation
- Tools and UX/UI experience
- Computer Vision applications
- AI deployment in various environments
- Reinforcement learning
- RAG (Retrieval Augmented Generation)
- Scaling laws - challenges and opportunities
- Knowledge graphs
- Evaluations
- AI hardware and infrastructure
- AI integration with other technologies
- Voice model challenges and opportunities
- Multimodality

Important Notes:
- Cover ALL AI-related topics mentioned in the interview, not just predetermined focus areas
- Provide detailed insights and discussions, not just high-level statements
- Focus on technical content, not the interviewee's personal background unless directly relevant
- Elaborate on each mentioned topic with second and third-level insights
- Cater to an audience of AI researchers looking for advanced information
- For topics like RAG, embeddings, agents, etc., provide equal depth of coverage
- If specific tools, frameworks, or methodologies are mentioned, provide context and technical details

FORMATTING INSTRUCTIONS:
- Use Slack-friendly formatting: *bold* for emphasis, _italics_ for secondary emphasis
- For section headers, use *SECTION NAME* in bold instead of markdown headers (###)
- For subsection headers, use _SUBSECTION NAME_ in italics instead of markdown subheaders (####)
- Use numbered lists with proper spacing (1., 2., 3.) instead of markdown numbered lists
- Use bullet points with proper spacing (•) instead of markdown bullet points
- Do not use markdown formatting like ###, ####, or other syntax that doesn't render well in Slack
- Ensure proper spacing between sections for readability

Example Approach:
Instead of: "They discussed machine learning models"
Provide: Detailed insights into the specific models mentioned, their architectures, novel features, training approaches, performance metrics, challenges in implementation, and potential advancements or applications. Include any comparisons made to other models or techniques, and elaborate on the technical reasoning behind preferences or criticisms expressed.

Your summary should not simply state that topics were discussed but should provide comprehensive details about those topics. For example, don't just mention that embeddings were discussed as a way to increase accuracy; explain the embedding process, the specific techniques used, and the technical reasons why they increase accuracy.

Also mention any insights, metrics, or statistics shared about AI. Remember that your audience consists of AI experts, so provide deep technical insights."""},
                {"role": "user", "content": f"Please create a comprehensive technical summary of this AI interview transcript, using Slack-friendly formatting (bold with * for headers, italics with _ for subheaders, and proper spacing for readability):\n\n{transcript}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        return f"Error generating summary: {str(e)}"

def send_dm_to_user(user_id, username, message):
    """Send a direct message to a user."""
    try:
        print(f"Attempting to send DM to {username}...")
        
        # Use the Arcade tools interface to send a DM
        response = client.tools.execute(
            tool_name="Slack.SendDmToUser@0.4.0",
            input={
                "user_name": username,  # Changed from 'username' to 'user_name'
                "message": message
            },
            user_id=user_id,
        )
        
        # Check if the response was successful
        if response and response.output and response.output.value:
            print(f"Successfully sent DM to {username}")
            return True
        else:
            print(f"Error sending message: Response did not contain expected data")
            return False
            
    except Exception as e:
        print(f"Error sending DM: {e}")
        traceback.print_exc()
        return False

def handle_query_command(user_id, channel_name, message_text, target_username=TARGET_USERNAME):
    """Handle a query command from Slack (!query)."""
    try:
        # Remove any Slack formatting (code blocks, bullets, etc.)
        clean_text = message_text.replace("", "").replace("•", "").strip()
        
        # Check if the message is a query command
        if clean_text.startswith("!query "):
            query = clean_text[7:].strip()  # Remove the !query prefix
            
            if query:
                print(f"Processing query: {query}")
                
                # Query the summaries
                result = query_summaries(query)
                
                print(f"Query result obtained. Sending to {target_username}...")
                
                # Send the result as a DM
                message = f"*Query Results:*\n\n{result}"
                success = send_dm_to_user(user_id, target_username, message)
                
                if success:
                    print(f"Query results successfully sent to {target_username}")
                else:
                    print(f"Failed to send query results to {target_username}")
                
                return True
            else:
                print("Empty query received, ignoring")
        return False
    except Exception as e:
        print(f"Error handling query command: {e}")
        traceback.print_exc()
        return False

def process_youtube_url(user_id, channel_name, message_text, original_url=None, target_username=TARGET_USERNAME):
    """Process a message with a YouTube URL."""
    try:
        print("\n==================================================")
        print("Starting to process a message with a YouTube URL")
        print("==================================================")
        
        # Use the provided URL if available, otherwise extract from message
        if original_url:
            url = original_url
            print(f"Using provided URL: {url}")
        else:
            # Extract URL from message
            match = re.search(YOUTUBE_URL_PATTERN, message_text)
            if match:
                url = match.group(0).replace("<", "").replace(">", "").split("&amp;")[0]
                print(f"Extracted URL from message: {url}")
            else:
                print("No YouTube URL found in message")
                return
        
        # Extract video ID
        video_id = extract_video_id(url)
        if not video_id:
            print("Could not extract video ID from URL")
            send_dm_to_user(user_id, target_username, f"Could not extract video ID from the URL: {url}")
            return
            
        print(f"Processing YouTube video ID: {video_id}")
        
        # IMPORTANT: Get transcript FIRST and complete the entire process before sending any messages
        print("Retrieving transcript...")
        
        # Get the transcript
        transcript = None
        transcript_error = None
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            print("Successfully retrieved transcript with default language")
            transcript = format_transcript(transcript_list)
        except Exception as e:
            print(f"Failed to retrieve transcript: {e}")
            transcript_error = str(e)
        
        # Check if we have a transcript
        if not transcript:
            error_message = f"Could not retrieve transcript for the YouTube video: {url}\n\nThis is likely because the video doesn't have subtitles/captions enabled, or they're in a language not supported by the API. Please try a different video with captions enabled."
            print(f"Failed to retrieve transcript. Error: {transcript_error}")
            send_dm_to_user(user_id, target_username, error_message)
            return
        
        print(f"Successfully retrieved transcript with {len(transcript)} characters")
        
        # Now that we have a transcript, generate the summary
        print("Summarizing transcript...")
        summary = None
        summary_error = None
        
        try:
            summary = summarize_transcript(transcript)
        except Exception as e:
            print(f"Error summarizing transcript: {e}")
            summary_error = str(e)
        
        if not summary:
            error_message = f"Could not generate summary for the YouTube video: {url}"
            if summary_error:
                error_message += f"\n\nError: {summary_error}"
            print("Failed to summarize transcript")
            send_dm_to_user(user_id, target_username, error_message)
            return
            
        print(f"Successfully generated summary with {len(summary)} characters")
        
        # Now that we have both transcript and summary, proceed with sending to Slack and storing in Weaviate
        
        # 1. Send the summary to Slack
        message = f"*Summary of YouTube Video:*\n\n{summary}\n\n*Original URL:* {url}"
        print("Sending summary to Slack...")
        slack_result = send_dm_to_user(user_id, target_username, message)
        
        if slack_result:
            print("Successfully sent summary to Slack")
        else:
            print("Failed to send summary to Slack")
        
        # 2. Store the summary in Weaviate
        weaviate_result = False
        try:
            if weaviate_enabled:
                print("Storing summary in Weaviate...")
                weaviate_result = store_summary_in_weaviate(video_id, url, summary)
                if weaviate_result:
                    print("Summary successfully stored in Weaviate")
                else:
                    print("Failed to store summary in Weaviate")
            else:
                print("Weaviate integration is disabled. Skipping summary storage.")
        except Exception as e:
            print(f"Error storing summary in Weaviate: {e}")
            traceback.print_exc()
        
        print("YouTube video processing complete!")
        print("==================================================")
        
        return {
            "success": True,
            "slack_result": slack_result,
            "weaviate_result": weaviate_result,
            "video_id": video_id,
            "url": url
        }
        
    except Exception as e:
        print(f"Error processing YouTube URL: {e}")
        traceback.print_exc()
        try:
            send_dm_to_user(user_id, target_username, f"An error occurred while processing the YouTube video: {str(e)}")
        except:
            print("Failed to send error message to Slack")
        return {
            "success": False,
            "error": str(e)
        }

def listen_to_slack_messages(user_id, channel_name, target_username=TARGET_USERNAME):
    """Listen to Slack messages in a specific channel and process YouTube URLs."""
    print(f"Starting to monitor Slack channel #{channel_name} for YouTube URLs")
    print(f"Summaries will be sent as DMs to {target_username}")
    
    # Setup Weaviate collection
    try:
        setup_weaviate_collection()
    except Exception as e:
        print(f"Error setting up Weaviate: {e}")
        traceback.print_exc()
        print("Continuing without Weaviate integration...")
    
    # Get authorization for Slack
    # First authorize for channel metadata
    print("Authorizing Slack access for GetChannelMetadataByName...")
    auth_response = client.tools.authorize(
        tool_name="Slack.GetChannelMetadataByName@0.4.0",
        user_id=user_id,
    )
    
    # Then authorize for messages
    print("Authorizing Slack access for GetMessagesInChannelByName...")
    auth_response = client.tools.authorize(
        tool_name="Slack.GetMessagesInChannelByName@0.4.0",
        user_id=user_id,
    )
    
    # Authorize for sending DMs
    print("Authorizing Slack access for SendDmToUser...")
    auth_response = client.tools.authorize(
        tool_name="Slack.SendDmToUser@0.4.0",
        user_id=user_id,
    )
    
    # Get channel metadata
    channel_response = client.tools.execute(
        tool_name="Slack.GetChannelMetadataByName@0.4.0",
        input={
            "channel_name": channel_name,
        },
        user_id=user_id,
    )
    
    channel_id = channel_response.output.value.get("id")
    print(f"Found channel #{channel_name} with ID {channel_id}")
    
    print(f"Starting to monitor channel #{channel_name} for YouTube URLs")
    print("Waiting for messages... (Press Ctrl+C to stop)")
    
    # Send a test DM to the target user
    message = "YouTube Transcript Bot is now running and monitoring the #youtube channel for URLs and query commands."
    send_dm_to_user(user_id, target_username, message)
    
    # Keep track of processed messages to avoid duplicates
    processed_messages = set()
    
    # Record the start time of the app
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    start_time_epoch = start_time.timestamp()  # Convert to epoch timestamp for comparison with Slack ts
    print(f"App started at {start_time_str} (epoch: {start_time_epoch}). Only processing messages after this time.")
    
    # Create a dictionary to track message content to prevent duplicates with different timestamps
    message_content_tracker = {}
    
    while True:
        try:
            print("Checking for new messages...")
            # Get recent messages in the channel using oldest_relative to ensure we get new messages
            messages_response = client.tools.execute(
                tool_name="Slack.GetMessagesInChannelByName@0.4.0",
                input={
                    "channel_name": channel_name,
                    "limit": 10,
                    "oldest_relative": "0:0:5"  # Last 5 minutes
                },
                user_id=user_id,
            )
            
            # Debug response type
            response_data = messages_response.output.value
            print(f"Response data type: {type(response_data)}")
            
            if response_data is None:
                print("No messages returned from Slack API")
                messages = []
            elif isinstance(response_data, dict):
                print(f"Response keys: {list(response_data.keys())}")
                messages = response_data.get("messages", [])
            else:
                messages = response_data
            
            print(f"Retrieved {len(messages)} messages from channel")
            
            # Process each message
            for message in messages:
                if not isinstance(message, dict):
                    continue
                
                message_id = message.get("ts", "")
                message_text = message.get("text", "")
                user_id_from_message = message.get("user", "")
                
                # Skip if no ts value
                if not message_id:
                    continue
                
                # Convert Slack ts to float for timestamp comparison
                try:
                    message_ts_float = float(message_id)
                except ValueError:
                    print(f"Could not convert ts to float: {message_id}")
                    continue
                
                # Skip messages from before the app started using ts as source of truth
                if message_ts_float < start_time_epoch:
                    print(f"Skipping old message with ts {message_id} (before app start time)")
                    processed_messages.add(message_id)  # Add to processed to avoid checking again
                    continue
                
                # Create a unique content identifier (user + text) to detect duplicates even with different ts
                content_key = f"{user_id_from_message}:{message_text}"
                
                # Skip if we've already processed this exact content
                if content_key in message_content_tracker:
                    print(f"Skipping duplicate content (different ts): {message_text[:50]}...")
                    continue
                
                # Skip if already processed by ts
                if message_id in processed_messages:
                    continue
                
                # Track this message's content
                message_content_tracker[content_key] = message_id
                
                # Add to processed messages
                processed_messages.add(message_id)
                
                print(f"Processing message: {message_text[:100]}...")
                
                # Check if it's a query command
                clean_text = message_text.replace("", "").replace("•", "").strip()
                if "!query" in clean_text:
                    print(f"Found potential query command: {clean_text}")
                    if handle_query_command(user_id, channel_name, message_text):
                        continue
                
                # Check if message contains a YouTube URL
                if re.search(YOUTUBE_URL_PATTERN, message_text):
                    print(f"Found YouTube URL in message: {message_text[:50]}...")
                    
                    # Extract the URL
                    match = re.search(YOUTUBE_URL_PATTERN, message_text)
                    if match:
                        url = match.group(0).replace("<", "").replace(">", "").split("&amp;")[0]
                        
                        # Process the URL
                        process_youtube_url(user_id, channel_name, message_text, url)
            
            # Limit the size of our trackers to prevent memory growth
            if len(processed_messages) > 1000:
                # Keep only the 500 most recent messages
                processed_messages = set(list(processed_messages)[-500:])
            
            if len(message_content_tracker) > 1000:
                # Convert to list of tuples, sort by values (timestamps), and keep most recent 500
                sorted_items = sorted(message_content_tracker.items(), key=lambda x: x[1])
                message_content_tracker = dict(sorted_items[-500:])
            
            # Sleep for a bit before checking again
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"Sleeping for 10 seconds before checking again... (Current time: {current_time})")
            time.sleep(10)
            
        except Exception as e:
            print(f"Error checking messages: {e}")
            traceback.print_exc()
            time.sleep(10)  # Sleep a bit before retrying

def test_youtube_connectivity():
    """Test connectivity to YouTube and print diagnostic information."""
    try:
        print("\n--- Testing YouTube Connectivity ---")
        
        # Test basic connectivity to YouTube
        response = requests.get("https://www.youtube.com", timeout=10)
        print(f"YouTube connectivity: {response.status_code} {response.reason}")
        
        # Get and display IP information
        try:
            ip_response = requests.get("https://api.ipify.org?format=json", timeout=5)
            ip_data = ip_response.json()
            print(f"Current IP address: {ip_data.get('ip', 'Unknown')}")
        except Exception as e:
            print(f"Could not determine IP address: {e}")
        
        # Test a known video with transcripts
        test_video_id = "dQw4w9WgXcQ"  # A popular video known to have transcripts
        try:
            # Try to fetch transcript with default language
            transcript_list = YouTubeTranscriptApi.get_transcript(test_video_id)
            print(f"Successfully fetched test transcript with {len(transcript_list)} entries")
        except Exception as e:
            print(f"Error testing transcript API: {e}")
            traceback.print_exc()
            
        print("--- End of YouTube Connectivity Test ---\n")
        return True
    except Exception as e:
        print(f"YouTube connectivity test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main application function."""
    # Load configuration from environment variables
    user_id = USER_ID
    channel_name = CHANNEL_NAME
    target_username = TARGET_USERNAME
    
    # Validate required environment variables
    missing_vars = []
    if not user_id:
        missing_vars.append("USER_ID")
    if not channel_name:
        missing_vars.append("CHANNEL_NAME")
    if not target_username:
        missing_vars.append("TARGET_USERNAME")
    
    if missing_vars:
        print("Error: The following required environment variables are missing:")
        for var in missing_vars:
            print(f" - {var}")
        print("\nPlease add these variables to your .env file and try again.")
        return
    
    print("YouTube Transcript Summarizer for Slack")
    print("---------------------------------------")
    print(f"Will monitor the #{channel_name} channel for YouTube URLs")
    print(f"and send summaries as DMs to {target_username}")
    print("")
    
    # Test YouTube connectivity
    test_youtube_connectivity()
    
    print("Checking authorization status and channel access...")
    
    # Start listening for messages
    listen_to_slack_messages(user_id, channel_name, target_username)

if __name__ == "__main__":
    main()