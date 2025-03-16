# YouTube Transcript Bot

A Slack integration that monitors channels for YouTube URLs, extracts and summarizes video transcripts using OpenAI, and stores them in a Weaviate vector database for future reference and querying. Specifically designed for generative AI podcasts and interviews with founders, operators, and researchers to extract technical insights.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/slack-integration-4A154B.svg" alt="Slack Integration">
  <img src="https://img.shields.io/badge/OpenAI-powered-412991.svg" alt="OpenAI Powered">
  <img src="https://img.shields.io/badge/Weaviate-vectorDB-FF6D01.svg" alt="Weaviate Vector Database">
  <img src="https://img.shields.io/badge/Arcade.dev-integration-6047ff.svg" alt="Arcade.dev Integration">
</p>

## Overview

YouTube Transcript Bot processes YouTube videos shared in Slack. When a team member posts a YouTube link in a monitored channel, the bot:

1. Extracts the video transcript
2. Generates a concise summary using OpenAI
3. Delivers the summary as a direct message to the specified user
4. Stores the summary in a Weaviate vector database for future semantic searching

## Key Features

- **Slack Channel Monitoring**: Watches specified Slack channels for YouTube URLs
- **Transcript Extraction**: Pulls transcripts from YouTube videos
- **AI-Powered Summarization**: Uses OpenAI to create summaries
- **Private Delivery**: Sends summaries as direct messages
- **Vector Database Storage**: Stores summaries in Weaviate for semantic searching
- **Query Functionality**: Supports natural language queries with `!query` command

## How It Works

The application uses:

- **[Arcade.dev](https://arcade.dev)** for seamless Slack integration without requiring a custom Slack app
- **YouTube Transcript API** for extracting video transcripts
- **OpenAI API** for generating summaries
- **[Weaviate](https://weaviate.io)** serverless cluster for vector database storage
- **[Weaviate Query Agent](https://weaviate.io/developers/weaviate/modules/reader-generator-modules/query-agent)** for efficient natural language querying of stored summaries

### Weaviate Query Agent

The bot leverages Weaviate's Query Agent to provide powerful natural language querying capabilities. This pre-built agentic service abstracts away the complexities of vector search, allowing users to query stored summaries using simple natural language.

When a user issues a `!query` command, the Weaviate Query Agent analyzes the query, determines the appropriate search strategy, and returns relevant summaries that best match the semantic intent.

### Arcade.dev Integration

The bot uses Arcade.dev to seamlessly integrate with Slack without requiring a custom Slack app. This integration provides secure user impersonation, independent scaling, and enterprise-ready security while significantly simplifying the development process.

## Installation

### Prerequisites

1. Python 3.8 or higher
2. An [Arcade.dev](https://arcade.dev) account with API key
3. An [OpenAI API key](https://platform.openai.com/)
4. A [Weaviate Cloud](https://weaviate.io/developers/weaviate/installation/weaviate-cloud-services) serverless cluster (free tier available)
5. A Slack workspace where you have permissions to add apps

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youtube-transcript-bot.git
   cd youtube-transcript-bot
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root with the following variables:
   ```
   # API Keys
   ARCADE_API_KEY=your_arcade_api_key
   OPENAI_API_KEY=your_openai_api_key
   WEAVIATE_URL=your_weaviate_cluster_url
   WEAVIATE_API_KEY=your_weaviate_api_key

   # User Configuration
   USER_ID=your_email@example.com
   CHANNEL_NAME=youtube
   TARGET_USERNAME=username_to_receive_summaries
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

   On first run, you'll need to authorize the application with Slack through the Arcade.dev API.

## Usage

1. Start the bot using `python main.py`
2. Share YouTube links in the configured Slack channel
3. The bot will process videos and send summaries as direct messages
4. Query past summaries with `!query [your question]` in the Slack channel

Example: `!query What are the best practices for React performance?`

## Troubleshooting

If the bot isn't working as expected:

- Ensure all API keys in your `.env` file are correct
- Verify the YouTube link has an available transcript
- Check that the channel name matches exactly (case-sensitive)
- Make sure your Weaviate cluster is running

---

<p align="center">
  MIT Licensed
</p>
