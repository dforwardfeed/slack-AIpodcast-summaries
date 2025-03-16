#!/bin/bash

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install arcadepy youtube-transcript-api openai python-dotenv

# Create .env file template
echo "Creating .env file template..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
ARCADE_API_KEY=your_arcade_api_key
OPENAI_API_KEY=your_openai_api_key
EOF
    echo ".env file created. Please edit it to add your API keys."
else
    echo ".env file already exists."
fi

# Login to Arcade
echo "Logging in to Arcade..."
arcade login

echo "Setup completed successfully!"
echo "Next steps:"
echo "1. Edit the .env file to add your API keys"
echo "2. Edit the USER_ID in main.py"
echo "3. Run the bot with: python main.py"