# Sarcastic Bot

A fun web application that generates sarcastic roasts based on user input. The bot uses AI to create personalized, humorous roasts and pairs them with reaction GIFs and voice responses.

## Features

- **Text-based Roasts**: Enter text about yourself and get a personalized roast
- **Voice Input**: Record your voice and have it transcribed using the Whisper model
- **Reaction GIFs**: Each roast comes with a matching reaction GIF
- **Voice Responses**: Roasts are read aloud using ElevenLabs text-to-speech
- **Dark/Light Mode**: Toggle between dark and light themes
- **Model Selection**: Choose between different AI models for roasting

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   > **Note**: The speech recognition feature requires specific dependencies including TensorFlow and Keras. If you encounter any issues, make sure these are properly installed.

3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   GIPHY_API_KEY=your_giphy_api_key
   ELEVEN_API_KEY=your_elevenlabs_api_key
   ```
4. Run the application:
   ```
   python roast_bot.py
   ```
5. Open your browser and navigate to `http://localhost:5000`

## Speech Recognition

The application uses the Whisper model from Hugging Face for speech recognition. To use this feature:

1. Click the microphone button to start recording
2. Speak your message
3. Click the button again to stop recording
4. The transcribed text will appear in the input field
5. Click "Roast" to generate a response

### Troubleshooting Speech Recognition

If you encounter issues with the speech recognition feature:

1. Make sure you have installed all the required dependencies:
   ```
   pip install tensorflow==2.15.0 keras==2.15.1 transformers==4.36.2 torch==2.1.2 accelerate==0.25.0
   ```
2. Check the console output for any error messages related to the Whisper model
3. If you're still having issues, you can still use the application with text input only

## Technologies Used

- **Backend**: Flask, Python
- **AI Models**: Groq (LLM), Whisper (Speech Recognition), ElevenLabs (Text-to-Speech)
- **Frontend**: HTML, CSS, JavaScript
- **APIs**: Giphy API for reaction GIFs

## License

MIT 