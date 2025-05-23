document.getElementById('themeToggle').addEventListener('click', function() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
});

document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Initialize voice recording functionality
    initVoiceRecording();
});

// Voice recording variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Initialize voice recording functionality
function initVoiceRecording() {
    const voiceButton = document.getElementById('voiceButton');
    
    voiceButton.addEventListener('click', function() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });
}

// Start recording audio
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioToServer(audioBlob);
            audioChunks = [];
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        const voiceButton = document.getElementById('voiceButton');
        voiceButton.classList.add('recording');
        voiceButton.title = 'Stop recording';
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access your microphone. Please check permissions and try again.');
    }
}

// Stop recording audio
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        const voiceButton = document.getElementById('voiceButton');
        voiceButton.classList.remove('recording');
        voiceButton.title = 'Record voice';
        
        // Stop all tracks
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
}

// Send audio to server for transcription
async function sendAudioToServer(audioBlob) {
    try {
        // Show loading indicator
        const userInput = document.getElementById('userInput');
        userInput.value = 'Transcribing...';
        userInput.disabled = true;
        
        const formData = new FormData();
        formData.append('audio', audioBlob);
        
        const response = await fetch('http://localhost:5000/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Update input field with transcription
        userInput.value = data.transcription;
        userInput.disabled = false;
        
    } catch (error) {
        console.error('Transcription error:', error);
        alert('Error transcribing audio. Please try again or type your message.');
        
        // Reset input field
        const userInput = document.getElementById('userInput');
        userInput.value = '';
        userInput.disabled = false;
    }
}

async function getRoast() {
    const userInput = document.getElementById('userInput').value;
    const modelChoice = document.getElementById('modelSelect').value;
    if (!userInput) return;

    // Add user message to chat with right alignment
    const chatContainer = document.getElementById('chatContainer');
    const userMessageDiv = document.createElement('div');
    userMessageDiv.className = 'message user';  // Add 'user' class for right alignment
    userMessageDiv.innerHTML = `
        <div class="message-content">
            ${userInput}
        </div>
    `;
    chatContainer.appendChild(userMessageDiv);

    // Clear input and disable controls
    document.getElementById('userInput').value = '';
    document.getElementById('userInput').disabled = true;
    document.getElementById('modelSelect').disabled = true;
    const buttons = document.getElementsByTagName('button');
    for (let button of buttons) {
        button.disabled = true;
    }

    try {
        const response = await fetch('http://localhost:5000/roast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text: userInput,
                model: modelChoice 
            })
        });

        const data = await response.json();

        // Create bot message container with left alignment
        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'message bot';  // Add 'bot' class for left alignment
        
        // Add roast text
        const roastDiv = document.createElement('div');
        roastDiv.className = 'message-content';
        roastDiv.textContent = data.roast;
        botMessageDiv.appendChild(roastDiv);

        // Add GIF
        const gifContainer = document.createElement('div');
        gifContainer.className = 'gif-container';
        const gifImage = document.createElement('img');
        gifImage.src = data.gif;
        gifContainer.appendChild(gifImage);
        botMessageDiv.appendChild(gifContainer);

        // Add audio control button
        const audioControl = document.createElement('button');
        audioControl.className = 'audio-control';
        audioControl.innerHTML = '<span class="play-icon">🔊 Play</span><span class="pause-icon">⏸️ Pause</span>';
        audioControl.onclick = function() { toggleAudio(this); };
        botMessageDiv.appendChild(audioControl);

        // Add to chat
        chatContainer.appendChild(botMessageDiv);

        // Setup audio
        const audioPlayer = document.createElement('audio');
        audioPlayer.src = `data:audio/mpeg;base64,${data.audio}`;
        botMessageDiv.appendChild(audioPlayer);

        // Play audio automatically
        audioPlayer.play();
        audioControl.classList.add('playing');

        // Add event listener for when audio ends
        audioPlayer.onended = function() {
            audioControl.classList.remove('playing');
        };

        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;

    } catch (error) {
        console.error('Error:', error);
        alert('Something went wrong! Try again later.');
    }

    // Re-enable controls
    document.getElementById('userInput').disabled = false;
    document.getElementById('modelSelect').disabled = false;
    for (let button of buttons) {
        button.disabled = false;
    }
}

// Allow Enter key to trigger roast
document.getElementById('userInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        getRoast();
    }
});

// Add event listener for audio ended
document.getElementById('audioPlayer').addEventListener('ended', function() {
    document.getElementById('audioControl').classList.remove('playing');
});

// Update toggleAudio function to work with multiple audio players
function toggleAudio(button) {
    const audioPlayer = button.nextElementSibling; // The audio element is right after the button
    
    if (audioPlayer.paused || audioPlayer.ended) {
        audioPlayer.play();
        button.classList.add('playing');
    } else {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;  // Reset to beginning
        button.classList.remove('playing');
    }
}