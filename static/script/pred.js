// Trigger file input dialog
function triggerFileInput() {
  document.getElementById('file-input').click();
}

// Handle file selection
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
      const reader = new FileReader();
      reader.onload = function(e) {
          document.getElementById('image-preview').src = e.target.result;
          document.getElementById('image-preview').style.display = 'block'; // Show the preview image
      };
      reader.readAsDataURL(file);
  }
}

// Send message to Flask API
function sendMessage() {
  const question = document.getElementById('user-input').value;
  const fileInput = document.getElementById('file-input');
  const chatbotMessages = document.getElementById('chatbot-messages');

  if (!question.trim() || !fileInput.files[0]) {
      alert("Please upload an image and enter a question.");
      return;
  }

  // Display user's question
  const userMessage = document.createElement('div');
  userMessage.classList.add('message', 'user');
  userMessage.innerHTML = `<p>${question}</p>`;

  // Display user's uploaded image
  if (fileInput.files[0]) {
    const imageUrl = URL.createObjectURL(fileInput.files[0]);
    userMessage.innerHTML += `<img src="${imageUrl}" alt="User image" style="max-width: 100px; max-height: 100px;" />`;
  }

  chatbotMessages.appendChild(userMessage);

  // Display "typing..." from chatbot with animation
  const botMessageTyping = document.createElement('div');
  botMessageTyping.classList.add('message', 'bot', 'typing');
  botMessageTyping.innerHTML = `<p>.</p>`;  // Bot is typing...
  chatbotMessages.appendChild(botMessageTyping);
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight;  // Scroll to bottom

  let typingInterval;
  let typingDots = 1;
  
  // Simulate typing with changing dots
  typingInterval = setInterval(() => {
      typingDots = (typingDots % 3) + 1; // Rotate between 1, 2, and 3 dots
      botMessageTyping.innerHTML = `<p>${'.'.repeat(typingDots)}</p>`;
  }, 500); // Change dots every 500ms

  // Send message and image to backend (Flask API)
  const formData = new FormData();
  formData.append('question', question);
  formData.append('image', fileInput.files[0]);

  fetch('/predict_batch', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      // Stop typing animation after response is received
      clearInterval(typingInterval);
      botMessageTyping.remove(); // Remove typing message

      if (data.error) {
          displayMessage(`Error: ${data.error}`, 'bot');
      } else {
          // Get the first prediction (assuming the response contains a predictions array)
          const firstPrediction = data.predictions[0];
          displayMessage(`${firstPrediction}`, 'bot');  // Display only the first answer
      }
  })
  .catch(error => {
      // Stop typing animation and display error message
      clearInterval(typingInterval);
      botMessageTyping.remove(); // Remove typing message
      displayMessage(`Error: ${error.message}`, 'bot');
  });
}

// Display message in chat window
function displayMessage(message, sender) {
  const chatbotMessages = document.getElementById('chatbot-messages');
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message', sender);

  const label = document.createElement('label');
  label.textContent = sender === 'bot' ? 'Bot:' : 'You:';
  messageDiv.appendChild(label);

  const text = document.createElement('p');
  text.innerHTML = message; // Use innerHTML for displaying line breaks
  messageDiv.appendChild(text);

  chatbotMessages.appendChild(messageDiv);
  chatbotMessages.scrollTop = chatbotMessages.scrollHeight; // Auto-scroll to bottom
}