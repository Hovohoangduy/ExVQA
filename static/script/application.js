let uploadedImage = null;
let uploadedFileName = null; // Variable to store the file name

const rules = [
    { question: "what shows branching papillae having flbrovascular stalk covered by a single layer of cuboidal cells having ground-glass nuclei?", 
        image: "img1.jpg",
        answer: "yes" },
    { question: "what shows scattered inflammatory cells, calcification arrowheads, and neovascularization?", 
        image: "img2.jpg",
        answer: "high-power view of the junction of the fibrous cap and core" },
    { question: "how are the histone subunits charged?", 
        image: "img3.jpg",
        answer: "positively charged" },
];

function sendMessage() {
    const userInput = document.getElementById('user-input').value.trim().toLowerCase();
    const messagesContainer = document.getElementById('chatbot-messages');
    const preview = document.getElementById('image-preview');

    if (userInput || uploadedImage) {
        // Display user's message
        const userMessage = document.createElement('div');
        userMessage.classList.add('message', 'user');
        userMessage.innerHTML = `<p>${userInput}</p>`;
        if (uploadedImage) {
            userMessage.innerHTML += `<img src="${uploadedImage}" alt="User image" style="max-width: 100px; max-height: 100px;" />`;
        }
        messagesContainer.appendChild(userMessage);

        // Display "typing..." from chatbot with animation
        const botMessageTyping = document.createElement('div');
        botMessageTyping.classList.add('message', 'bot', 'typing');
        botMessageTyping.innerHTML = `<p>.</p>`;  // Bot is typing...
        messagesContainer.appendChild(botMessageTyping);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;  // Scroll to bottom

        let typingInterval;
        let typingDots = 1;
        
        // Simulate typing with changing dots
        typingInterval = setInterval(() => {
            typingDots = (typingDots % 3) + 1; // Rotate between 1, 2, and 3 dots
            botMessageTyping.innerHTML = `<p>${'.'.repeat(typingDots)}</p>`;
        }, 500); // Change dots every 500ms

        // Delay the bot's response by 5 seconds
        setTimeout(() => {
            // Stop the typing animation after 5 seconds
            clearInterval(typingInterval);
            botMessageTyping.remove(); // Remove typing message

            // Generate bot response after 5 seconds
            const botMessage = document.createElement('div');
            botMessage.classList.add('message', 'bot');

            // Use uploadedFileName instead of the base64 encoding
            console.log("File Name:", uploadedFileName); // Check file name

            const matchedRule = rules.find(rule =>
                rule.question.toLowerCase() === userInput &&  // Kiểm tra câu hỏi (chuyển thành chữ thường)
                (!rule.image || (uploadedFileName && uploadedFileName === rule.image))  // Kiểm tra tên hình ảnh chính xác
            );

            console.log("Matched Rule:", matchedRule); // Debug rule matching

            if (matchedRule) {
                botMessage.innerHTML = `<p>${matchedRule.answer}</p>`;
            } else {
                botMessage.innerHTML = `<p>Sorry, I don't understand your question. </p>`;
            }
            messagesContainer.appendChild(botMessage);

            // Reset input and preview
            document.getElementById('user-input').value = '';
            preview.style.display = 'none';
            uploadedImage = null;
            uploadedFileName = null; // Reset file name
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }, 5000); // 5000 milliseconds = 5 seconds
    }
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        const preview = document.getElementById('image-preview');
        reader.onload = function(e) {
            uploadedImage = e.target.result; // Save Base64 image
            uploadedFileName = file.name; // Save the file name
            preview.src = uploadedImage;
            preview.style.display = 'block'; // Show preview
        };
        reader.readAsDataURL(file);
    }
}

function triggerFileInput() {
    document.getElementById('file-input').click();
}

function allowDrop(event) {
    event.preventDefault();
}

function drop(event) {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            const preview = document.getElementById('image-preview');
            reader.onload = function(e) {
                uploadedImage = e.target.result; // Save Base64 image
                uploadedFileName = file.name; // Save the file name
                preview.src = uploadedImage;
                preview.style.display = 'block'; // Show preview
            };
            reader.readAsDataURL(file);
        }
    }
}