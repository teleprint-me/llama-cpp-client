// app/scripts/request.js

// These parameters will be set by the user once the UI/UX is implemented.
// Note: The /v1/chat/completions endpoint uses ChatML messaging structure.
// Default parameters to enable basic functionality.
const parameters = {
  stream: true, // Optional: Get tokens as they're generated.
  cache_prompt: true, // Optional: Enhance model response times.
  seed: 1337, // Useful for testing; can be set to null in production.
  prompt: '', // Optional: For the /completion endpoint.
  messages: [], // Optional: For the /v1/chat/completions endpoint.
  top_k: 50, // Top-k parameter for token sampling.
  top_p: 0.9, // Top-p (nucleus) parameter for token sampling.
  min_p: 0.1, // Minimum probability threshold for token sampling.
  temperature: 0.7, // Temperature parameter for token sampling.
  presence_penalty: 0.0, // Presence penalty parameter for token sampling.
  frequency_penalty: 0.0, // Frequency penalty parameter for token sampling.
  repeat_penalty: 1.1, // Repeat penalty parameter for token sampling.
  n_predict: -1 // -1 allows the model to choose when to stop prediction.
};

// Make a RESTful API request
async function llamaCppRequest(prompt) {
  parameters.prompt = prompt;
  const response = await fetch('http://127.0.0.1:8080/completion', {
    method: 'POST',
    body: JSON.stringify(parameters),
    headers: { 'Content-Type': 'application/json' }
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response; // Return the response stream for processing
}

// Function to handle streamed tokens and update the UI in real-time
async function handleStreamedTokens(assistantMessageDiv) {
  const responseStream = await llamaCppRequest(parameters.prompt);
  const reader = responseStream.body.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      // Convert stream value to text
      const tokenString = new TextDecoder('utf-8').decode(value);
      // Handle the "data: " prefix
      const token = JSON.parse(tokenString.substring(6));

      // Dynamically update the assistant's message content
      assistantMessageDiv.textContent += token.content;

      // If the completion process is finished, stop the animation
      if (token.stop) {
        assistantMessageDiv.classList.remove('animated-border');
        break; // Exit the loop if the message is complete
      }
    }
  } catch (e) {
    console.error('Stream reading failed:', e);
    assistantMessageDiv.classList.remove('animated-border'); // Stop the normal animation
    assistantMessageDiv.classList.add('animated-border-error'); // Indicate an error state
  }
}

// This function handles user input and initiates the model response process
function generateModelCompletion(event) {
  // Get the users input prompt
  const prompt = document.querySelector('textarea#user-prompt').value.trim();
  if (!prompt) {
    alert('Please enter a prompt.'); // Basic validation
    return;
  }

  // Create the users message div
  const userMessageDiv = createChatMessage(modelChatTemplate, 'user');
  // Add the user message to the context window
  userMessageDiv.innerText = prompt;
  // Append the users message div to the context window
  document.querySelector('div#context-window').appendChild(userMessageDiv);
  // Clear the input field
  document.querySelector('textarea#user-prompt').value = '';

  // Create the assistant's message div with a placeholder or loading state
  const assistantMessageDiv = createChatMessage(modelChatTemplate, 'assistant');
  // Append the assistants message div to the context window
  document.querySelector('div#context-window').appendChild(assistantMessageDiv);
  // Update parameters with the current prompt
  parameters.prompt = prompt;

  // Handle streamed tokens and update the assistant message div in real-time
  handleStreamedTokens(assistantMessageDiv);
}
