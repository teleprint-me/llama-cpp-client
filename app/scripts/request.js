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

/**
 * Processes a streamed value from the server response, handling potential variations in format.
 *
 * This function decodes a Uint8Array value into a string and attempts to parse it as JSON.
 * Due to an identified issue with concatenated server responses (e.g., multiple JSON objects prefixed
 * with "data: " in a single response), this function splits the string on "data: " and parses the last segment.
 * This approach is a workaround for handling intermittent concatenated responses and may be subject to revision
 * upon further investigation into the server's behavior.
 *
 * @param {Uint8Array} value - The streamed value received from the server.
 * @returns {Object} The parsed JSON object from the server response.
 */
function processStreamValue(value) {
  // Decode the streamed value into a string
  const tokenString = new TextDecoder('utf-8').decode(value);

  // Split the string on "data: " to handle concatenated responses
  const split = tokenString.split('data: ');

  // Get the last segment as it is expected to contain the most recent JSON object
  const potentialJson = split[split.length - 1];

  // Parse the potential JSON object
  const token = JSON.parse(potentialJson);

  return token;
}

function signalErrorState(element) {
  element.classList.remove('animated-border'); // Stop the normal animation
  element.classList.add('animated-border-error'); // Indicate an error state
}

// Function to handle streamed tokens and update the UI in real-time
async function handleStreamedTokens(assistantMessageDiv) {
  try {
    const responseStream = await llamaCppRequest(parameters.prompt);
    const reader = responseStream.body.getReader();

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.warn('Stream ended without a stop token.');
        break; // Exit the loop when the stream is done
      }

      // Attempt to process the stream's value
      const token = processStreamValue(value);
      // Update UI; Note that token.content may be an empty string and will have no effect when that is the case.
      assistantMessageDiv.textContent += token.content;
      if (token.stop) {
        console.log('Received stop token.');
        break; // Exit the loop on receiving stop token
      }
    }
  } catch (error) {
    console.error('Failed to read stream:', error);
    signalErrorState(assistantMessageDiv);
  } finally {
    // Ensure the animation is stopped regardless of how the loop exits
    assistantMessageDiv.classList.remove('animated-border');
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
