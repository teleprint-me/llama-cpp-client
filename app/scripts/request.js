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

// Function to handle streamed tokens and update the UI in real-time
async function handleStreamedTokens(chatTemplate, assistantMessageDiv) {
  const responseStream = await llamaCppRequest(parameters.prompt);
  const reader = responseStream.body.getReader();
  const prefix = chatTemplate.assistant.prefix || '';
  const postfix = chatTemplate.assistant.postfix || '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      // Clean up before exiting the loop
      if (done || token.stop) {
        // If the completion process is finished, stop the animation
        assistantMessageDiv.classList.remove('animated-border');
        break; // Exit the loop on completion
      }

      // Convert stream value to text
      const tokenString = new TextDecoder('utf-8').decode(value);
      // Handle the "data: " prefix
      const token = JSON.parse(tokenString.substring(6));
      // NOTE: Skip content when empty, e.g. ""
      // Dynamically update the assistant's message content
      if (token.content) {
        messageDiv.textContent += prefix + token.content + postfix;
      }
    }
  } catch (e) {
    console.error('Stream reading failed', e);
  }
}

// This function handles the submission of user input
function generateModelCompletion(event) {
  console.log(event);
  const prompt = document.getElementById('user-prompt').value;
  // we want to clear the textarea once the user submits their message and we've stored it. We can only clear it after we've stored it.

  // There's another intermediary step here that's tricky because we need to have the model parameters as well as well as the chat template if there is any.

  // it's possible the model has no special tokens for a chat template because its a continuation model, e.g. the model assists with coding and is not fine-tuned on instructions or conversation.

  // There will need to be a method for handling this at some point, but my intuition tells me to separate concerns and simply focus on chat to keep it simple until basic functionality is implemented.

  // create the models message
  // assistantMessage = createChatMessage(chatTemplate, role, content = null)

  // add the div to the context window; this needs to be implemented
  // contextWindow = document.querySelector("div#context-window")
  // contextWindow.append(assistantMessage);

  // Call API endpoint to process and respond to the input
  let modelResponse = llamaCppRequest(prompt);

  // populate the div's content using the models response to stream the chunks, e.g. hadnleStreamedTokens

  // Process model response and concat tokens to output message as they're received
  console.log(modelResponse);
}

function getModelCompletionButton() {
  // Handle user input on button click
  let button = document.querySelector('#generate-completion');
  button.addEventListener('click', generateModelCompletion);
  return button;
}

// Make a RESTful API request here. For example:
async function llamaCppRequest(prompt) {
  parameters.prompt = prompt;
  // Perform some action based on the user's input.
  const response = await fetch('http://127.0.0.1:8080/completion', {
    method: 'POST',
    body: JSON.stringify(parameters),
    headers: { 'Content-Type': 'application/json' }
  });

  if (response.ok) {
    return response.json();
  } else {
    throw new Error(response.statusText);
  }
}

// Example usage of handleStreamedTokens function (replace with actual API response handling)
let responseStrings = [
  'data: {"content":"Hello","multimodal":false,"slot_id":0,"stop":false}',
  'data: {"content":" Austin","multimodal":false,"slot_id":0,"stop":false}',
  'data: {"content":"!","multimodal":false,"slot_id":0,"stop":false}'
  // Add more token strings as needed
];

// Simulate handling streamed tokens
responseStrings.forEach(handleStreamedTokens);
