// app/scripts/request.js

// these will be set by the user once the ui/ux is implemented
// /v1/chat/completions uses ChatML messaging structure
let parameters = {
  stream: true, // optional, get tokens as they're generated
  cache_prompt: true, // optional, enhance model response times
  seed: 1337, // useful for testing, can be set to null in production
  prompt: '', // optional, for /completion endpoint
  messages: [], // optional, for /v1/chat/completions endpoint
  top_k: 50,
  top_p: 0.9,
  min_p: 0.1,
  temperature: 0.7,
  presence_penalty: 0.0,
  frequency_penalty: 0.0,
  repeat_penalty: 1.1,
  n_predict: -1 // -1 allows model to choose when to stop
};

// Function to handle streamed tokens
function handleStreamedTokens(chatTemplate, tokenString) {
  // Create a new html element for the models resposne
  let div = createChatMessage(chatTemplate, 'assistant', null);

  // Strip the "data: " prefix and parse the JSON object
  let token = JSON.parse(tokenString.substring(6));

  // Update the assistant's message content with the token content
  div.textContent += token.content;

  // Check if the completion process is finished
  if (token.stop) {
    // Remove the animated border class to stop the animation
    div.classList.remove('animated-border');
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

  // Call API endpoint to process and respond to the input
  let modelResponse = llamaCppRequest(prompt);

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
