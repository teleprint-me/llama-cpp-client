// main.js

// Initialization function to setup event listeners
function setup() {
  // Handle user input on button click
  let generateCompletionButton = document.querySelector('#generate-completion');
  generateCompletionButton.addEventListener('click', generateModelCompletion);
}

setup();
