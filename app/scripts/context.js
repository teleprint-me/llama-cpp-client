/*
 * context.js
 *
 * This module manages the context window between the user and the language model.
 * The context window may contain a sequence of messages exchanged between the user and the model.
 * Each message consists of a role (e.g., system, user, assistant) and content.
 */

// Array to store messages in the context window
let messages = [];

// Template for formatting chat messages
let modelChatTemplate = {
  model: 'llama',
  system: {
    prefix: '<<SYS>>',
    postfix: '<</SYS>>'
  },
  user: {
    prefix: '[INST] ',
    postfix: ' [/INST]'
  },
  assistant: {
    prefix: '',
    postfix: ''
  },
  getModelAsName: function () {
    return this.model.charAt(0).toUpperCase() + this.model.slice(1);
  }
};

/**
 * Retrieves the messages in the model's context window.
 * @returns {NodeList} The messages in the context window.
 */
function getModelContextWindow() {
  let contextWindow = document.querySelector('div#context-window');
  return contextWindow.querySelectorAll('div.message');
}

/**
 * Generates a system message based on the chat template.
 * @param {Object} chatTemplate - The chat template object.
 * @returns {string} The generated system message.
 */
function getModelSystemMessage(chatTemplate) {
  let model = chatTemplate.getModelAsName();
  let prefix = chatTemplate.system.prefix || '';
  let postfix = chatTemplate.system.postfix || '';

  return `${prefix}My name is ${model}. I am an advanced Large Language Model designed to assist you.${postfix}`;
}

/**
 * Creates a message element for the chat interface.
 * @param {Object} chatTemplate - The chat template object.
 * @param {string} role - The role of the message (system, user, assistant).
 * @param {string} content - The content of the message.
 * @returns {HTMLElement} The created message element.
 */
function createChatMessage(chatTemplate, role, content = null) {
  // Create a new <div> element
  let div = document.createElement('div');

  // Set the role attribute as metadata
  div.setAttribute('data-role', role);

  // Add the "message" class to the element
  div.classList.add('message');

  // Add additional classes based on the role
  if (role === 'assistant') {
    div.classList.add('animated-border');
  }

  // Set the content of the message, if provided
  if (content !== null) {
    // Check if the role exists in the chatTemplate
    if (chatTemplate.hasOwnProperty(role)) {
      // Retrieve prefix and postfix values from the chatTemplate
      let prefix = chatTemplate[role].prefix || '';
      let postfix = chatTemplate[role].postfix || '';

      // Create a new <p> element
      let p = document.createElement('p');

      // Set inner HTML content allowing for HTML formatting
      p.innerHTML = `${prefix}${content}${postfix}`;

      // Append the <p> element to the <div>
      div.appendChild(p);
    } else {
      console.error(`Role '${role}' not found in chat template.`);
    }
  }

  // Return the created message element
  return div;
}
