// app/scripts/llama.js

class LlamaRequest {
  constructor(baseUrl = null, port = null, headers = null) {
    this.baseUrl = baseUrl || 'http://127.0.0.1';
    this.port = port || '8080';
    this.headers = headers || {
      'Content-Type': 'application/json'
    };
  }

  async _handleResponse(url, method = 'GET', body = null) {
    let response = null;
    let headers = this.headers;

    // !IMPORTANT: GET requests cannot have a body!
    if (method === 'GET') {
      console.log('GET', 'handle response', url);
      response = await fetch(url, {
        method: method,
        headers: headers
      });
    } else {
      console.log('POST', 'handle response', url);
      response = await fetch(url, {
        method: method,
        headers: headers,
        body: JSON.stringify(body)
      });
    }

    if (!response.ok) {
      console.error(response.status, response.type, response.headers);
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    console.log('response is ok');
    return response; // NOTE: Returning json breaks streaming!
  }

  async get(endpoint, requestData) {
    const url = `${this.baseUrl}:${this.port}${endpoint}`;
    return await this._handleResponse(url, 'GET', requestData);
  }

  async post(endpoint, requestData) {
    console.log('POST', endpoint, requestData);
    const url = `${this.baseUrl}:${this.port}${endpoint}`;
    return await this._handleResponse(url, 'POST', requestData);
  }

  /**
   * Processes a streamed value from the server response, handling potential variations in format.
   *
   * This method decodes a Uint8Array value into a string and attempts to parse it as JSON.
   * Due to an identified issue with concatenated server responses (e.g., multiple JSON objects prefixed
   * with "data: " in a single response), this function splits the string on "data: " and parses the last segment.
   * This approach is a workaround for handling intermittent concatenated responses and may be subject to revision
   * upon further investigation into the server's behavior.
   *
   * @param {Uint8Array} value - The streamed value received from the server.
   * @returns {Object} The parsed JSON object from the server response.
   */
  _processStreamValue(value) {
    console.log('Processing stream value', value);
    // Decode the streamed value into a string
    const tokenString = new TextDecoder('utf-8').decode(value);
    // Split the string on "data: " to handle concatenated responses
    const split = tokenString.split('data: ');
    // Get the last segment as it is expected to contain the most recent JSON object
    const potentialJson = split[split.length - 1];
    // Parse the potential JSON object
    const token = JSON.parse(potentialJson);
    console.log('Parsed JSON', token);
    // Return extracted chunked response
    return token;
  }

  async stream(endpoint, requestData, callback) {
    console.log('starting stream...');
    // response is a promise
    const response = await this.post(endpoint, requestData);
    console.log('getting reader...');
    const reader = response.body.getReader();
    let endStream = false;

    while (true) {
      console.log('processing stream for', endpoint);
      const { done, value } = await reader.read();
      if (done) {
        console.warn('Stream ended without a stop token.');
        break; // Exit the loop when the stream is done
      }

      const token = this._processStreamValue(value);
      endStream = callback(token);

      if (endStream) {
        console.log(
          `Received stop token. Predicted ${token.tokens_predicted} tokens.`
        );
        break; // Exit the loop on receiving stop token
      }
    }
  }
}

class LlamaAPI {
  constructor(llamaRequest = null, parameters = null) {
    this.request = llamaRequest || LlamaRequest();

    // These parameters will be set by the user once the UI/UX is implemented.
    // Note: The /v1/chat/completions endpoint uses ChatML messaging structure.
    // Default parameters to enable basic functionality.
    this.parameters = parameters || {
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
  }

  async getHealth() {
    return this.request.get('/health');
  }

  async getSlots() {
    return this.request.get('/slots');
  }

  async getCompletions(prompt, callback = null, stream = false) {
    // NOTE: prompt can be a string or an array of strings
    this.parameters.prompt = prompt;
    if (stream) {
      return this.request.stream('/v1/completions', this.parameters, callback);
    } else {
      return this.request.get('/v1/completions', this.parameters);
    }
  }

  async getChatCompletions(messages, callback = null, stream = false) {
    // NOTE: messages is an array of objects where each object
    // has a role and content where role is one of system,
    // assistant, or user
    this.parameters.messages = messages;
    if (stream) {
      return this.request.stream(
        '/v1/chat/completions',
        this.parameters,
        callback
      );
    } else {
      return this.request.get('/v1/chat/completions', this.parameters);
    }
  }
}

class LlamaCompletions {
  constructor(llamaRequest = null, parameters = null) {
    this.completions = []; // track raw completions

    this.llamaAPI = new LlamaAPI(
      llamaRequest || new LlamaRequest(),
      parameters
    );

    // context window has formatted completions
    this.contextWindow = document.querySelector('div#context-window');
    // text area has user input
    this.userPrompt = document.querySelector('textarea#user-prompt');

    // buttons trigger events based on user input
    this.generateButton = document.querySelector('button#generate-completion');
    this.regenerateButton = document.querySelector(
      'button#regenerate-completion'
    );
    this.parametersButton = document.querySelector('button#model-parameters');

    // setup event listeners
    this.setupListeners();
  }

  setupListeners() {
    // Add event listener to the generate button
    this.generateButton.addEventListener(
      'click',
      this.generateModelCompletion.bind(this)
    );
    // this.regenerateButton.addEventListener(
    //   'click',
    //   this.handleRegenerateCompletion.bind(this)
    // );
    // this.parametersButton.addEventListener(
    //   'click',
    //   this.handleModelParameters.bind(this)
    // );
  }

  /**
   * Creates a message element for the completion interface.
   * @param {string} content - The content of the message.
   * @returns {HTMLElement} The created message element.
   */
  createCompletion(prompt = null) {
    // Create a new <div> element
    let div = document.createElement('div');
    // Set the role attribute as metadata
    div.setAttribute('data-role', 'completion');
    // Add the "message" class to the element
    div.classList.add('message');
    // Add animation during generation
    div.classList.add('animated-border');
    // Set the prompt for the completion, if provided
    if (prompt !== null) {
      // Set inner HTML content allowing for automated formatting
      div.innerHTML = marked.parse(prompt);
    }
    // Return the created message element
    return div; // this is added to contextWindow
  }

  _toggleGenerateStop() {
    this.generateButton.querySelector('p').innerText = 'Stop';
    this.generateButton.querySelector('i').classList.remove('bx-play');
    this.generateButton.querySelector('i').classList.add('bx-stop');
  }

  _toggleGenerateStart() {
    this.generateButton.querySelector('p').innerText = 'Generate';
    this.generateButton.querySelector('i').classList.remove('bx-stop');
    this.generateButton.querySelector('i').classList.add('bx-play');
  }

  _signalErrorState(completionDiv) {
    completionDiv.classList.remove('animated-border'); // Stop the normal animation
    completionDiv.classList.add('animated-border-error'); // Indicate an error state
  }

  async handleGenerateCompletion(completionDiv, prompt) {
    console.log('using prompt to handle generating completion');
    try {
      // Toggle the generate button
      this._toggleGenerateStop();
      // Request the model completion
      const promise = await this.llamaAPI.getCompletions(
        prompt, // prompt
        // use lambda to avoid 'this' conflicts
        function (token) {
          // callback handles the models completion
          if (token.stop) {
            console.log(
              `Received stop token. Predicted ${token.tokens_predicted} tokens.`
            );
            return true; // no content left to extract
          }
          console.log('got token', token.content);
          prompt += token.content;
          // Update the context window with the generated completion
          completionDiv.innerHTML = marked.parse(prompt);
          // Need to update to highlight all code blocks within the element
          completionDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
          });
          MathJax.typesetPromise();
          return false; // continue processing
        },
        true // enable streaming
      );
      this.completions.push(prompt);
    } catch (error) {
      console.error('Failed to read stream:', error);
      this._signalErrorState(completionDiv);
    } finally {
      // Ensure the animation is stopped regardless of how the loop exits
      completionDiv.classList.remove('animated-border');
      // restore original button widget in the ui
      this._toggleGenerateStart();
    }
  }

  generateModelCompletion(event) {
    // Get the users input
    const prompt = this.userPrompt.value.trim();
    // Basic validation of user input
    if (!prompt) {
      alert('Please enter a prompt.');
      return;
    }
    // Clear the input field
    this.userPrompt.value = '';
    // Create the assistants completions div with a loading state
    const completionDiv = this.createCompletion(prompt);
    // Add the completions div to the context window
    this.contextWindow.appendChild(completionDiv);
    MathJax.typesetPromise();
    // Handle the streamed tokens and update the completion in real-time
    this.handleGenerateCompletion(completionDiv, prompt);
  }
}

class LlamaClient {
  constructor() {
    this.name = 'llama-cpp-client';
    this.version = 'v1';
    this.state = 'prototype';
    this.completions = new LlamaCompletions();
  }
}

document.addEventListener('DOMContentLoaded', function () {
  // Initialize Llama client
  let llama = new LlamaClient();
  console.log('Successfully initialized llama.js');
  console.log(llama.name, llama.version, llama.state);

  // Configure marked.js with highlight.js for code syntax highlighting
  marked.setOptions({
    highlight: function (code, lang) {
      // Function to handle code syntax highlighting using highlight.js
      // Get the language if available, otherwise set it as plain text
      const language = highlight.getLanguage(lang) ? lang : 'plaintext';
      // Apply highlighting and get the highlighted code
      return highlight.highlight(code, { language }).value;
    },
    // Use 'hljs' class prefix for compatibility with highlight.js CSS
    langPrefix: 'hljs language-'
  });
  console.log('Successfully initialized marked.js');

  // Highlight all the code snippets in the document
  hljs.highlightAll(); // Initial code highlighting, if any
  console.log('Successfully initialized highlight.js');
});
