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
      response = await fetch(url, {
        method: method,
        headers: headers
      });
    } else {
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
    return response.json();
  }

  async get(endpoint, requestData) {
    const url = `${this.baseUrl}:${this.port}${endpoint}`;
    return await this._handleResponse(url, 'GET', requestData);
  }

  async post(endpoint, requestData) {
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
    // Decode the streamed value into a string
    const tokenString = new TextDecoder('utf-8').decode(value);
    // Split the string on "data: " to handle concatenated responses
    const split = tokenString.split('data: ');
    // Get the last segment as it is expected to contain the most recent JSON object
    const potentialJson = split[split.length - 1];
    // Parse the potential JSON object
    const token = JSON.parse(potentialJson);
    // Return extracted chunked response
    return token;
  }

  async stream(endpoint, requestData, callback) {
    const response = await this.post(endpoint, requestData);
    const reader = response.body.getReader();
    let endStream = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.warn('Stream ended without a stop token.');
        break; // Exit the loop when the stream is done
      }

      const token = this._processStreamValue(value);
      endStream = callback(token, requestData);

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
  constructor(llamaRequest, parameters = null) {
    this.request = llamaRequest;

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

  async getCompletions(prompt, stream = false) {
    // NOTE: prompt can be a string or an array of strings
    this.parameters.prompt = prompt;
    if (stream) {
      return this.request.stream('/v1/completions', this.parameters);
    } else {
      return this.request.get('/v1/completions', this.parameters);
    }
  }

  async getChatCompletions(messages, stream = false) {
    // NOTE: messages is an array of objects where each object
    // has a role and content where role is one of system,
    // assistant, or user
    this.parameters.messages = messages;
    if (stream) {
      return this.request.stream('/v1/chat/completions', this.parameters);
    } else {
      return this.request.get('/v1/chat/completions', this.parameters);
    }
  }
}

class LlamaClient {
  constructor() {
    this.name = 'llama-cpp-client';
    this.version = 'v1';
    this.state = 'prototype';
    this.setup();
  }

  setup() {
    this.request = new LlamaRequest();
    this.client = new LlamaAPI(this.request);
  }
}

// Initialize Llama client
let llama = new LlamaClient();
llama.setup();
console.log(llama.name, llama.version, llama.state);
console.log('Successfully initialized llama.js');

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

// Initialize MathJax for rendering math equations
MathJax = {
  tex: {
    inlineMath: [
      ['$', '$'], // Dollar signs for inline math
      ['\\(', '\\)'], // Parentheses for display math
      ['[', ']'] // Brackets for display math (alternative to parentheses)
    ]
  }
};
console.log('Successfully initialized mathjax');

// Highlight all the code snippets in the document
hljs.highlightAll(); // Initial code highlighting, if any
console.log('Successfully initialized highlight.js');
