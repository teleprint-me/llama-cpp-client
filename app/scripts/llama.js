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
    return response;
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

class LlamaClient {
  constructor(llamaRequest, ...parameters) {
    this.request = llamaRequest;

    // These parameters will be set by the user once the UI/UX is implemented.
    // Note: The /v1/chat/completions endpoint uses ChatML messaging structure.
    // Default parameters to enable basic functionality.
    this.parameters = {
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

  async getServerHealth() {
    try {
      const response = await fetch('http://127.0.0.1:8080/health', {
        method: 'GET'
      });
      if (!response.ok) {
        throw Error(`Response is not ok with status code ${response.status}`);
      }
      return response.json();
    } catch (e) {
      console.error(e);
    }
  }
}
