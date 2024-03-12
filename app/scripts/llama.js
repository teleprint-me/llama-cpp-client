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
