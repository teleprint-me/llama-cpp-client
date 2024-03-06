// main.js

// Initialization function to setup event listeners
function setup() {
  // Handle user input on button click
  let generateCompletionButton = document.querySelector('#generate-completion');
  generateCompletionButton.addEventListener('click', generateModelCompletion);

  // Configure marked with highlight.js for code syntax highlighting
  marked.setOptions({
    highlight: function (code, lang) {
      const language = highlight.getLanguage(lang) ? lang : 'plaintext';
      return highlight.highlight(code, { language }).value;
    },
    langPrefix: 'hljs language-' // Use 'hljs' class prefix for compatibility with highlight.js CSS
  });

  hljs.highlightAll(); // initial code highlighting, if any
}

setup();
