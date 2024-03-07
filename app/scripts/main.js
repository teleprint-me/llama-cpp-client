// main.js

function modelGreetsUser(event) {
  const contextWindow = document.querySelector('div#context-window');
  if (contextWindow.children.length === 0) {
    const prompt = 'As a state of the art large language model, I';
    const completion = createCompletion('assistant', prompt);
    console.log(completion);
    contextWindow.appendChild(completion);
    parameters.prompt = prompt;
    handleStreamedTokens(completion, prompt);
  }
}

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

  MathJax = {
    tex: {
      inlineMath: [
        ['$', '$'],
        ['\\(', '\\)']
      ]
    }
  };

  hljs.highlightAll(); // initial code highlighting, if any

  window.addEventListener('load', modelGreetsUser);
}

setup();
