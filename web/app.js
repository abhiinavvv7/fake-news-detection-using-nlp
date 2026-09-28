const form = document.querySelector('#analysis-form');
const textInput = document.querySelector('#article-text');
const countOutput = document.querySelector('#word-count');
const analyzeButton = document.querySelector('#analyze-button');
const sampleButton = document.querySelector('#sample-button');
const clearButton = document.querySelector('#clear-button');
const sampleNote = document.querySelector('#sample-note');
const resultContent = document.querySelector('#result-content');

function wordCount(value) {
  return value.trim() ? value.trim().split(/\s+/).length : 0;
}

function updateInputState() {
  const count = wordCount(textInput.value);
  countOutput.textContent = `${count.toLocaleString()} ${count === 1 ? 'word' : 'words'}`;
  analyzeButton.disabled = count === 0;
}

function showError(message) {
  resultContent.className = 'result-content';
  resultContent.innerHTML = '';
  const notice = document.createElement('div');
  notice.className = 'error-state';
  notice.textContent = message;
  resultContent.appendChild(notice);
}

function showLoading() {
  resultContent.className = 'result-content idle loading';
  resultContent.innerHTML = '<p class="micro-label">COMPARING LANGUAGE PATTERNS</p><p class="idle-title">Reading the lines…</p><p class="idle-copy">The model is comparing this passage with patterns in its training set.</p>';
}

function addText(parent, tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  parent.appendChild(element);
  return element;
}

function showResult(result) {
  const label = result.label === 'fake' ? 'fake' : 'real';
  const title = label === 'fake' ? 'Likely fake' : 'Likely real';
  const confidence = result.confidence * 100;
  resultContent.className = 'result-content';
  resultContent.replaceChildren();

  const topline = addText(resultContent, 'div', 'report-topline', 'TEXT PATTERN MATCH');
  const words = document.createElement('span');
  words.textContent = `${result.word_count.toLocaleString()} WORDS`;
  topline.appendChild(words);
  addText(resultContent, 'h3', `result-title ${label}`, title);

  const confidenceRow = document.createElement('div');
  confidenceRow.className = 'confidence-row';
  addText(confidenceRow, 'span', '', 'Estimated model confidence');
  addText(confidenceRow, 'strong', 'confidence-value', `${confidence.toFixed(1)}%`);
  resultContent.appendChild(confidenceRow);

  const meter = document.createElement('div');
  meter.className = 'meter';
  meter.setAttribute('role', 'meter');
  meter.setAttribute('aria-label', 'Estimated model confidence');
  meter.setAttribute('aria-valuemin', '0');
  meter.setAttribute('aria-valuemax', '100');
  meter.setAttribute('aria-valuenow', confidence.toFixed(1));
  const fill = document.createElement('div');
  fill.className = `meter-fill ${label}`;
  fill.style.width = `${Math.max(0, Math.min(100, confidence))}%`;
  meter.appendChild(fill);
  resultContent.appendChild(meter);

  const list = document.createElement('ul');
  list.className = 'probability-list';
  for (const className of ['fake', 'real']) {
    if (!(className in result.probabilities)) continue;
    const row = document.createElement('li');
    addText(row, 'span', '', `${className[0].toUpperCase()}${className.slice(1)} signal`);
    addText(row, 'strong', '', `${(result.probabilities[className] * 100).toFixed(1)}%`);
    list.appendChild(row);
  }
  resultContent.appendChild(list);
  addText(resultContent, 'p', 'result-advice', 'A high estimate can still be wrong. Check the publisher, date, and evidence before sharing.');
}

textInput.addEventListener('input', () => {
  sampleNote.hidden = true;
  updateInputState();
});

sampleButton.addEventListener('click', () => {
  textInput.value = 'The city library opened a new reading room on Tuesday. Visitors can borrow books and attend free workshops each weekend.';
  sampleNote.hidden = false;
  updateInputState();
  textInput.focus();
});

clearButton.addEventListener('click', () => {
  textInput.value = '';
  sampleNote.hidden = true;
  updateInputState();
  resultContent.className = 'result-content idle';
  resultContent.innerHTML = '<div class="idle-mark" aria-hidden="true"><span></span><span></span><span></span></div><p class="idle-title">Waiting for a passage.</p><p class="idle-copy">Add a headline or article. The model will compare its language patterns with examples from its training set.</p>';
  textInput.focus();
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const count = wordCount(textInput.value);
  if (count < 5) {
    showError('Add at least five words so the model has some context.');
    textInput.focus();
    return;
  }
  analyzeButton.disabled = true;
  showLoading();
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: textInput.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'The model could not read that passage.');
    showResult(payload);
  } catch (error) {
    showError(error.message || 'Could not reach the local model. Is the app still running?');
  } finally {
    updateInputState();
  }
});

updateInputState();
