'use strict';
const $ = id => document.getElementById(id);
let busy = false;
let openedCollection = null;
function collection() {
  const value = $('collection').value;
  if (!/^[a-z][a-z0-9_-]{0,63}$/.test(value)) throw new Error('Use a lowercase collection name starting with a letter (1–64 characters).');
  return value;
}
function selectedCollection() {
  const value = collection();
  if (value !== openedCollection) throw new Error('Open this collection before importing or asking.');
  return value;
}
function endpoint(resource, group = selectedCollection()) { return `/api/collections/${encodeURIComponent(group)}/${resource}`; }
async function api(path, method = 'GET', data) {
  const options = {method, headers: {}};
  if (data !== undefined) { options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(data); }
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function node(tag, text, className) {
  const element = document.createElement(tag); element.textContent = text;
  if (className) element.className = className;
  return element;
}
async function task(action) {
  if (busy) return;
  busy = true;
  document.querySelectorAll('button,input,select,textarea').forEach(el => el.disabled = true);
  $('status').textContent = 'Working locally…';
  try { await action(); $('status').textContent = 'Ready.'; }
  catch (error) { $('status').textContent = error.message; }
  finally { busy = false; document.querySelectorAll('button,input,select,textarea').forEach(el => el.disabled = false); }
}
async function library() {
  const group = collection();
  const documents = await api(endpoint('documents', group));
  openedCollection = group;
  $('documents').replaceChildren(); $('replacement').replaceChildren(node('option', 'Add a new document')); $('replacement').firstChild.value = '';
  if (!documents.length) $('documents').append(node('li', 'This collection is empty.', 'muted'));
  for (const doc of documents) {
    const item = node('li', `${doc.title} · v${doc.version} · ${doc.chunk_count} chunks`);
    const link = node('a', 'Inspect'); link.href = endpoint(`documents/${encodeURIComponent(doc.id)}`, group); link.target = '_blank'; link.rel = 'noopener'; item.append(link);
    const remove = node('button', 'Delete'); remove.type = 'button';
    remove.addEventListener('click', () => { if (window.confirm(`Delete ${doc.title} and its indexed passages?`)) task(async () => { await api(endpoint(`documents/${encodeURIComponent(doc.id)}`, group), 'DELETE'); $('answer').replaceChildren(node('p', 'Document removed. Ask again using the current sources.', 'muted')); await library(); }); });
    item.append(remove); $('documents').append(item);
    const option = node('option', `${doc.title} (v${doc.version})`); option.value = doc.id; $('replacement').append(option);
  }
}
$('load').addEventListener('click', () => task(async () => { $('answer').replaceChildren(node('p', 'Collection opened. Ask a question using these documents.', 'muted')); await library(); }));
$('import-form').addEventListener('submit', event => { event.preventDefault(); task(async () => {
  const file = $('file').files[0];
  if (!file || !/\.(md|txt)$/i.test(file.name)) throw new Error('Choose a .txt or .md document.');
  if (file.size > 2 * 1024 * 1024) throw new Error('Document exceeds 2 MiB.');
  const text = new TextDecoder('utf-8', {fatal: true}).decode(await file.arrayBuffer());
  await api(endpoint('documents'), 'POST', {title: $('title').value || file.name, text, document_id: $('replacement').value || null});
  $('file').value = ''; $('title').value = ''; $('answer').replaceChildren(node('p', 'Document indexed. Ask again using the current sources.', 'muted')); await library();
}); });
$('ask-form').addEventListener('submit', event => { event.preventDefault(); task(async () => {
  const group = selectedCollection();
  const result = await api(endpoint('ask', group), 'POST', {question: $('question').value});
  $('answer').replaceChildren(node('h3', result.supported ? 'Answer from your sources' : 'Sources do not answer this'), node('p', result.answer));
  if (result.citations.length) $('answer').append(node('h3', 'Supporting passages'));
  for (const source of result.citations) {
    const details = document.createElement('details');
    details.append(node('summary', `${source.title} · v${source.version} · passage ${source.ordinal + 1}`));
    details.append(node('blockquote', source.text));
    const link = node('a', 'Open source record'); link.href = source.url; link.target = '_blank'; link.rel = 'noopener'; details.append(link);
    $('answer').append(details);
  }
  $('answer').append(node('p', `Embedding: ${result.models.embedding} · Generation: ${result.models.generation}`, 'muted'));
}); });
(async () => {
  try {
    await library();
    const health = await api('/api/health');
    $('status').textContent = health.models.ready ? 'Local models available. Ready.' : (health.models.error || `Install the selected Ollama models: ${(health.models.missing_models || []).join(', ')}`);
  } catch (error) { $('status').textContent = error.message; }
})();
