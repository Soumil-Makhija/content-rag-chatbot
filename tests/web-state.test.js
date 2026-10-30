'use strict';
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

// Exercise the shipped browser event handlers without adding a DOM dependency.
class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.listeners = new Map();
    this.value = '';
    this.files = [];
    this.disabled = false;
    this._text = '';
  }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  get firstChild() { return this.children[0] || null; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ''; this.children = children; }
  addEventListener(type, handler) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(handler);
  }
  fire(type) {
    for (const handler of this.listeners.get(type) || []) handler({preventDefault() {}});
  }
}

async function browser() {
  const ids = ['collection', 'documents', 'replacement', 'answer', 'status', 'load',
    'import-form', 'file', 'title', 'ask-form', 'question'];
  const elements = Object.fromEntries(ids.map(id => [id, new Element(id)]));
  elements.collection.value = 'alpha';
  const collections = {
    alpha: [{id: 'a-doc', title: 'Alpha handbook', version: 1, chunk_count: 2}],
    beta: [{id: 'b-doc', title: 'Beta notes', version: 1, chunk_count: 1}],
  };
  const requests = [];
  const confirmations = [];
  const context = vm.createContext({
    document: {
      getElementById: id => elements[id],
      createElement: tag => new Element(tag),
      querySelectorAll: () => Object.values(elements),
    },
    window: {confirm: message => { confirmations.push(message); return true; }},
    TextDecoder,
    fetch: async (url, options = {}) => {
      const method = options.method || 'GET';
      const body = options.body === undefined ? undefined : JSON.parse(options.body);
      requests.push({url, method, body});
      let result;
      if (url === '/api/health') result = {models: {ready: true}};
      else {
        const match = /^\/api\/collections\/([^/]+)\/(documents|ask)(?:\/([^/]+))?$/.exec(url);
        assert.ok(match, `Unexpected request: ${method} ${url}`);
        const [, group, resource, id] = match;
        if (resource === 'ask') result = {
          supported: true, answer: `Answer for ${group}`, citations: [],
          models: {embedding: 'test-embedding', generation: 'test-generation'},
        };
        else if (method === 'DELETE') {
          collections[group] = collections[group].filter(doc => doc.id !== id);
          result = {deleted: true};
        } else if (method === 'POST') {
          result = {id: 'imported'};
          collections[group].push({id: 'imported', title: body.title, version: 1, chunk_count: 1});
        } else result = collections[group].map(doc => ({...doc}));
      }
      return {ok: true, status: 200, json: async () => result};
    },
  });
  await vm.runInContext(readFileSync(path.join(__dirname, '../content_chat/web/app.js'), 'utf8'), context);
  async function fire(idOrElement, type) {
    (typeof idOrElement === 'string' ? elements[idOrElement] : idOrElement).fire(type);
    for (let attempt = 0; vm.runInContext('busy', context); attempt += 1) {
      assert.ok(attempt < 100, 'Browser action did not settle');
      await new Promise(resolve => setImmediate(resolve));
    }
  }
  return {elements, requests, confirmations, fire,
    deleteButton: () => elements.documents.firstChild.children.find(child => child.tagName === 'button')};
}

test('editing a collection clears its sources, replacement choices, answer, and delete access', async () => {
  const app = await browser();
  const oldDelete = app.deleteButton();
  app.elements.question.value = 'What is in this handbook?';
  await app.fire('ask-form', 'submit');
  assert.match(app.elements.answer.textContent, /Answer for alpha/);
  assert.equal(app.elements.replacement.children.length, 2);

  app.elements.collection.value = 'beta';
  await app.fire('collection', 'input');
  assert.equal(app.elements.documents.children.length, 0);
  assert.equal(app.elements.answer.children.length, 0);
  assert.equal(app.elements.replacement.children.length, 1);
  assert.equal(app.elements.replacement.firstChild.value, '');
  assert.match(app.elements.status.textContent, /Open this collection/);

  await app.fire(oldDelete, 'click');
  assert.equal(app.requests.filter(request => request.method === 'DELETE').length, 0);
  assert.equal(app.confirmations.length, 0);
  assert.match(app.elements.status.textContent, /Open this collection/);
});

test('an old detached delete stays invalid after opening another collection; its own delete works', async () => {
  const app = await browser();
  const oldDelete = app.deleteButton();
  app.elements.collection.value = 'beta';
  await app.fire('collection', 'input');
  await app.fire('load', 'click');
  assert.match(app.elements.documents.textContent, /Beta notes/);

  await app.fire(oldDelete, 'click');
  assert.equal(app.requests.filter(request => request.method === 'DELETE').length, 0);
  assert.match(app.elements.status.textContent, /previously opened collection/);

  await app.fire(app.deleteButton(), 'click');
  assert.deepEqual(app.requests.filter(request => request.method === 'DELETE').map(request => request.url),
    ['/api/collections/beta/documents/b-doc']);
  assert.match(app.elements.documents.textContent, /This collection is empty/);
  assert.match(app.elements.answer.textContent, /Document removed/);
});

test('ask and import require reopening after edits and target the opened collection', async () => {
  const app = await browser();
  app.elements.collection.value = 'beta';
  await app.fire('collection', 'input');
  app.elements.question.value = 'Summarize the notes';
  app.elements.file.files = [{name: 'new.md', size: 12, arrayBuffer: async () => new TextEncoder().encode('New content.').buffer}];
  for (const form of ['ask-form', 'import-form']) {
    const before = app.requests.length;
    await app.fire(form, 'submit');
    assert.equal(app.requests.length, before);
    assert.match(app.elements.status.textContent, /Open this collection/);
  }

  // Returning to the previous name also requires an explicit open.
  app.elements.collection.value = 'alpha';
  await app.fire('collection', 'input');
  await app.fire('ask-form', 'submit');
  assert.match(app.elements.status.textContent, /Open this collection/);

  app.elements.collection.value = 'beta';
  await app.fire('collection', 'input');
  await app.fire('load', 'click');
  await app.fire('ask-form', 'submit');
  assert.match(app.elements.answer.textContent, /Answer for beta/);
  await app.fire('import-form', 'submit');
  assert.deepEqual(app.requests.filter(request => request.method === 'POST').map(request => request.url),
    ['/api/collections/beta/ask', '/api/collections/beta/documents']);
  assert.equal(app.requests.at(-2).body.text, 'New content.');
  assert.match(app.elements.documents.textContent, /new.md/);
});
