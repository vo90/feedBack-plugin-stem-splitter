/* Drive the shipped settings script, with HTTP/WebSocket responses and a small
 * DOM adapter. Tests assert user actions, not copied rendering functions. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'settings.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const API = '/api/plugins/stem_splitter';

function harness(overrides = {}) {
  const nodes = new Map();
  for (const match of html.matchAll(/<([a-z0-9]+)\b[^>]*\bid="([^"]+)"[^>]*>/gi)) {
    const listeners = new Map();
    const raw = match[0];
    nodes.set(match[2], {
      tagName: match[1].toUpperCase(), type: raw.match(/type="([^"]+)"/)?.[1] || '',
      value: raw.match(/value="([^"]*)"/)?.[1] || '', checked: /\bchecked\b/.test(raw),
      disabled: /\bdisabled\b/.test(raw), hidden: /\bhidden\b/.test(raw),
      textContent: '', innerHTML: '', className: '', options: [], style: {},
      scrollTop: 0, scrollHeight: 0, clientHeight: 0,
      classList: {add() {}, remove() {}},
      addEventListener(event, fn) {
        listeners.set(event, [...(listeners.get(event) || []), fn]);
      },
      fire(event) { return Promise.all((listeners.get(event) || []).map(fn => fn())); },
    });
  }
  const requests = [];
  const sockets = [];
  const intervals = [];
  const inventory = {installed: true, source_commit: 'a'.repeat(40), legacy: true,
    dependencies: {'audio-separator': '0.44.5'}, models: {}};
  const defaults = {
    '/config': {settings: {remote_model: 'bs_roformer_sw', local_server_autostart: false,
      local_server_gpu: false, local_server_port: 7865}},
    '/server_status': {installed: true, running: true, models_downloaded: true,
      manageable: true, models_present: {}, health: {}, source: {}, install_info: {}},
    '/engine_status': {installed: {}}, '/sidecar_status': {docker: false, in_container: false},
    '/jobs': {jobs: []}, '/api/settings': {},
    '/server/runtime/inventory': inventory,
    '/server/runtime/status': {state: 'idle', active: false},
    '/server/runtime/check': {can_update: true, plan_id: 'checked-plan', state: 'available',
      model: 'bs_roformer_sw', installed: inventory, available: {server: 'b'.repeat(40),
        dependencies: {'audio-separator': {version: '0.47.0'}},
        models: [{id: 'bs_roformer_sw', installed: null, available: 'verified-r1'}]}},
    '/server/runtime/update': {ok: true, started: 'runtime_update'},
    '/server/runtime/cancel': {cancel_requested: true},
    ...overrides,
  };
  const storage = new Map();
  const context = {
    document: {getElementById: id => nodes.get(id)},
    window: {localStorage: {getItem: k => storage.get(k), setItem: (k, v) => storage.set(k, v), removeItem: k => storage.delete(k)}},
    navigator: {}, location: {protocol: 'http:', host: '127.0.0.1:19000'},
    WebSocket: class {constructor() { sockets.push(this); } close() {}},
    setTimeout() { return 1; }, clearTimeout() {}, setInterval(fn) { intervals.push(fn); return 1; },
    confirm() { throw new Error('An unexpected confirmation was shown'); },
    fetch: async (url, options = {}) => {
      const endpoint = url.startsWith(API) ? url.slice(API.length) : url;
      requests.push({endpoint, method: options.method || 'GET', body: options.body && JSON.parse(options.body)});
      const configured = defaults[endpoint];
      const data = typeof configured === 'function' ? await configured(options) : configured;
      if (data instanceof Error) throw data;
      return {ok: true, status: 200, json: async () => structuredClone(data || {ok: true})};
    },
    console, Promise,
  };
  vm.runInNewContext(script, context, {filename: 'settings.html'});
  const flush = async () => { for (let i = 0; i < 5; i++) await new Promise(setImmediate); };
  return {nodes, requests, sockets, defaults, intervals, flush,
    click: async id => { await nodes.get(id).fire('click'); await flush(); }};
}

test('opening settings reads versions without checking the internet or installing', async () => {
  const ui = harness();
  await ui.flush();
  assert.match(ui.nodes.get('ss-runtime-versions').innerHTML, /0\.44\.5/);
  assert.equal(ui.nodes.get('ss-srv-install').disabled, false);
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, true);
  assert.ok(!ui.requests.some(r => /runtime\/(check|update)|server\/(update|install)$/.test(r.endpoint)));
});

test('checking exposes versions but installation requires the separate apply action', async () => {
  const ui = harness();
  await ui.flush();
  await ui.click('ss-srv-update');
  assert.match(ui.nodes.get('ss-runtime-versions').innerHTML, /0\.47\.0/);
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, false);
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/runtime/update'));
  await ui.click('ss-runtime-apply');
  assert.deepEqual(ui.requests.find(r => r.endpoint === '/server/runtime/update').body, {plan_id: 'checked-plan'});
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/install' || r.endpoint === '/server/update'));
});

test('a failed check cannot claim current or enable installation', async () => {
  const ui = harness({'/server/runtime/check': {can_update: false, state: 'unknown', reason: 'Network unavailable'}});
  await ui.flush();
  await ui.click('ss-srv-update');
  assert.match(ui.nodes.get('ss-runtime-note').textContent, /Network unavailable/);
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, true);
  assert.equal(ui.nodes.get('ss-srv-update').disabled, false);
  assert.match(ui.nodes.get('ss-runtime-versions').innerHTML, /0\.44\.5/);
});

test('model revision text is escaped before it is inserted into the table', async () => {
  const ui = harness();
  ui.defaults['/server/runtime/check'].available.models[0].available = '<img src=x onerror=bad()>';
  await ui.flush();
  await ui.click('ss-srv-update');
  const rendered = ui.nodes.get('ss-runtime-versions').innerHTML;
  assert.ok(rendered.includes('&lt;img'));
  assert.ok(!rendered.includes('<img'));
});

test('prepared legacy update is distinct from activated and waits for server stop', async () => {
  const ui = harness({'/server/runtime/status': {state: 'waiting_to_activate', active: false,
    pending_activation: true, phase: 'Prepared', pct: .8}});
  await ui.flush();
  assert.match(ui.nodes.get('ss-runtime-note').textContent, /prepared/);
  assert.equal(ui.nodes.get('ss-runtime-activate').hidden, false);
  assert.equal(ui.nodes.get('ss-runtime-activate').disabled, true);
});

test('progress permits cancellation and never turns a runtime result into server status', async () => {
  const ui = harness();
  await ui.flush();
  ui.sockets[0].onmessage({data: JSON.stringify({type: 'server', op: 'runtime_update',
    active: true, state: 'installing', phase: 'Installing libraries', pct: .4})});
  assert.equal(ui.nodes.get('ss-runtime-cancel').hidden, false);
  assert.equal(ui.nodes.get('ss-runtime-cancel').disabled, false);
  await ui.click('ss-runtime-cancel');
  assert.ok(ui.requests.some(r => r.endpoint === '/server/runtime/cancel'));
  ui.defaults['/server/runtime/status'] = {state: 'waiting_to_activate', active: false,
    pending_activation: true, phase: 'Prepared', pct: .8};
  ui.sockets[0].onmessage({data: JSON.stringify({type: 'server_done', op: 'runtime_update',
    status: ui.defaults['/server/runtime/status']})});
  await ui.flush();
  assert.match(ui.nodes.get('ss-srv-toggle-label').textContent, /Running/);
  assert.match(ui.nodes.get('ss-runtime-note').textContent, /prepared/);
});

test('changing a target invalidates an already checked plan', async () => {
  const ui = harness();
  await ui.flush();
  await ui.click('ss-srv-update');
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, false);
  ui.nodes.get('ss-srv-ref').value = 'another-release';
  await ui.nodes.get('ss-srv-ref').fire('input');
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, true);
});

test('an older check response cannot restore a plan after settings change', async () => {
  const ui = harness();
  await ui.flush();
  const original = ui.defaults['/server/runtime/check'];
  let finish;
  ui.defaults['/server/runtime/check'] = () => new Promise(resolve => { finish = resolve; });
  const checking = ui.click('ss-srv-update');
  await ui.flush();
  assert.equal(typeof finish, 'function');
  ui.nodes.get('ss-srv-ref').value = 'another-release';
  await ui.nodes.get('ss-srv-ref').fire('input');
  finish(original);
  await checking;
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, true);
  assert.match(ui.nodes.get('ss-runtime-note').textContent, /Settings changed/);
  await ui.click('ss-runtime-apply');
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/runtime/update'));
});

test('a failed settings save prevents checking and applying different saved settings', async () => {
  const ui = harness();
  await ui.flush();
  ui.defaults['/config'] = new Error('Settings write failed');
  await ui.click('ss-srv-update');
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, true);
  assert.match(ui.nodes.get('ss-runtime-note').textContent, /Settings write failed/);
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/runtime/check'));
});

test('polling recovers enabled controls when the completion event was missed', async () => {
  const ui = harness();
  await ui.flush();
  await ui.click('ss-srv-update');
  await ui.click('ss-runtime-apply');
  ui.sockets[0].onmessage({data: JSON.stringify({type: 'server', op: 'runtime_update',
    active: true, state: 'installing', phase: 'Installing libraries', pct: .4})});
  assert.equal(ui.nodes.get('ss-srv-update').disabled, true);
  ui.defaults['/server/runtime/status'] = {state: 'active', active: false, busy: null};
  ui.intervals.forEach(fn => fn());
  await ui.flush();
  assert.equal(ui.nodes.get('ss-srv-update').disabled, false);
});

test('reopening a prepared update shows its saved versions without re-enabling apply', async () => {
  const ui = harness({'/server/runtime/status': {state: 'waiting_to_activate', active: false,
    pending_activation: true, phase: 'Prepared', checked_plan: {can_update: false,
      available: {server: 'b'.repeat(40), dependencies: {'audio-separator': {version: '0.47.0'}},
        models: [{id: 'bs_roformer_sw', available: 'verified-r1', download_bytes: 699416805}]}}}});
  await ui.flush();
  assert.match(ui.nodes.get('ss-runtime-versions').innerHTML, /0\.47\.0/);
  assert.match(ui.nodes.get('ss-runtime-details').textContent, /667 MiB/);
  assert.equal(ui.nodes.get('ss-runtime-apply').disabled, true);
  assert.match(ui.nodes.get('ss-runtime-note').textContent, /prepared/);
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/runtime/check'));
});

test('completed update renders current model receipt instead of the old plan inventory', async () => {
  const ui = harness({'/server/runtime/inventory': {installed: true, legacy: false,
    dependencies: {'audio-separator':'0.47.0'}, models: {bs_roformer_sw:{revision:'verified-r1',state:'verified'}}},
    '/server/runtime/status': {state: 'active', active: false, checked_plan: {can_update:false,
      available: {models: [{id:'bs_roformer_sw',installed:null,available:'verified-r1'}]}}}});
  await ui.flush();
  const rows = ui.nodes.get('ss-runtime-versions').innerHTML;
  assert.match(rows, /verified-r1 — verified/);
  assert.ok(!rows.includes('Verification pending'));
});
