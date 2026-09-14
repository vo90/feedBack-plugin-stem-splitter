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
  const timers = new Map();
  let timerId = 0;
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
    setTimeout(fn, delay) { timers.set(++timerId, {fn, delay}); return timerId; },
    clearTimeout(id) { timers.delete(id); }, setInterval(fn) { intervals.push(fn); return 1; },
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
    runTimers: async delay => {
      for (const [id, timer] of [...timers]) {
        if (timer.delay === delay) { timers.delete(id); timer.fn(); }
      }
      await flush();
    },
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

function verifiedServer() {
  return {installed: true, running: true, port: 7865, manageable: true,
    models_downloaded: true, models_ready: true, models_present: {bs_roformer_sw: true},
    gpu_build: true, gpu_detected: {name: 'Test NVIDIA card'},
    health: {device: 'cpu', gpu: true, demucs_model: 'bs_roformer_sw',
      warmup: {demucs: 'skipped', bs_roformer_sw: 'skipped', whisperx: 'skipped', crepe: 'skipped'}},
    source: {}, install_info: {},
    presentation: {selected_model: 'bs_roformer_sw',
      model: {state: 'on_demand', label: 'Installed and verified — loads when needed',
        present: true, verified: true, ready: true, warmup_state: 'skipped'},
      features: {
        demucs: {state: 'not_selected', label: 'Not selected'},
        whisperx: {state: 'on_demand', label: 'Not checked at startup', verified: null},
        crepe: {state: 'on_demand', label: 'Not checked at startup', verified: null},
      },
      requested_device: 'cpu', started_device: 'cpu', effective_device: 'cpu',
      restart_required: false, settled: true}};
}

test('verified on-demand model is separate from optional features and actual CPU execution', async () => {
  const ui = harness({'/server_status': verifiedServer()});
  await ui.flush();
  const server = ui.nodes.get('ss-srv-chips').innerHTML;
  const model = ui.nodes.get('ss-srv-model').innerHTML;
  const features = ui.nodes.get('ss-srv-feature-chips').innerHTML;
  assert.match(server, /ss-chip info[^>]*>Using CPU/);
  assert.match(server, /NVIDIA GPU support installed/);
  assert.ok(!/CPU · GPU|GPU idle|Reinstall|Models downloaded|Warm/.test(server));
  assert.match(model, /Selected separator: BS Roformer \(6 stems\)/);
  assert.match(model, /ss-chip ok[^>]*>Installed and verified/);
  assert.match(model, /ss-chip info[^>]*>Starts when needed/);
  assert.ok(!/skipped|Warm|Loaded/.test(model + features));
  assert.match(features, /Demucs separator: Not selected/);
  assert.match(features, /ss-chip off[^>]*>Lyrics transcription: Not checked at startup/);
  assert.match(features, /ss-chip off[^>]*>Pitch detection: Not checked at startup/);
  assert.ok(!features.includes('ss-chip ok'));
  assert.equal(ui.nodes.get('ss-srv-other-features').hidden, false);
  assert.equal(ui.nodes.get('ss-srv-device-note').hidden, true);
  assert.match(ui.nodes.get('ss-srv-gpu-label').textContent, /Install NVIDIA GPU support/);
  assert.match(html, /switching to CPU does not require reinstalling/);
  assert.match(html, /<label for="ss-srv-device">Run on<\/label>/);
});

test('legacy skipped status stays neutral even when old aggregate flags say ready and downloaded', async () => {
  const st = verifiedServer(); delete st.presentation;
  const ui = harness({'/server_status': st});
  await ui.flush();
  const model = ui.nodes.get('ss-srv-model').innerHTML;
  assert.match(model, /ss-chip off[^>]*>Not checked at startup/);
  assert.ok(!/ss-chip ok|Installed and verified|Warm|skipped/.test(model));
  const before = ui.requests.filter(r => r.endpoint === '/server_status').length;
  ui.intervals.forEach(fn => fn()); await ui.flush();
  assert.equal(ui.requests.filter(r => r.endpoint === '/server_status').length, before,
    'Intentional skipped startup is settled; it must not poll forever');
});

test('HTDemucs is the primary model and other-feature failures do not change its readiness', async () => {
  const st = verifiedServer();
  st.presentation.selected_model = 'htdemucs_6s';
  st.presentation.model = {state: 'ready', label: 'Ready to use', present: true, verified: true};
  st.presentation.features = {
    bs_roformer_sw: {state: 'not_selected', label: 'Not selected'},
    whisperx: {state: 'failed', label: 'Preparation failed', detail: 'ASR initialization failed'},
  };
  const ui = harness({'/server_status': st}); await ui.flush();
  const model = ui.nodes.get('ss-srv-model').innerHTML;
  assert.match(model, /Selected separator: HTDemucs \(6 stems\)/);
  assert.match(model, /ss-chip ok[^>]*>Ready to use/);
  assert.ok(!model.includes('BS Roformer'));
  assert.match(ui.nodes.get('ss-srv-feature-chips').innerHTML, /BS Roformer separator: Not selected/);
  assert.match(ui.nodes.get('ss-srv-feature-chips').innerHTML, /ss-chip bad[^>]*>Lyrics transcription: Preparation failed/);
  assert.equal(ui.nodes.get('ss-srv-model-errors').hidden, false);
  assert.match(ui.nodes.get('ss-srv-model-error-text').textContent, /Lyrics transcription: ASR initialization failed/);
});

test('legacy failed-prefix details stay readable and unknown states do not invent readiness', async () => {
  const st = verifiedServer(); delete st.presentation;
  st.health.warmup.bs_roformer_sw = 'failed: CUDA Init <device> failed';
  const ui = harness({'/server_status': st}); await ui.flush();
  assert.match(ui.nodes.get('ss-srv-model').innerHTML, /ss-chip bad[^>]*>Preparation failed/);
  assert.match(ui.nodes.get('ss-srv-model-error-text').textContent, /CUDA Init <device> failed/);
  assert.equal(ui.nodes.get('ss-srv-model-errors').hidden, false);
  st.health.warmup = {};
  await ui.click('ss-srv-test');
  assert.match(ui.nodes.get('ss-srv-model').innerHTML, /ss-chip off[^>]*>Status not reported/);
  assert.equal(ui.nodes.get('ss-srv-model-errors').hidden, true);
});

test('preparation polls until settled and ambiguous old downloading never claims a network download', async () => {
  const st = verifiedServer(); delete st.presentation;
  st.health.warmup.bs_roformer_sw = 'downloading';
  const ui = harness({'/server_status': st}); await ui.flush();
  assert.match(ui.nodes.get('ss-srv-model').innerHTML, /ss-chip info live[^>]*>.*Preparing model/);
  assert.ok(!ui.nodes.get('ss-srv-model').innerHTML.includes('Downloading'));
  const before = ui.requests.filter(r => r.endpoint === '/server_status').length;
  st.health.warmup.bs_roformer_sw = 'ready';
  ui.intervals.forEach(fn => fn()); await ui.flush();
  assert.equal(ui.requests.filter(r => r.endpoint === '/server_status').length, before + 1);
  assert.match(ui.nodes.get('ss-srv-model').innerHTML, /Ready to use/);
  ui.intervals.forEach(fn => fn()); await ui.flush();
  assert.equal(ui.requests.filter(r => r.endpoint === '/server_status').length, before + 1);
});

test('saved requested device shows restart notice without relabeling the running GPU', async () => {
  const st = verifiedServer();
  st.health.device = 'cuda';
  Object.assign(st.presentation, {requested_device: 'cuda', started_device: 'cuda', effective_device: 'cuda'});
  const ui = harness({'/server_status': st, '/config': {settings: {local_server_device: 'cuda'}}});
  await ui.flush();
  ui.defaults['/config'] = options => {
    const body = JSON.parse(options.body);
    st.presentation.requested_device = body.local_server_device || 'auto';
    st.presentation.restart_required = true;
    return {ok: true};
  };
  ui.nodes.get('ss-srv-device').value = 'cpu';
  await ui.nodes.get('ss-srv-device').fire('change');
  assert.equal(ui.nodes.get('ss-srv-device-note').hidden, true, 'Unsaved input is not persisted intent');
  await ui.runTimers(400);
  assert.equal(ui.nodes.get('ss-srv-device-note').hidden, false);
  assert.match(ui.nodes.get('ss-srv-device-note').textContent, /Restart needed.*CPU.*still using NVIDIA GPU/);
  assert.match(ui.nodes.get('ss-srv-chips').innerHTML, /Using NVIDIA GPU/);
  st.health.device = 'cpu';
  Object.assign(st.presentation, {started_device: 'cpu', effective_device: 'cpu', restart_required: false});
  await ui.click('ss-srv-test');
  assert.equal(ui.nodes.get('ss-srv-device-note').hidden, true);
  assert.match(ui.nodes.get('ss-srv-chips').innerHTML, /Using CPU/);
  assert.ok(!ui.requests.some(r => /server\/(start|stop|install)|runtime\/update/.test(r.endpoint)));
});

test('Auto remains the saved launch mode when the actual device is a GPU', async () => {
  const st = verifiedServer(); st.health.device = 'cuda:0';
  Object.assign(st.presentation, {requested_device: 'auto', started_device: 'auto', effective_device: 'cuda:0'});
  const ui = harness({'/server_status': st}); await ui.flush();
  assert.match(ui.nodes.get('ss-srv-chips').innerHTML, /Using NVIDIA GPU 0/);
  assert.equal(ui.nodes.get('ss-srv-device-note').hidden, true);
});

test('Start persists a recent CPU choice before launch and returns to enabled actual status', async () => {
  const st = verifiedServer(); st.running = false; st.health = {};
  const ui = harness({'/server_status': st}); await ui.flush();
  ui.nodes.get('ss-srv-device').value = 'cpu';
  await ui.nodes.get('ss-srv-device').fire('change');
  let saved = false;
  ui.defaults['/config'] = options => {
    assert.equal(JSON.parse(options.body).local_server_device, 'cpu');
    saved = true; return {ok: true};
  };
  ui.defaults['/server/start'] = () => {
    assert.equal(saved, true);
    st.running = true; st.health = {device: 'cpu', gpu: true};
    return {ok: true, started: 'start'};
  };
  await ui.click('ss-srv-toggle');
  assert.match(ui.nodes.get('ss-srv-chips').innerHTML, /Using CPU/);
  assert.equal(ui.nodes.get('ss-srv-toggle').disabled, false);
  assert.match(ui.nodes.get('ss-srv-toggle-label').textContent, /Running/);
  assert.ok(!ui.requests.some(r => /runtime\/(check|update)|server\/install/.test(r.endpoint)));
});

test('a failed device save blocks Start while Stop remains available without saving settings', async () => {
  const st = verifiedServer(); st.running = false;
  const ui = harness({'/server_status': st}); await ui.flush();
  ui.defaults['/config'] = new Error('Disk write failed');
  ui.nodes.get('ss-srv-device').value = 'cpu';
  await ui.click('ss-srv-toggle');
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/start'));
  assert.match(ui.nodes.get('ss-srv-saved').textContent, /Disk write failed/);
  assert.equal(ui.nodes.get('ss-srv-toggle').disabled, false);
  st.running = true;
  await ui.click('ss-srv-test');
  await ui.click('ss-srv-toggle');
  assert.ok(ui.requests.some(r => r.endpoint === '/server/stop'));
});

for (const failOld of [false, true]) {
  test('Start waits for an earlier autosave before persisting its newest CPU choice' + (failOld ? ' after failure' : ''), async () => {
    const st = verifiedServer(); st.running = false; st.health = {};
    const ui = harness({'/server_status': st}); await ui.flush();
    const writes = [];
    let finish, finishNew, persisted;
    ui.defaults['/config'] = options => {
      const device = JSON.parse(options.body).local_server_device;
      writes.push(device);
      if (writes.length === 1) return new Promise((resolve, reject) => {
        finish = () => {
          if (failOld) reject(new Error('Earlier save failed'));
          else { persisted = device; resolve({ok: true}); }
        };
      });
      return new Promise(resolve => { finishNew = () => { persisted = device; resolve({ok: true}); }; });
    };
    ui.nodes.get('ss-srv-device').value = 'cuda';
    await ui.nodes.get('ss-srv-device').fire('change');
    await ui.runTimers(400);
    assert.equal(typeof finish, 'function');
    ui.nodes.get('ss-srv-device').value = 'cpu';
    await ui.nodes.get('ss-srv-device').fire('change');
    ui.defaults['/server/start'] = () => {
      assert.equal(persisted, 'cpu');
      st.running = true; st.health = {device: 'cpu', gpu: true};
      return {ok: true, started: 'start'};
    };
    const starting = ui.click('ss-srv-toggle'); await ui.flush();
    assert.deepEqual(writes, ['cuda'], 'Only one settings write can be in flight');
    assert.ok(!ui.requests.some(r => r.endpoint === '/server/start'));
    finish(); await ui.flush();
    assert.equal(typeof finishNew, 'function');
    assert.equal(ui.nodes.get('ss-srv-toggle').disabled, true,
      'Idle status reads during a queued settings save must not unlock Start');
    finishNew(); await starting;
    assert.deepEqual(writes, ['cuda', 'cpu']);
    assert.equal(persisted, 'cpu');
    assert.match(ui.nodes.get('ss-srv-chips').innerHTML, /Using CPU/);
  });
}

test('an old status GET cannot replace an authoritative server completion event', async () => {
  const old = verifiedServer(); old.running = false; old.health = {};
  const ui = harness({'/server_status': old}); await ui.flush();
  let finish;
  ui.defaults['/server_status'] = () => new Promise(resolve => { finish = resolve; });
  const checking = ui.click('ss-srv-test'); await ui.flush();
  assert.equal(typeof finish, 'function');
  const running = verifiedServer();
  ui.sockets[0].onmessage({data: JSON.stringify({type: 'server_done', op: 'start', status: running})});
  await ui.flush();
  assert.match(ui.nodes.get('ss-srv-toggle-label').textContent, /Running/);
  finish(old); await checking; await ui.flush();
  assert.match(ui.nodes.get('ss-srv-toggle-label').textContent, /Running/);
  assert.match(ui.nodes.get('ss-srv-chips').innerHTML, /Using CPU/);
});

test('a stale initial status read cannot unlock Start while its settings save is pending', async () => {
  let finishStatus, finishSave;
  const ui = harness({'/server_status': () => new Promise(resolve => { finishStatus = resolve; })});
  await ui.flush();
  ui.defaults['/config'] = () => new Promise(resolve => { finishSave = resolve; });
  ui.nodes.get('ss-srv-device').value = 'cpu';
  const starting = ui.click('ss-srv-toggle'); await ui.flush();
  assert.equal(typeof finishSave, 'function');
  assert.equal(ui.nodes.get('ss-srv-toggle').disabled, true);
  const before = ui.requests.filter(r => r.endpoint === '/server/runtime/status').length;
  const stopped = verifiedServer(); stopped.running = false; stopped.health = {};
  ui.defaults['/server_status'] = stopped;
  finishStatus(stopped); await ui.flush();
  assert.equal(ui.nodes.get('ss-srv-toggle').disabled, true);
  assert.equal(ui.requests.filter(r => r.endpoint === '/server/runtime/status').length, before,
    'Discarded old server status must not start a fresh runtime read that unlocks controls');
  assert.ok(!ui.requests.some(r => r.endpoint === '/server/start'));
  finishSave({ok: true}); await starting;
  assert.equal(ui.requests.filter(r => r.endpoint === '/server/start').length, 1);
});

test('language-specific word timing failures and preparation remain visible independently', async () => {
  for (const legacy of [false, true]) {
    const st = verifiedServer();
    if (legacy) {
      delete st.presentation;
      st.health.warmup.whisperx_aligners = {en: 'ready', fr: 'failed: Missing French model', de: 'downloading'};
    } else {
      Object.assign(st.presentation.features, {
        'whisperx_aligners:en': {state: 'ready', label: 'Ready to use'},
        'whisperx_aligners:fr': {state: 'failed', label: 'Preparation failed', detail: 'Missing French model'},
        'whisperx_aligners:de': {state: 'loading', label: 'Preparing model…'},
      });
      st.presentation.settled = false;
    }
    const ui = harness({'/server_status': st}); await ui.flush();
    const features = ui.nodes.get('ss-srv-feature-chips').innerHTML;
    assert.match(features, /Word timing \(en\): Ready to use/);
    assert.match(features, /ss-chip bad[^>]*>Word timing \(fr\): Preparation failed/);
    assert.match(features, /ss-chip info live[^>]*>.*Word timing \(de\): Preparing model/);
    assert.match(ui.nodes.get('ss-srv-model-error-text').textContent, /Word timing \(fr\): .*Missing French model/);
    const before = ui.requests.filter(r => r.endpoint === '/server_status').length;
    ui.intervals.forEach(fn => fn()); await ui.flush();
    assert.equal(ui.requests.filter(r => r.endpoint === '/server_status').length, before + 1);
  }
});
