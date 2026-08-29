/* CacheLLM dashboard behaviour.
 *
 * Accessibility notes, since they drove most of the decisions here:
 *  - Tabs implement the ARIA tabs pattern: arrow keys move between tabs,
 *    Home/End jump to the ends, and only the selected tab is in the tab order.
 *  - Nothing is announced by colour alone; every outcome has a text label.
 *  - Auto-refresh does NOT touch aria-live, because a table quietly rewriting
 *    itself every 5 seconds would make a screen reader unusable. Only actions
 *    the user took are announced.
 *  - Focus is moved deliberately (e.g. onto the entry heading after "Inspect")
 *    and never stolen otherwise.
 *  - All text is built with textContent, so a prompt containing markup can
 *    never inject anything.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 5000;
let CURRENCY = '';
let timer = null;

/* ---------------------------------------------------------------- helpers */

function say(message, assertive = false) {
  const region = assertive ? $('alerts') : $('live');
  region.textContent = '';                       // force a re-announce
  window.setTimeout(() => { region.textContent = message; }, 30);
}

function num(value) {
  return (value ?? 0).toLocaleString();
}

function money(value, currency) {
  const amount = Number(value ?? 0);
  const text = amount === 0 ? '0.00' : amount.toFixed(amount < 0.01 ? 6 : 2);
  return currency ? `${text} ${currency}` : text;
}

function clockTime(epochSeconds) {
  if (!epochSeconds) return 'unknown';
  return new Date(epochSeconds * 1000).toLocaleTimeString();
}

function seconds(value) {
  if (value === null || value === undefined) return 'never expires';
  return `${Math.max(0, Math.round(value))} seconds`;
}

async function api(path, options) {
  const response = await fetch(path, Object.assign(
    { headers: { 'Content-Type': 'application/json' } }, options || {}));
  const text = await response.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (!response.ok) {
    const message = (data.error && (data.error.message || data.error)) || response.statusText;
    throw new Error(message);
  }
  return data;
}

function cell(row, text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  row.appendChild(td);
  return td;
}

function rowHeader(row, text, className) {
  const th = document.createElement('th');
  th.scope = 'row';
  th.textContent = text;
  if (className) th.className = className;
  row.appendChild(th);
  return th;
}

/* Human wording for the outcome codes, plus a shape so colour is never the
 * only signal. */
const OUTCOMES = {
  exact_hit:    { label: 'hit (exact)',    kind: 'hit' },
  semantic_hit: { label: 'hit (similar)',  kind: 'hit' },
  tool_hit:     { label: 'hit (tool)',     kind: 'hit' },
  coalesced:    { label: 'hit (merged)',   kind: 'hit' },
  miss:         { label: 'miss',           kind: 'miss' },
  bypass:       { label: 'not cacheable',  kind: 'miss' },
  error:        { label: 'error',          kind: 'error' },
};

function outcomeCell(row, outcome) {
  const info = OUTCOMES[outcome] || { label: outcome || 'unknown', kind: 'miss' };
  const td = document.createElement('td');
  const span = document.createElement('span');
  span.className = `outcome outcome-${info.kind}`;
  span.textContent = (info.kind === 'hit' ? '\u2713 ' : info.kind === 'error' ? '\u2715 ' : '\u2022 ') + info.label;
  td.appendChild(span);
  row.appendChild(td);
}

function emptyRow(tbody, colspan, message) {
  const tr = document.createElement('tr');
  const td = document.createElement('td');
  td.colSpan = colspan;
  td.className = 'muted';
  td.textContent = message;
  tr.appendChild(td);
  tbody.appendChild(tr);
}

function actionButton(label, accessibleLabel, handler, danger) {
  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = label;
  button.setAttribute('aria-label', accessibleLabel);
  if (danger) button.className = 'danger';
  button.addEventListener('click', handler);
  return button;
}

/* ------------------------------------------------------------------- tabs */

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const loaders = {
  'tab-requests': loadRequests,
  'tab-cache': loadCache,
  'tab-tools': loadTools,
  'tab-pricing': loadPricing,
  'tab-config': loadConfig,
};

function selectTab(tab, { focus = true } = {}) {
  tabs.forEach((candidate) => {
    const selected = candidate === tab;
    candidate.setAttribute('aria-selected', String(selected));
    candidate.tabIndex = selected ? 0 : -1;
    $(candidate.getAttribute('aria-controls')).hidden = !selected;
  });
  if (focus) tab.focus();
  const loader = loaders[tab.id];
  if (loader) loader();
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectTab(tab, { focus: false }));
  tab.addEventListener('keydown', (event) => {
    const map = {
      ArrowRight: (index + 1) % tabs.length,
      ArrowLeft: (index - 1 + tabs.length) % tabs.length,
      Home: 0,
      End: tabs.length - 1,
    };
    const target = map[event.key];
    if (target === undefined) return;
    event.preventDefault();
    selectTab(tabs[target]);
  });
});

/* --------------------------------------------------------------- overview */

function card(list, label, value, note) {
  const wrapper = document.createElement('div');
  wrapper.className = 'card';
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.textContent = value;
  if (note) {
    const small = document.createElement('span');
    small.className = 'note';
    small.textContent = note;
    dd.appendChild(small);
  }
  wrapper.append(dt, dd);
  list.appendChild(wrapper);
}

async function refresh({ announce = false } = {}) {
  let stats;
  let health;
  try {
    [stats, health] = await Promise.all([api('/api/stats'), api('/health')]);
  } catch (error) {
    say(`Could not load statistics: ${error.message}`, true);
    return;
  }

  CURRENCY = stats.cost.currency || '';
  $('st-version').textContent = `version ${health.version}`;
  $('st-backend').textContent = `cache: ${health.cache.exact.backend || 'unknown'}`;
  const up = health.upstream || {};
  $('st-upstream').textContent = `provider: ${up.ok ? 'reachable' : 'unreachable'}`;

  const c = stats.counters;
  const cards = $('cards');
  cards.textContent = '';
  card(cards, 'Requests', num(c.total_requests), `${num(c.upstream_requests)} went to the provider`);
  card(cards, 'Served from cache', `${stats.cache_hit_rate_pct}%`,
       `${num(c.cache_hits)} hits, ${num(c.cache_misses)} misses`);
  card(cards, 'Exact hits', num(c.exact_hits));
  card(cards, 'Similar-question hits', num(c.semantic_hits));
  card(cards, 'Tool result hits', num(c.tool_cache_hits));
  card(cards, 'Duplicate calls merged', num(c.coalesced_requests),
       'identical requests that arrived at the same time');
  card(cards, 'Tokens saved', num(stats.estimated_tokens_saved),
       `${num(c.saved_input_tokens)} in, ${num(c.saved_output_tokens)} out`);
  card(cards, 'Money saved', money(stats.cost.savings, CURRENCY),
       `${stats.cost.savings_pct}% of what it would have cost`);
  card(cards, 'Provider errors', num(c.upstream_errors), `${num(c.cache_errors)} cache errors`);

  $('cost-actual').textContent = money(stats.cost.actual_cost, CURRENCY);
  $('cost-without').textContent = money(stats.cost.cost_without_cache, CURRENCY);
  $('cost-saved').textContent = money(stats.cost.savings, CURRENCY);
  $('cost-pct').textContent = `${stats.cost.savings_pct}%`;
  const pct = Math.max(0, Math.min(100, Number(stats.cost.savings_pct) || 0));
  $('cost-meter').style.width = `${pct}%`;
  $('cost-meter-wrap').setAttribute('aria-label', `Saved ${pct} percent of the would-be cost`);
  $('cost-note').textContent = stats.cost.cost_without_cache === 0
    ? 'No prices set yet, so this stays at zero. Add them on the Pricing tab.'
    : 'This session. Lifetime totals come from the SQLite usage table.';

  const latency = $('latency');
  latency.textContent = '';
  card(latency, 'Average round trip', `${stats.latency.avg_total_ms} ms`,
       `over ${stats.latency.samples} requests`);
  card(latency, 'Cache lookup', `${stats.latency.avg_cache_lookup_ms} ms`);
  card(latency, 'Provider call', `${stats.latency.avg_upstream_ms} ms`);

  const byModel = $('by-model');
  byModel.textContent = '';
  const models = stats.by_model || [];
  if (!models.length) {
    emptyRow(byModel, 8, 'No traffic yet.');
  } else {
    models.forEach((m) => {
      const tr = document.createElement('tr');
      rowHeader(tr, m.model || 'unknown', 'mono');
      cell(tr, num(m.requests), 'num');
      cell(tr, num(m.hits), 'num');
      cell(tr, num(m.prompt_tokens), 'num');
      cell(tr, num(m.completion_tokens), 'num');
      cell(tr, num((m.saved_prompt_tokens || 0) + (m.saved_completion_tokens || 0)), 'num');
      cell(tr, money(m.cost, ''), 'num');
      cell(tr, money(m.cost_without_cache, ''), 'num');
      byModel.appendChild(tr);
    });
  }
  $('by-model-caption').textContent =
    `Usage grouped by model. ${models.length} model${models.length === 1 ? '' : 's'}.`;

  const recent = $('recent');
  recent.textContent = '';
  const items = stats.recent_requests || [];
  if (!items.length) {
    emptyRow(recent, 7, 'No requests yet.');
  } else {
    items.forEach((r) => {
      const tr = document.createElement('tr');
      cell(tr, clockTime(r.created_at));
      outcomeCell(tr, r.outcome);
      cell(tr, r.model || 'unknown', 'mono');
      cell(tr, r.category || 'unknown');
      cell(tr, `${r.total_latency_ms} ms`, 'num');
      cell(tr, num((r.prompt_tokens || 0) + (r.completion_tokens || 0)), 'num');
      cell(tr, (r.prompt_excerpt || '').slice(0, 70) || 'not stored', 'muted');
      recent.appendChild(tr);
    });
  }

  if (announce) {
    say(`Refreshed. ${num(c.total_requests)} requests, ${stats.cache_hit_rate_pct} percent served from cache, ` +
        `${money(stats.cost.savings, CURRENCY)} saved.`);
  }
}

/* --------------------------------------------------------------- requests */

async function loadRequests() {
  const query = new URLSearchParams({ limit: $('req-limit').value });
  const outcome = $('req-outcome').value;
  if (outcome) query.set('outcome', outcome);
  let data;
  try { data = await api(`/api/requests?${query}`); }
  catch (error) { say(`Could not load the request log: ${error.message}`, true); return; }

  const tbody = $('requests');
  tbody.textContent = '';
  if (!data.requests.length) {
    emptyRow(tbody, 12, 'Nothing logged yet.');
  } else {
    data.requests.forEach((r) => {
      const tr = document.createElement('tr');
      cell(tr, clockTime(r.created_at));
      cell(tr, r.request_id, 'mono');
      cell(tr, r.endpoint);
      outcomeCell(tr, r.outcome);
      cell(tr, r.model || 'unknown', 'mono');
      cell(tr, r.stream ? 'yes' : 'no');
      cell(tr, Number(r.total_latency_ms || 0).toFixed(1), 'num');
      cell(tr, Number(r.cache_latency_ms || 0).toFixed(1), 'num');
      cell(tr, Number(r.upstream_latency_ms || 0).toFixed(1), 'num');
      cell(tr, num((r.prompt_tokens || 0) + (r.completion_tokens || 0)), 'num');
      cell(tr, Number(r.cost || 0).toFixed(6), 'num');
      cell(tr, r.error || '', 'muted');
      tbody.appendChild(tr);
    });
  }
  const label = outcome ? `, filtered to ${OUTCOMES[outcome]?.label || outcome}` : '';
  $('req-caption').textContent = `${data.requests.length} requests${label}.`;
  say(`Loaded ${data.requests.length} requests${label}.`);
}

/* ---------------------------------------------------------- cache browser */

async function loadCache() {
  const query = new URLSearchParams({ limit: '100' });
  const search = $('cache-search').value.trim();
  const model = $('cache-model').value.trim();
  const namespace = $('cache-ns').value.trim();
  if (search) query.set('search', search);
  if (model) query.set('model', model);
  if (namespace) query.set('namespace', namespace);

  let data;
  try { data = await api(`/api/cache?${query}`); }
  catch (error) { say(`Could not load the cache: ${error.message}`, true); return; }

  const tbody = $('cache-rows');
  tbody.textContent = '';
  if (!data.entries.length) {
    emptyRow(tbody, 10, search ? 'Nothing matched that search.' : 'The cache is empty.');
  } else {
    data.entries.forEach((entry) => {
      const tr = document.createElement('tr');
      rowHeader(tr, entry.key.slice(0, 16), 'mono');
      cell(tr, entry.model || 'unknown', 'mono');
      cell(tr, entry.category || 'unknown');
      cell(tr, entry.namespace);
      cell(tr, clockTime(entry.created_at));
      cell(tr, entry.expired ? 'expired' : seconds(entry.ttl_remaining));
      cell(tr, num(entry.hits), 'num');
      cell(tr, num(entry.size_bytes), 'num');
      cell(tr, (entry.prompt_excerpt || '').slice(0, 60) || 'not stored', 'muted');
      const actions = document.createElement('td');
      actions.append(
        actionButton('Inspect', `Inspect entry ${entry.key.slice(0, 16)}`,
                     () => inspectEntry(entry.key)),
        document.createTextNode(' '),
        actionButton('Delete', `Delete entry ${entry.key.slice(0, 16)}`,
                     () => deleteEntry(entry.key), true),
      );
      tr.appendChild(actions);
      tbody.appendChild(tr);
    });
  }
  $('cache-caption').textContent =
    `Showing ${data.entries.length} of ${data.total} stored responses` +
    (data.note ? `. ${data.note}` : '.');
  say(`${data.entries.length} of ${data.total} entries shown.`);
}

async function inspectEntry(key) {
  try {
    const data = await api(`/api/cache/${key}`);
    $('entry-box').hidden = false;
    $('entry-json').textContent = JSON.stringify(data, null, 2);
    $('entry-heading').textContent = `Entry ${key.slice(0, 16)}`;
    $('entry-heading').focus();              // take the reader straight there
    say('Entry loaded below.');
  } catch (error) {
    say(`Could not load that entry: ${error.message}`, true);
  }
}

async function deleteEntry(key) {
  if (!window.confirm(`Delete cache entry ${key.slice(0, 16)}?`)) return;
  try {
    await api(`/api/cache/${key}`, { method: 'DELETE' });
    say('Entry deleted.');
    loadCache();
  } catch (error) {
    say(`Could not delete that entry: ${error.message}`, true);
  }
}

async function invalidate(body, description) {
  const clean = Object.fromEntries(Object.entries(body).filter(([, v]) => v));
  if (!Object.keys(clean).length) { say('Nothing to remove; fill in a value first.'); return; }
  try {
    const result = await api('/api/cache/invalidate',
                             { method: 'POST', body: JSON.stringify(clean) });
    const removed = result.removed;
    say(`Removed ${removed.cache_entries} responses and ${removed.tool_entries} tool results ${description}.`);
    loadCache();
    refresh();
  } catch (error) {
    say(`Could not remove those entries: ${error.message}`, true);
  }
}

/* ------------------------------------------------------------- tool cache */

async function loadTools() {
  let data;
  try { data = await api('/api/tools/cache?limit=100'); }
  catch (error) { say(`Could not load tool results: ${error.message}`, true); return; }

  const tbody = $('tool-rows');
  tbody.textContent = '';
  if (!data.entries.length) {
    emptyRow(tbody, 7, 'No tool results cached.');
  } else {
    const now = Date.now() / 1000;
    data.entries.forEach((entry) => {
      const tr = document.createElement('tr');
      rowHeader(tr, entry.key.slice(0, 16), 'mono');
      cell(tr, entry.tool_name, 'mono');
      cell(tr, (entry.arguments || '').slice(0, 60) || 'not stored', 'muted');
      cell(tr, clockTime(entry.created_at));
      cell(tr, entry.expired ? 'expired'
            : seconds(entry.expires_at ? entry.expires_at - now : null));
      cell(tr, num(entry.hits), 'num');
      const actions = document.createElement('td');
      actions.appendChild(actionButton('Delete', `Delete tool result for ${entry.tool_name}`,
                                       () => deleteToolEntry(entry.key), true));
      tr.appendChild(actions);
      tbody.appendChild(tr);
    });
  }
  $('tool-caption').textContent = `Showing ${data.entries.length} of ${data.total} tool results.`;
}

async function deleteToolEntry(key) {
  try {
    await api(`/api/tools/cache/${key}`, { method: 'DELETE' });
    say('Tool result deleted.');
    loadTools();
  } catch (error) {
    say(`Could not delete that tool result: ${error.message}`, true);
  }
}

async function checkToolPolicy() {
  const tool = $('tool-check').value.trim();
  if (!tool) { say('Type a tool name first.'); return; }
  try {
    const result = await api('/v1/tools/cache/policy',
                             { method: 'POST', body: JSON.stringify({ tool }) });
    const message = result.cacheable
      ? `${result.tool} can be cached for ${result.ttl} seconds, as category ${result.category}.`
      : `${result.tool} is never cached. Reason: ${result.reason}.`;
    $('tool-policy-result').textContent = message;
  } catch (error) {
    say(`Could not check that tool: ${error.message}`, true);
  }
}

/* ---------------------------------------------------------------- pricing */

async function loadPricing() {
  let data;
  try { data = await api('/api/pricing'); }
  catch (error) { say(`Could not load prices: ${error.message}`, true); return; }

  $('p-currency').textContent =
    `Currency: ${data.currency}. Token counting: ${data.token_estimator}.`;
  const tbody = $('pricing-rows');
  tbody.textContent = '';
  if (!data.models.length) {
    emptyRow(tbody, 5, 'No prices set. Savings will read zero until you add some.');
  } else {
    data.models.forEach((m) => {
      const tr = document.createElement('tr');
      rowHeader(tr, m.model, 'mono');
      cell(tr, String(m.input_per_1m), 'num');
      cell(tr, String(m.output_per_1m), 'num');
      cell(tr, m.cached_input_per_1m === null ? 'not set' : String(m.cached_input_per_1m), 'num');
      const actions = document.createElement('td');
      actions.appendChild(actionButton('Delete', `Delete pricing for ${m.model}`,
                                       () => deletePricing(m.model), true));
      tr.appendChild(actions);
      tbody.appendChild(tr);
    });
  }
  $('pricing-caption').textContent = `${data.models.length} model prices configured.`;
}

async function savePricing() {
  const model = $('p-model').value.trim();
  if (!model) { say('Enter a model name first.'); return; }
  try {
    await api('/api/pricing', {
      method: 'POST',
      body: JSON.stringify({
        model,
        input_per_1m: parseFloat($('p-in').value || '0'),
        output_per_1m: parseFloat($('p-out').value || '0'),
      }),
    });
    say(`Saved prices for ${model}.`);
    loadPricing();
    refresh();
  } catch (error) {
    say(`Could not save that price: ${error.message}`, true);
  }
}

async function deletePricing(model) {
  try {
    await api(`/api/pricing/${encodeURIComponent(model)}`, { method: 'DELETE' });
    say(`Removed prices for ${model}.`);
    loadPricing();
  } catch (error) {
    say(`Could not remove that price: ${error.message}`, true);
  }
}

/* ----------------------------------------------------------------- config */

const EDITABLE = [
  ['cache.enabled', 'bool', 'Caching turned on'],
  ['cache.default_ttl', 'int', 'Default lifetime, seconds'],
  ['cache.namespace', 'str', 'Namespace'],
  ['cache.fail_open', 'bool', 'Keep working if the cache breaks'],
  ['cache.cache_streaming', 'bool', 'Cache streamed replies'],
  ['cache.max_temperature', 'float', 'Highest temperature still cached'],
  ['cache.cache_zero_temperature_only', 'bool', 'Only cache deterministic requests'],
  ['policy.cache_responses_with_tools', 'bool', 'Cache replies that carry tool schemas (needed for agents)'],
  ['semantic.enabled', 'bool', 'Match similar questions'],
  ['semantic.threshold', 'float', 'How similar counts as a match, 0 to 1'],
  ['semantic.require_single_turn', 'bool', 'Similar matching only for single questions'],
  ['semantic.allow_tools', 'bool', 'Allow similar matching when tools are present'],
  ['privacy.store_prompts', 'bool', 'Keep prompt text'],
  ['privacy.store_responses', 'bool', 'Keep response text'],
  ['privacy.store_tool_arguments', 'bool', 'Keep tool arguments'],
  ['privacy.store_tool_results', 'bool', 'Keep tool results'],
  ['privacy.retention_days', 'int', 'History kept, days (0 keeps forever)'],
  ['concurrency.single_flight', 'bool', 'Merge duplicate simultaneous requests'],
  ['pricing.token_estimator', 'str', 'Token counting method'],
  ['pricing.chars_per_token', 'float', 'Characters per token when estimating'],
  ['pricing.currency', 'str', 'Currency label'],
];

async function loadConfig() {
  let cfg;
  let policy;
  try { [cfg, policy] = await Promise.all([api('/api/config'), api('/api/policy')]); }
  catch (error) { say(`Could not load the configuration: ${error.message}`, true); return; }

  $('config-json').textContent = JSON.stringify(cfg.config, null, 2);
  $('policy-json').textContent = JSON.stringify(policy, null, 2);

  const envRows = $('env-rows');
  envRows.textContent = '';
  cfg.env_docs.forEach((item) => {
    const tr = document.createElement('tr');
    rowHeader(tr, item.name, 'mono');
    cell(tr, item.description, 'muted');
    envRows.appendChild(tr);
  });

  const container = $('editable');
  container.textContent = '';
  EDITABLE.forEach(([path, kind, description]) => {
    const [section, key] = path.split('.');
    const value = (cfg.config[section] || {})[key];
    const id = `set-${path.replace(/\./g, '-')}`;

    const label = document.createElement('label');
    label.htmlFor = id;
    label.textContent = description;

    const wrapper = document.createElement('div');
    const input = document.createElement('input');
    input.id = id;
    input.dataset.path = path;
    input.dataset.kind = kind;
    if (kind === 'bool') {
      input.type = 'checkbox';
      input.checked = Boolean(value);
    } else {
      input.type = kind === 'str' ? 'text' : 'number';
      if (kind === 'float') input.step = 'any';
      input.value = value ?? '';
    }
    // The setting name itself is useful to a screen-reader user, so expose it
    // as description rather than hiding it in a tooltip.
    const help = document.createElement('span');
    help.id = `${id}-help`;
    help.className = 'sr-only';
    help.textContent = `Setting name ${path}`;
    input.setAttribute('aria-describedby', help.id);

    wrapper.append(input, help);
    container.append(label, wrapper);
  });
}

async function saveConfig() {
  const body = {};
  document.querySelectorAll('#editable [data-path]').forEach((element) => {
    body[element.dataset.path] =
      element.dataset.kind === 'bool' ? element.checked : element.value;
  });
  try {
    const result = await api('/api/config', { method: 'POST', body: JSON.stringify(body) });
    const applied = Object.keys(result.applied).length;
    const rejected = Object.keys(result.rejected);
    let message = `Applied ${applied} setting${applied === 1 ? '' : 's'}.`;
    if (rejected.length) message += ` Rejected: ${rejected.join(', ')}.`;
    $('cfg-result').textContent = message;
    say(message);
    loadConfig();
  } catch (error) {
    say(`Could not save the configuration: ${error.message}`, true);
  }
}

/* ------------------------------------------------------------------ wiring */

$('refresh-now').addEventListener('click', () => refresh({ announce: true }));

$('auto-toggle').addEventListener('click', () => {
  const button = $('auto-toggle');
  const on = button.getAttribute('aria-pressed') === 'true';
  const next = !on;
  button.setAttribute('aria-pressed', String(next));
  $('auto-state').textContent = next ? 'on' : 'off';
  if (next) {
    timer = window.setInterval(refresh, REFRESH_MS);
    say('Auto-refresh on, every five seconds.');
  } else {
    window.clearInterval(timer);
    timer = null;
    say('Auto-refresh off. Use Refresh now when you want fresh numbers.');
  }
});

$('req-load').addEventListener('click', loadRequests);
$('cache-search-btn').addEventListener('click', loadCache);
$('cache-search').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadCache(); });
$('inv-model-btn').addEventListener('click',
  () => invalidate({ model: $('inv-model').value.trim() }, `for that model`));
$('inv-tool-btn').addEventListener('click',
  () => invalidate({ tool: $('inv-tool').value.trim() }, `for that tool`));
$('inv-ns-btn').addEventListener('click',
  () => invalidate({ namespace: $('inv-ns').value.trim() }, `in that namespace`));
$('clear-all-btn').addEventListener('click', () => {
  if (!window.confirm('Delete every cached response and tool result?')) return;
  invalidate({ all: true }, 'in total');
});
$('tool-reload').addEventListener('click', loadTools);
$('tool-check-btn').addEventListener('click', checkToolPolicy);
$('tool-check').addEventListener('keydown', (e) => { if (e.key === 'Enter') checkToolPolicy(); });
$('pricing-save').addEventListener('click', savePricing);
$('config-save').addEventListener('click', saveConfig);

/* If someone has asked the OS to reduce motion, they probably don't want the
 * page rewriting itself either. Start paused and let them opt in. */
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
refresh();
if (reduceMotion) {
  $('auto-toggle').setAttribute('aria-pressed', 'false');
  $('auto-state').textContent = 'off';
} else {
  timer = window.setInterval(refresh, REFRESH_MS);
}
