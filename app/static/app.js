"use strict";

const $ = id => document.getElementById(id);
const create = (tag, text = "", className = "") => {
  const item = document.createElement(tag);
  item.textContent = text;
  if (className) item.className = className;
  return item;
};
const label = value => String(value ?? "Unavailable").replaceAll("_", " ");
const titleCase = value => label(value).replace(/^./, first => first.toUpperCase());
const numeric = (value, suffix = "") => {
  if (value == null) return "Unavailable";
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString(undefined, { maximumFractionDigits: 2 }) + suffix : "Unavailable";
};

let active = null;
let formType = null;
let conversations = [];
let companies = [];
let openRequest = 0;
let formRevision = 0;
let newestPanelRequest = 0;
const refreshVersions = new Map();
const threads = new Map();
const pending = new Map();
const recentProgress = new Map();
const errors = new Map();
const polls = new Map();
const previousPages = new Map();

function safeLink(text, href) {
  try {
    const url = new URL(href);
    if (!["https:", "http:"].includes(url.protocol) || url.username || url.password) return create("span", text);
    const anchor = create("a", text);
    anchor.href = url.href;
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    return anchor;
  } catch {
    return create("span", text);
  }
}

function inlineMarkdown(parent, text) {
  const tokens = /\*\*([^*\n]+)\*\*|`([^`\n]+)`|\[([^\]\n]+)\]\(([^\s)]+)\)/g;
  let last = 0;
  for (const match of text.matchAll(tokens)) {
    parent.append(document.createTextNode(text.slice(last, match.index)));
    parent.append(match[1] ? create("strong", match[1]) : match[2] ? create("code", match[2]) : safeLink(match[3], match[4]));
    last = match.index + match[0].length;
  }
  parent.append(document.createTextNode(text.slice(last)));
}

function markdown(content) {
  const bubble = create("div", "", "bubble assistant");
  const lines = String(content).split("\n");
  let list = null;
  for (const line of lines) {
    const bullet = /^\s*([-*]|\d+\.)\s+(.+)$/.exec(line);
    if (bullet) {
      const tag = bullet[1].endsWith(".") ? "ol" : "ul";
      if (!list || list.localName !== tag) {
        list = create(tag);
        bubble.append(list);
      }
      const entry = create("li");
      inlineMarkdown(entry, bullet[2]);
      list.append(entry);
    } else {
      list = null;
      const heading = /^#{1,6}\s+(.+)$/.exec(line);
      if (heading) {
        const item = create("strong");
        inlineMarkdown(item, heading[1]);
        bubble.append(item);
      } else {
        inlineMarkdown(bubble, line);
      }
      if (line !== lines.at(-1) || lines.length > 1) bubble.append(document.createTextNode("\n"));
    }
  }
  return bubble;
}

function details(parent, title) {
  const section = create("details");
  section.append(create("summary", title));
  parent.append(section);
  return section;
}

function items(parent, values) {
  if (!Array.isArray(values) || !values.length) return;
  const list = create("ul");
  for (const value of values) list.append(create("li", typeof value === "string" ? value : JSON.stringify(value)));
  parent.append(list);
}

function displayActivity(parent, body) {
  if (!body?.gate) return;
  const section = details(parent, "Customer activity and derived metrics");
  const metrics = body.gate.metrics || {};
  items(section, [
    "Gate: " + titleCase(body.gate.status),
    "Indicative annualized GBV (USD): " + numeric(metrics.annualized_gbv),
    "Estimated adoption: " + numeric(metrics.adoption_rate_percent, "%"),
    "Monthly support requests per 100 active users: " + numeric(metrics.monthly_support_per_100_active_users),
  ]);
  const submitted = body.submitted_activity || {};
  items(section, Object.entries(submitted).filter(([key]) => !["company_name", "company_domain", "current_tier"].includes(key))
    .map(([key, value]) => `${titleCase(key)}: ${key.endsWith("_trend") ? titleCase(value) : numeric(value)}`));
  items(section, body.gate.triggered_conditions?.map(titleCase));
  items(section, body.gate.information_gaps);
}

function evaluation(message, conversationId) {
  const metadata = message.metadata;
  const body = metadata.result || {};
  const decision = body.recommendation || {};
  const card = create("article", "", "result");
  const kind = body.evaluation_type || (metadata.activity_id ? "re_evaluation" : "lead_submit");
  const company = companies.find(item => item.id === metadata.company_id);
  card.append(
    create("div", kind === "lead_submit" ? "LEAD EVALUATION" : "CUSTOMER RE-EVALUATION", "muted"),
    create("h2", body.company_name || company?.name || "Company evaluation"),
    create("strong", `Tier ${numeric(body.current_tier)} · ${titleCase(decision.outcome)}${decision.recommended_tier == null ? "" : " Tier " + decision.recommended_tier}`),
    create("div", `${titleCase(decision.confidence)} confidence · ${titleCase(metadata.decision_status)}`, "muted"),
    create("h4", "Recommendation rationale"),
    create("p", decision.rationale || message.content)
  );

  if (body.predictions || decision.supporting_evidence?.length) {
    const section = details(card, "Predictions and evidence");
    const predictions = body.predictions || {};
    const leadPredictions = [
      ["travel_spend_potential", "Travel demand"], ["growth_potential", "Growth potential"],
      ["expected_support_needs", "Support needs"],
    ];
    for (const [key, title] of leadPredictions) {
      const prediction = predictions[key];
      if (!prediction) continue;
      section.append(create("h4", `${title} · ${titleCase(prediction.level)} · ${label(prediction.confidence)} confidence`),
        create("p", prediction.rationale || ""));
      items(section, prediction.supporting_signals?.map(titleCase));
      items(section, prediction.information_gaps);
    }
    if (predictions.findings) {
      section.append(create("h4", `Activity assessment · ${titleCase(predictions.confidence)} confidence`),
        create("p", predictions.rationale || ""));
      items(section, predictions.findings.map(row => `${(row.evidence || []).map(titleCase).join(", ")}. ${row.interpretation || ""}`));
      items(section, predictions.information_gaps);
    }
    if (decision.supporting_evidence?.length) {
      section.append(create("h4", "Supporting evidence"));
      items(section, decision.supporting_evidence.map(titleCase));
    }
    items(section, decision.key_findings);
  }
  if (body.research?.signals?.length) {
    const section = details(card, "Research findings and sources");
    for (const signal of body.research.signals) {
      section.append(create("h4", `${titleCase(signal.signal)} · ${titleCase(signal.status)}`));
      for (const evidence of signal.evidence || []) {
        const range = evidence.estimated_range;
        const value = range ? `${range.minimum}–${range.maximum} ${range.unit}` : evidence.value;
        section.append(create("p", `${value ?? "Unavailable"} · ${label(evidence.reliability)} reliability · ${label(evidence.evidence_type)}`));
        if (evidence.context) section.append(create("p", evidence.context, "muted"));
        for (const source of evidence.sources || []) {
          const row = create("p");
          row.append(safeLink(source.title || source.url, source.url));
          if (source.publication_date) row.append(document.createTextNode(" · " + source.publication_date));
          section.append(row);
        }
      }
      if (signal.information_gap) section.append(create("p", signal.information_gap, "muted"));
    }
  }
  if (decision.information_gaps?.length || decision.follow_up_actions?.length) {
    const section = details(card, "Information gaps and follow-up actions");
    if (decision.information_gaps?.length) {
      section.append(create("h4", "Information gaps"));
      items(section, decision.information_gaps);
    }
    if (decision.follow_up_actions?.length) {
      section.append(create("h4", "Follow-up actions"));
      items(section, decision.follow_up_actions);
    }
  }
  displayActivity(card, body);

  if (metadata.decision_status === "pending" && metadata.approval_available && metadata.evaluation_id) {
    const actions = create("div", "", "actions");
    for (const accept of [true, false]) {
      const button = create("button", accept ? "Accept" : "Reject", accept ? "primary" : "");
      button.disabled = isBusy(conversationId);
      button.onclick = () => run(conversationId, () => APIClient.decide(conversationId, metadata.evaluation_id, accept));
      actions.append(button);
    }
    card.append(actions);
  }
  return card;
}

function isBusy(id) {
  return pending.has(id) || Boolean(threads.get(id)?.active);
}

function drawPanels() {
  $("threads").replaceChildren(...conversations.map(thread => {
    const button = create("button", thread.title, thread.id === active ? "active" : "");
    button.onclick = () => selectThread(thread.id).catch(showGlobalError);
    return button;
  }));
  $("count").textContent = "· " + companies.length;
  $("companies").replaceChildren(...(companies.length ? companies.map(company => {
    const row = create("div", "", "company");
    row.append(create("span", company.name), create("span", "Tier " + company.tier, "tier"));
    return row;
  }) : [create("p", "No companies yet.", "muted")]));
}

function drawThread(scroll = true) {
  const id = active;
  const current = threads.get(id);
  const area = $("messages");
  $("title").textContent = current?.title || "";
  $("chat").querySelector("button").disabled = !id || isBusy(id);
  $("lead").disabled = !id || isBusy(id);
  $("customer").disabled = !id || isBusy(id);
  const error = errors.get(id);
  const errorArea = $("chatError");
  errorArea.hidden = !error;
  errorArea.replaceChildren();
  if (error) {
    errorArea.append(create("p", error.message));
    const recover = create("button", error.retry ? "Retry" : "Refresh conversation");
    recover.disabled = isBusy(id) && Boolean(error.retry);
    recover.onclick = () => error.retry
      ? run(id, progress => APIClient.retry(id, error.retry, progress))
      : refresh(id).catch(showGlobalError);
    errorArea.append(recover);
  }
  if (!current) return;
  const expanded = Array.from(area.querySelectorAll("details")).map(item => item.open);
  const position = area.scrollTop;
  const older = previousPages.get(id) || [];
  const messages = [...older, ...current.messages];
  const rows = messages.map(message => {
    if (message.metadata?.kind === "evaluation_result") return evaluation(message, id);
    if (message.role === "user") {
      const bubble = create("div", message.content, "bubble user");
      if (!isBusy(id) && message.metadata?.kind === "chat" && !messages.some(row => row.metadata?.reply_to === message.id)
          && !error?.retry) {
        const retry = create("button", "Retry reply");
        retry.onclick = () => run(id, progress => APIClient.retry(id, { kind: "chat", message_id: message.id }, progress));
        bubble.append(document.createElement("br"), retry);
      }
      return bubble;
    }
    const bubble = markdown(message.content);
    const metadata = message.metadata || {};
    if (metadata.kind === "activity_update") displayActivity(bubble, metadata.result);
    if (metadata.kind === "workflow_failure" && !messages.some(row => row.metadata?.retried_failure_id === message.id)
        && metadata.result?.can_retry && error?.retry?.message_id !== message.id) {
      const retry = create("button", "Retry evaluation");
      retry.disabled = isBusy(id);
      retry.onclick = () => run(id, progress => APIClient.retry(id, { kind: "workflow", message_id: message.id }, progress));
      bubble.append(document.createElement("br"), retry);
    }
    return bubble;
  });
  if ((older.length === 0 && current.messages.length === 100) || older.length && older.length % 100 === 0) {
    const more = create("button", "Load earlier messages");
    more.onclick = async () => {
      more.disabled = true;
      try {
        const page = await APIClient.getConversation(id, messages[0].id);
        if (!page.messages.length) { more.remove(); return; }
        previousPages.set(id, [...page.messages, ...older]);
        if (active === id) drawThread(false);
      } catch (err) { showGlobalError(err); }
    };
    rows.unshift(more);
  }
  if (!rows.length) {
    const empty = create("div", "", "empty");
    empty.append(create("h1", "Make better tier decisions."),
      create("p", "Submit a lead or customer activity to get started, or ask the assistant a question about your companies."));
    rows.push(empty);
  }
  const operation = pending.get(id);
  if (operation || current.active) {
    rows.push(operationProgress(operation));
  } else if (recentProgress.has(id)) {
    const previous = recentProgress.get(id);
    const record = create("details", "", "progress progress-record");
    record.append(create("summary", previous.outcome));
    record.append(progressList(previous));
    rows.push(record);
  }
  area.replaceChildren(...rows);
  Array.from(area.querySelectorAll("details")).forEach((item, index) => { item.open = Boolean(expanded[index]); });
  area.scrollTop = scroll ? area.scrollHeight : position;
}

function showGlobalError(error) {
  errors.set(active, error);
  drawThread(false);
}

async function refresh(id) {
  const version = ++openRequest;
  refreshVersions.set(id, version);
  const [conversation, list, portfolio] = await Promise.all([
    APIClient.getConversation(id), APIClient.listConversations(), APIClient.listCompanies(),
  ]);
  if (refreshVersions.get(id) !== version) return;
  const older = previousPages.get(id);
  if (older?.length) {
    const previous = threads.get(id)?.messages || [];
    const currentIds = new Set(conversation.messages.map(message => message.id));
    previousPages.set(id, [...older, ...previous].filter(message => !currentIds.has(message.id)));
  }
  threads.set(id, conversation);
  if (version > newestPanelRequest) {
    newestPanelRequest = version;
    conversations = list;
    companies = portfolio;
    drawPanels();
  }
  if (active === id) drawThread();
  if (conversation.active && !pending.has(id)) poll(id);
}

function poll(id) {
  if (polls.has(id)) return;
  polls.set(id, setTimeout(async () => {
    polls.delete(id);
    try { await refresh(id); } catch (error) { if (id === active) showGlobalError(error); }
  }, 2000));
}

async function selectThread(id) {
  active = id;
  try { localStorage.setItem("navan.activeConversation", id); } catch {}
  drawPanels();
  if (!threads.has(id)) $("messages").replaceChildren(create("p", "Loading conversation…", "muted"));
  drawThread();
  await refresh(id);
}

const stageEvents = {
  company_registered: ["company", "Save company information", "done"],
  initial_tier_assigned: ["tier", "Assign initial tier", "done"],
  customer_registered: ["customer", "Save customer details", "done"],
  customer_updated: ["customer", "Save customer details", "done"],
  customer_activity_persisted: ["activity", "Save customer activity", "done"],
  reevaluation_gate_assessed: ["gate", "Check re-evaluation triggers", "done"],
  research_started: ["research", "Research company", "running"],
  research_completed: ["research", "Research company", "done"],
  prediction_started: ["prediction", "Assess the evidence", "running"],
  prediction_completed: ["prediction", "Assess the evidence", "done"],
  recommendation_started: ["recommendation", "Prepare tier recommendation", "running"],
  recommendation_completed: ["recommendation", "Prepare tier recommendation", "done"],
  evaluation_completed: ["evaluation", "Save evaluation", "done"],
  chat_response_streaming: ["response", "Write response", "running"],
};
const committed = new Set(["company_registered", "customer_activity_persisted", "evaluation_completed"]);
const stageStatus = { running: "In progress", done: "Complete", failed: "Failed", interrupted: "Not confirmed" };

function updateStage(operation, key, title, state) {
  const previous = operation.stages.get(key);
  if (previous?.state === state) return;
  operation.stages.set(key, { title: previous?.title || title, state });
  operation.revision += 1;
}

function finishRunning(operation, state) {
  for (const [key, step] of operation.stages) {
    if (step.state === "running") updateStage(operation, key, step.title, state);
  }
}

function applyProgress(operation, event) {
  const stage = stageEvents[event.kind];
  if (stage) {
    updateStage(operation, ...stage);
    if (["research_completed", "prediction_completed", "recommendation_completed"].includes(event.kind)) {
      const correction = operation.stages.get("correction:" + event.component);
      if (correction) updateStage(operation, "correction:" + event.component, correction.title, "done");
    }
    if (event.kind === "recommendation_completed") updateStage(operation, "evaluation", "Save evaluation", "running");
  } else if (event.kind === "web_search_started" || event.kind === "web_search_completed") {
    const complete = event.kind === "web_search_completed";
    const key = "search:" + (event.tool_call_id || event.completed_tool_calls);
    const previous = operation.stages.get(key);
    const number = previous ? null : [...operation.stages.keys()].filter(step => step.startsWith("search:")).length + 1;
    updateStage(operation, key, previous?.title || "Search public sources · " + number, complete ? "done" : "running");
  } else if (event.kind === "chat_tool_started" || event.kind === "chat_tool_completed") {
    const state = event.kind === "chat_tool_started" ? "running" : event.tool_status === "completed" ? "done" : "failed";
    updateStage(operation, "tool:" + event.tool_call_id, titleCase(event.tool_name || "Look up workspace information"), state);
  } else if (event.kind === "output_correction_started") {
    updateStage(operation, "correction:" + event.component, "Correct an inconsistent result", "running");
  } else if (event.kind === "evaluation_failed") {
    if (event.workflow_stage && ![...operation.stages.values()].some(step => step.state === "running")) {
      updateStage(operation, "failure", titleCase(event.workflow_stage), "failed");
    }
    finishRunning(operation, "failed");
  }
}

function progressList(operation) {
  const list = create("ol", "", "progress-steps");
  const steps = operation?.stages.size ? operation.stages.values() : [{ title: "Waiting for the server…", state: "running" }];
  for (const { title, state } of steps) {
    const row = create("li", "", "progress-step progress-step--" + state);
    const icon = create("span", { done: "✓", failed: "×", interrupted: "–" }[state] || "", "step-icon");
    icon.setAttribute("aria-hidden", "true");
    row.append(icon, create("span", title), create("span", stageStatus[state], "step-status"));
    list.append(row);
  }
  return list;
}

function operationProgress(operation) {
  const progress = create("div", "", "progress");
  progress.id = "operationProgress";
  progress.append(create("p", "In progress", "progress-heading"), progressList(operation));
  const preview = create("div", operation?.preview || "", "bubble assistant");
  preview.id = "operationPreview";
  preview.hidden = !operation?.preview;
  progress.append(preview);
  return progress;
}

function updateProgress(id, operation, event) {
  if (event.conversation_id && event.conversation_id !== id) return;
  const before = operation.revision;
  applyProgress(operation, event);
  if (event.response_delta) {
    operation.raw += event.response_delta;
    const match = /"answer"\s*:\s*"((?:\\.|[^"\\])*)/.exec(operation.raw);
    if (match) {
      try { operation.preview = JSON.parse('"' + match[1] + '"'); } catch {}
    }
  }
  if (active === id) {
    const current = $("operationProgress");
    if (current) {
      if (operation.revision !== before) {
        current.querySelector(".progress-steps").replaceWith(progressList(operation));
      }
      const preview = $("operationPreview");
      preview.textContent = operation.preview;
      preview.hidden = !operation.preview;
      $("messages").scrollTop = $("messages").scrollHeight;
    } else drawThread();
  }
  if (committed.has(event.kind)) refresh(id).catch(() => {});
}

async function run(id, execute, { onProgress, onFailure } = {}) {
  if (!id || isBusy(id)) return;
  const operation = { stages: new Map(), revision: 0, raw: "", preview: "", outcome: "" };
  pending.set(id, operation);
  recentProgress.delete(id);
  errors.delete(id);
  if (active === id) drawThread();
  try {
    await execute(event => {
      onProgress?.(event);
      updateProgress(id, operation, event);
    });
    finishRunning(operation, "done");
    operation.outcome = "Completed steps";
  } catch (error) {
    const interrupted = error.code === "connection_lost";
    finishRunning(operation, interrupted ? "interrupted" : "failed");
    operation.outcome = interrupted ? "Steps received before interruption" : "Steps before the error";
    errors.set(id, error);
    onFailure?.(error);
  } finally {
    if (operation.stages.size) recentProgress.set(id, operation);
    pending.delete(id);
    if (threads.has(id)) threads.get(id).active = false;
    try { await refresh(id); } catch (error) {
      if (!errors.has(id)) errors.set(id, error);
      if (active === id) drawThread();
    }
  }
}

function field(name, title, type, options = {}) {
  const wrapper = create("label", title + (options.required ? " *" : ""));
  const input = create("input");
  Object.assign(input, { name, type, ...options });
  wrapper.append(input);
  return wrapper;
}

function select(name, title, values, explicit = false) {
  const wrapper = create("label", title);
  const input = create("select");
  input.name = name;
  input.required = explicit;
  if (explicit) {
    const option = create("option", "Select a tier");
    option.value = "";
    option.disabled = true;
    option.selected = true;
    input.append(option);
  }
  for (const value of values) {
    const option = create("option", titleCase(value));
    option.value = value;
    input.append(option);
  }
  wrapper.append(input);
  return wrapper;
}

function openForm(type) {
  formType = type;
  formRevision += 1;
  $("submission").querySelector('button[type="submit"]').disabled = false;
  $("modalTitle").textContent = type === "lead" ? "Submit Lead" : "Submit Customer Activity";
  $("description").textContent = type === "lead"
    ? "Enter initial company information to start a lead evaluation."
    : "Submit an aggregate activity snapshot for a new or existing customer.";
  const fields = [
    field("name", "Company name", "text", { required: true, maxLength: 200 }),
    field("domain", "Company domain", "text", { required: true, maxLength: 253, placeholder: "example.com" }),
  ];
  if (type === "lead") {
    fields.push(field("gbv", "Annual GBV (USD)", "number", { required: true, min: "0", step: "0.01" }),
      field("employee_count", "Employee count", "number", { min: "1", step: "1" }));
  } else {
    fields.push(select("current_tier", "Current tier *", ["1", "2", "3"], true),
      field("months", "Months since signup", "number", { required: true, min: "1", step: "1" }),
      field("total_gbv", "Total GBV since signup (USD)", "number", { min: "0", step: "0.01" }),
      field("avg_monthly_bookings", "Average monthly bookings", "number", { min: "0", step: "0.01" }),
      field("employee_count", "Current employee count", "number", { min: "1", step: "1" }),
      field("avg_monthly_active_users", "Average monthly active users", "number", { min: "0", step: "0.01" }),
      field("avg_monthly_support_requests", "Average monthly support requests", "number", { min: "0", step: "0.01" }));
    for (const key of ["travel_spend", "company_size", "product_adoption"]) {
      fields.push(select(key + "_trend", titleCase(key) + " trend", ["unknown", "increasing", "stable", "decreasing"]));
    }
  }
  $("fields").replaceChildren(...fields);
  $("formError").textContent = "";
  $("modal").showModal();
}

function formPayload(type, form) {
  const values = Object.fromEntries(new FormData(form));
  const integer = key => values[key] === "" ? null : Number(values[key]);
  const payload = { company_name: values.name, company_domain: values.domain, employee_count: integer("employee_count") };
  if (type === "lead") return { ...payload, submitted_annual_gbv: values.gbv };
  for (const key of ["total_gbv", "avg_monthly_bookings", "avg_monthly_active_users", "avg_monthly_support_requests"]) {
    payload[key] = values[key] === "" ? null : values[key];
  }
  return {
    ...payload, current_tier: integer("current_tier"), months_since_signup: integer("months"),
    travel_spend_trend: values.travel_spend_trend, company_size_trend: values.company_size_trend,
    product_adoption_trend: values.product_adoption_trend,
  };
}

function setHistory(open) {
  $("history").hidden = !open;
  document.querySelector(".layout").classList.toggle("history-open", open);
  const toggle = $("historyToggle");
  toggle.textContent = open ? "‹" : "›";
  toggle.setAttribute("aria-expanded", String(open));
  toggle.title = open ? "Hide conversation history" : "Show conversation history";
  toggle.setAttribute("aria-label", toggle.title);
}
$("historyToggle").onclick = () => setHistory($("history").hidden);
$("history").onclick = event => { if (event.target.closest("button")) setHistory(false); };
document.addEventListener("keydown", event => { if (event.key === "Escape") setHistory(false); });

$("new").onclick = async () => {
  $("new").disabled = true;
  try { await selectThread((await APIClient.createConversation()).id); }
  catch (error) { showGlobalError(error); }
  finally { $("new").disabled = false; }
};
$("example").onclick = () => {
  $("question").value = "Why was an upgrade recommended?";
  $("question").focus();
};
$("chat").onsubmit = async event => {
  event.preventDefault();
  const id = active;
  const question = $("question").value.trim();
  if (!question || !id || isBusy(id)) return;
  $("question").value = "";
  await run(id, progress => APIClient.chat(id, question, progress));
};
$("lead").onclick = () => openForm("lead");
$("customer").onclick = () => openForm("customer");
$("close").onclick = $("cancel").onclick = () => $("modal").close();
$("submission").onsubmit = async event => {
  event.preventDefault();
  const id = active;
  const type = formType;
  if (!id || isBusy(id)) {
    $("formError").textContent = "An operation is already running in this conversation.";
    return;
  }
  const payload = formPayload(type, event.currentTarget);
  const button = event.currentTarget.querySelector('button[type="submit"]');
  button.disabled = true;
  $("formError").textContent = "";
  const submitForm = event.currentTarget;
  const modal = $("modal");
  const revision = formRevision;
  try {
    await run(id, progress => APIClient.submit(id, type, payload, progress), {
      onProgress(item) {
        if (item.kind === "company_registered" || item.kind === "customer_activity_persisted") {
          if (revision === formRevision && modal.open) modal.close();
        }
      },
      onFailure(error) {
        if (revision === formRevision && modal.open) $("formError").textContent = error.message;
      },
    });
    if (revision === formRevision && !errors.has(id) && modal.open) modal.close();
  } finally {
    if (revision === formRevision) submitForm.querySelector('button[type="submit"]').disabled = false;
  }
};

async function initialize() {
  $("lead").disabled = $("customer").disabled = $("chat").querySelector("button").disabled = true;
  try {
    conversations = await APIClient.listConversations();
    let remembered = null;
    try { remembered = localStorage.getItem("navan.activeConversation"); } catch {}
    const available = conversations.find(row => row.id === remembered) || conversations[0];
    await selectThread(available?.id || (await APIClient.createConversation()).id);
  } catch (error) {
    const recovery = create("button", "Retry loading workspace");
    recovery.onclick = initialize;
    $("messages").replaceChildren(create("p", error.message, "error"), recovery);
  }
}
initialize();
