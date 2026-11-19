# Navan tier recommendation backend

The Flask backend uses the existing PostgreSQL database for the business policy,
prompts and workspace data. Stage 2 adds deterministic lead registration,
customer activity assessment and human approval. Stage 3 adds shared OpenAI
Responses infrastructure for later workflow integration.

## Project structure

```text
app/
  __init__.py        Flask application factory and GET /health
  database.py        PostgreSQL engine and session factory
  models.py          Explicit ORM mappings for the ten public tables
  reference_data.py  Business policy and prompt queries
  workspaces.py      Signed browser workspace resolution
  data_access.py     Workspace-scoped company and related-record access
  validation.py      Typed business inputs and validation
  tier_eligibility.py Shared Section 1 thresholds
  orchestrator.py    Lead registration and customer activity transactions
  reevaluation_gate.py Section 6 snapshot metrics and triggers
  human_approval.py  Recommendation decisions and supersession
  ai/
    client.py        Responses API, final parsing, errors and telemetry
    config.py        Per-component model, reasoning and execution settings
    prompts.py       Stored templates and explicit evaluation context
    schemas.py       Structured outputs and existing evaluation-field mapping
    tools.py         Hosted search and authorized read-only function dispatch
    progress.py      Typed progress events
tests/
  test_app.py        Local unit tests
  test_database.py   Read-only regression tests against PostgreSQL
  test_business_rules.py Policy boundaries, missing data and validation
  test_workspace_workflows.py Workspace and transaction tests on local PostgreSQL
  test_ai.py         Mocked OpenAI requests, streaming, validation and tooling
scripts/
  smoke_ai.py        Optional live OpenAI verification
.env.example         Environment variable template
requirements.txt     Runtime dependencies
requirements-dev.txt Test dependencies
pytest.ini           Integration marker registration
```

## Local setup

Use Python 3.11 or newer. Run the commands from the project directory.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

If you do not already have a local `.env`, copy `.env.example` to `.env` and
set `DATABASE_URL` to the existing PostgreSQL connection URL. Keep credentials
in that ignored file or in the process environment. The process environment
takes precedence, so Render can supply its connection URL directly. The app
uses psycopg 3 and defaults to TLS via `sslmode=require`. An explicit
`sslmode` in the connection URL overrides that default.

```sh
flask --app app run
curl -i http://127.0.0.1:5000/health
```

The health endpoint returns HTTP 200 with `{"status":"ok"}` when PostgreSQL
responds to a connectivity check, or HTTP 503 with
`{"status":"unavailable"}` when the database check fails. Missing or invalid
database configuration stops application initialization.

### Demo company templates

Before opening the demo, the database owner should provision
`public.demo_companies` and import the assignment's four sample rows
(Gradient Labs, FintechOS, NextGen Cloud, and Valarian). Render already has
this table. For a fresh local database, use the following schema:

```sql
CREATE TABLE public.demo_companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    domain TEXT NOT NULL UNIQUE,
    submitted_annual_gbv NUMERIC(15,2) NOT NULL,
    employee_count INTEGER,
    current_tier INTEGER NOT NULL CHECK (current_tier >= 1 AND current_tier <= 3)
);
```

The runtime database role needs `SELECT` on this table. Keep template writes
with the database owner or setup role. Each new workspace copies the templates
as regular customers with new IDs and the sample tiers, in the workspace
creation transaction. Existing workspaces keep their records as-is. The
sample figures are submitted data and initialization creates no research,
activity, evaluation or tier-change history. An empty template table
produces an empty workspace, so check that the four source rows are present
before opening the demo.

## Gunicorn

Install `requirements.txt` for a runtime-only environment. Gunicorn uses the
same factory. The bind uses the platform's `PORT` when set.

```sh
gunicorn --bind "0.0.0.0:${PORT:-8000}" 'app:create_app()'
```

## Tests

```sh
python -m pytest -m 'not integration and not write_integration' -q
python -m pytest -m integration -q
```

The first command runs without a database. The second uses `DATABASE_URL`
and enforces PostgreSQL read-only transactions. It compares the models with
the live schema and verifies that the policy and eight prompts can be read.

The write tests require a dedicated, disposable local database named
`navan_stage2_test`. They provision ORM tables only there and roll back each
test's records. Set `TEST_DATABASE_URL` explicitly to a local PostgreSQL URL
for that database. The fixture skips these tests when the variable is absent.

```sh
# On a local PostgreSQL instance configured for your user:
createdb navan_stage2_test
export TEST_DATABASE_URL='postgresql://localhost/navan_stage2_test'
python -m pytest -m write_integration -q
python -m pytest -q
```

## Database access

The application owns a pooled engine and shares its session factory through
`app.extensions["db_sessions"]`. Use `with sessions() as session` for reads so
the session closes and releases the connection. Use
`with sessions.begin() as session` when a caller needs an atomic transaction.
That context commits on success, rolls back on error, and closes the session.
The database handles the declared foreign-key, default and identity behavior.

The current policy is the singleton row with `id = 1`. Missing policy and
prompt records raise `BusinessPolicyNotFound` and `PromptNotFound`,
respectively. `product_activity` stores aggregate snapshots. The composite
relationships from evaluations to research and from tier changes to
evaluations are read-only ORM lookups. Both join on company ID as well as the
referenced record ID.

## Workspace and Stage 2 entry points

Set `SECRET_KEY` to a stable, random value in the environment or private
`.env`. To generate one for a fresh local environment, run
`python -c 'import secrets; print(secrets.token_urlsafe(48))'` and copy the
value into `.env`. Workspace resolution requires the key. The health endpoint
remains independent of workspace state. The signed Flask session cookie uses
HttpOnly and SameSite=Lax. It is Secure by default on Render. Set
`SESSION_COOKIE_SECURE=true` for HTTPS elsewhere and leave it unset for local
HTTP development.

- `resolve_workspace()` validates the signed cookie's workspace against the
  database and creates and seeds one when needed. Call it in a Flask request and pass its
  returned ID to the business operations.
- `WorkspaceData` scopes company lookup and direct retrieval of companies,
  research, activity, evaluations, tier changes, conversations and messages.
- `Orchestrator.register_lead(workspace_id, lead)` validates the lead,
  calculates its Section 1 tier and reports whether registration created a
  company. It preserves an existing company's tier and customer status.
- `Orchestrator.submit_customer_activity(workspace_id, activity)` records the
  submitted current tier, appends an activity snapshot, supersedes pending
  recommendations and returns the gate result. These writes share a transaction.
  Use Decimal values for the financial and activity measurements.
- `ReevaluationGate.assess(activity)` applies policy Sections 6.1–6.2 to the
  submitted snapshot. The Section 1 thresholds are shared with lead
  registration. When eligibility data is incomplete, a mismatch fires only
  if the known inputs rule out the submitted tier. Stable trends and valid
  support metrics can still make the gate assessable.
- `HumanApproval.accept(workspace_id, evaluation_id)` uses a conditional
  status update and checks the company's current tier and newer activity.
  The tier update, decision, linked history row and supersession are atomic.
  `reject(workspace_id, evaluation_id)` records the decision only.
  `supersede_pending(data, company_id)` invalidates pending recommendations
  within the caller's transaction.

Company names match by normalized whitespace and Unicode casefolding inside
each workspace. Domains are ASCII DNS names and are stored in lowercase
without a terminal dot. A customer submission may refresh the matched
company's domain when it is available in that workspace. The submitted tier
is treated as an externally observed fact. For an existing company that tier
change is logged separately from an accepted recommendation.

## Stage 3 AI infrastructure

Set `OPENAI_API_KEY` in the process environment or the private local `.env`
before invoking AI. Flask initialization and the health endpoint work without
the key. All four components default to `gpt-6.2-astra` and Fast processing.
Research uses medium reasoning. Prediction, decision and chat use low.
For each component set `AI_<COMPONENT>_MODEL`, `_REASONING_EFFORT`,
`_SERVICE_TIER` or `_MAX_OUTPUT_TOKENS` to override its settings. Component
names are `RESEARCH`, `PREDICTION`, `DECISION` and `CHAT`.
`AI_SERVICE_TIER` sets the common processing preference, and setting it to
`default` requests Standard processing. The recorded actual tier comes from
the API response. The client performs no automatic tier or model fallback.

Load the active policy once per evaluation with `PromptRenderer.load_policy()`.
Pass that snapshot to each `PromptRenderer.render()` call along with the typed
inputs and component settings. The renderer loads the applicable stored
prompts and separates developer instructions from untrusted task data.
`AIClient.execute()` accepts the rendered request and a progress callback.
Evaluation requests are independent. Chat continuation is limited to the
current execution and requires server-resolved workspace context for tools.

Research uses hosted `web_search` with low search context, at most five
calls by default and the API's standard result-token budget. Generated-token
defaults are 8,000 for research and 2,000 each for prediction and decision,
including reasoning. Deterministic validation can trigger one tool-free
correction per stage. Lead stages preserve evidence order and use a shared
source list within each request. Customer stages send derived metrics once.

The lead and customer workflows share a deadline and an estimated spend
budget across stages and corrections. Configure `AI_EVALUATION_TIMEOUT_SECONDS`
(default 240) and `AI_EVALUATION_MAX_ESTIMATED_USD` (default $4). Before a request,
the client reserves for submitted input, its output token cap, and an estimated
hosted input allowance when search is enabled. It uses observed usage to update
that allowance after each request and retains the reservation when usage is
unavailable. The USD budget is an application guardrail. Hosted search tokens
can exceed an estimate on a request already in flight. Evaluation models and
tiers need pricing in `app/ai/budget.py` to pass this check.

The SDK network timeout defaults to 120 seconds and is shortened to the
remaining workflow time when needed. The streaming client also checks the
deadline as events arrive. An incomplete response, including an output-token
limit termination, raises a typed `AIExecutionError` before validation or
correction. SDK retries are disabled. Server logs retain per-request status,
token usage, estimated USD, and timings without prompt or customer content.

Each lead or customer evaluation writes a terminal summary and a JSON report
under `.evaluation-diagnostics`. Set `AI_DIAGNOSTICS_REPORT_DIR` to change the
directory or set it to an empty value to disable report files. Reports use
owner-only permissions for newly created directories and files. Configure
`AI_DIAGNOSTICS_LOG_LEVEL` as `DEBUG`, `INFO` (default), `WARNING` or `ERROR`.
DEBUG includes request and hosted search details. Reports keep those details
regardless of terminal verbosity. Each retry gets its own attempt ID under
the same correlation ID. The HTTP workflow keeps the ID in saved failure
metadata. Direct Orchestrator callers can pass the failure's
`evaluation_correlation_id` as `correlation_id` on retry.

Context sizes are measured in UTF-8 bytes. Token usage comes from the API
and USD costs are estimates. Hosted retrieval token counts are unavailable.
When the API exposes a query, the report records its byte length and a
keyed fingerprint comparable inside the current server process. Queries can
contain submitted company identifiers, so the reports contain no query text,
source URLs, request bodies, response text, or hidden reasoning.

Research output maps to `company_research.signals` via
`ResearchOutput.model_dump(mode="json")`. It represents all Section 2
signals, including missing and conflicting findings and source metadata.
`LeadPredictionOutput` captures the three predictions in Sections 3–4.
`ProductAssessmentOutput` cites current activity and backend metrics under
Section 6 without recalculating them. `RecommendationOutput` follows Sections
5 and 6. `evaluation_output_fields()` maps predictions and recommendation
evidence into the existing `predictions` JSONB and maps tier, confidence,
rationale, follow-ups and gaps to their existing columns. It leaves
persistence metadata to the caller. Snapshot downgrade validation requires
clarification of the policy's undefined sufficient-adoption criterion.

### Optional live OpenAI smoke test

The standard tests mock OpenAI responses and do not make paid calls.
To exercise the live API with a public company and synthetic lead details,
configure the key in the private environment and run:

```sh
AI_RESEARCH_MAX_TOOL_CALLS=2 python -m scripts.smoke_ai
```

The smoke test reads the policy and prompts from PostgreSQL in read-only
mode, makes one OpenAI research request, and reports streamed search progress,
completed calls, service tier, elapsed time and token usage.

## Stage 4 Lead Submission

From application code, pass a workspace ID resolved on the server and a
`LeadInput` to `Orchestrator(app.extensions["db_sessions"]).submit_lead(...)`.
A new lead returns a `LeadEvaluationResult` containing company details,
current and initial tier, validated research and sources, predictions,
recommendation, evaluation ID, approval availability and timing metadata.
A duplicate returns the Stage 2 `LeadResult` with `created=False` before any
AI calls. Use `retry_lead_evaluation(workspace_id, company_id)` to start a
fresh evaluation for an existing registered lead without resetting its tier
or invalidating an earlier recommendation. For retried leads, the original
tier comes from the first tier change, or the current tier when no change
has been recorded. On failure, `LeadWorkflowFailure` reports the stage,
committed IDs, available execution diagnostics and any validated predictions
that had only been held in memory.

The orchestrator commits registration, validated research, and the completed
evaluation separately. Policy and Lead Submit prompts are snapshotted before
the first request. Research, prediction and recommendation run sequentially
with no open workflow database transaction. Only validated tier changes can
be pending for `HumanApproval`. Retain and undetermined evaluations use
`not_applicable`. Pass `on_progress` to receive committed workflow milestones
and actual hosted-search events.

Stage 4 integration tests mock OpenAI and reuse the isolated local
`TEST_DATABASE_URL` setup above:

```sh
python -m pytest -q tests/test_lead_submission.py
python -m pytest -q
```

The opt-in live test reads the approved policy and prompts from
`DATABASE_URL` in read-only mode and persists public-company and synthetic
lead data only in a dedicated local `navan_stage4_smoke` database. Set
`OPENAI_API_KEY` in your private environment before running it.

```sh
createdb navan_stage4_smoke
export LEAD_SMOKE_DATABASE_URL='postgresql://localhost/navan_stage4_smoke'
python -m scripts.smoke_lead gitlab
python -m scripts.smoke_lead python
```

The live runner creates a fresh workspace for each attempt and outputs
progress events and a JSON report. Each invocation makes live OpenAI requests.

## Stage 5 Customer Re-evaluation

From application code, call
`Orchestrator(app.extensions["db_sessions"]).submit_customer_reevaluation(workspace_id, activity)`
with a workspace ID resolved on the server and a `CustomerActivityInput`.
The existing activity submission transaction registers or updates the
customer, saves the submitted snapshot, records an external tier change when
needed, and supersedes obsolete pending recommendations. It commits before
any policy lookup or OpenAI request.

- `NO_TRIGGER` and `INSUFFICIENT_DATA` return a successful
  `CustomerReevaluationResult` with the saved activity and deterministic gate
  explanation. These outcomes use zero AI calls and create no evaluation.
- `TRIGGERED` snapshots the policy and stored re-evaluation prompts, then
  calls the existing prediction and decision engines with the latest
  user-reported activity, derived metrics, trends, and gate findings.
  Both requests have tools disabled. The evaluation commits only after
  both responses pass validation, without changing the company tier.
- A proposed upgrade remains pending until explicitly handled by
  `HumanApproval`. Retain and undetermined outcomes use `not_applicable`.
  Downgrade recommendations are intentionally disabled because policy
  section 6.3 leaves sufficient adoption undefined.

`CustomerWorkflowFailure` reports the failed stage, committed IDs, gate
result, and safe execution diagnostics. Call
`retry_customer_reevaluation(workspace_id, activity_id)` to restart prediction
and recommendation for the same current saved snapshot. Retry rejects
superseded snapshots, later tier changes, completed evaluations, and snapshots
from other workspaces. Snapshot identity is recorded as
`predictions.activity_snapshot_id` in the existing JSONB column. The workflow
uses the existing activity timestamps and tier history for freshness.
If two snapshot timestamps tie, freshness cannot be established and the
affected evaluation is blocked. Pass `on_progress` for registration or update,
activity persistence, gate outcome, AI milestones, and final status.

Run the mocked Stage 5 tests using the isolated `TEST_DATABASE_URL` above:

```sh
python -m pytest -q tests/test_customer_reevaluation.py
python -m pytest -q
```

The live test writes synthetic records only to a dedicated local database.
It reads the approved policy and prompts from `DATABASE_URL` in read-only
mode for the two triggered scenarios. If the model produces an actionable
upgrade, the runner explicitly accepts it in the local database and verifies
the linked history record. Set `OPENAI_API_KEY` for the triggered scenarios.

```sh
createdb navan_stage5_smoke
export REEVALUATION_SMOKE_DATABASE_URL='postgresql://localhost/navan_stage5_smoke'
python -m scripts.smoke_reevaluation
```

Pass one or more of `no_trigger insufficient upgrade retain` to select cases.
Each case prints progress and a JSON report with its result and telemetry.

## Stage 6 Chat and conversations

Create `ChatAssistant(app.extensions["db_sessions"])` and pass the workspace
ID resolved on the server. The application selects a conversation ID using
`create_conversation()`, `list_conversations()` and `get_conversation()`.
Use `list_messages()` for chronological pages and `rename_conversation()`
for an explicit title. The first interaction supplies a deterministic title
for a new conversation.

- `send_message()` commits the user message, reads its bounded history, and
  commits only a validated final assistant response. After an AI failure,
  call `retry_message()` with the original user message ID.
- `submit_lead()` and `submit_customer_reevaluation()` persist the initiating
  operation and deterministic result in the selected conversation. Without
  an ID, they reuse the newest workspace conversation or create one when
  none exists. Progress remains routed to the captured conversation ID.
- `decide_recommendation()` is for explicit application controls.
  `record_decision_confirmation()` adds an idempotent confirmation for
  an already recorded acceptance or rejection. Conversation reads overlay
  the current evaluation status from PostgreSQL.
- `ConversationPersistenceError` contains a server-owned prepared message
  and, where applicable, the completed business result. Call
  `save_prepared_message()` to retry just that write. Do not retry the
  underlying workflow or approval after its result has committed.

Message metadata distinguishes ordinary chat, workflow requests,
evaluation results, activity updates, workflow failures, and approval
confirmations. Detailed research is saved in the message for future UI use.
AI history receives only the concise message and selected result fields.
Streamed chat deltas are provisional JSON fragments. Tool events and all
workflow progress carry the initiating conversation ID, while only final
messages persist.

The chat allowlist provides company lists, company details, research,
evaluations, customer activity, tier history, and the active policy. The
policy is fetched only through its read-only tool. Model arguments never
grant workspace access. Set `AI_CHAT_HISTORY_LIMIT`, `AI_CHAT_TOOL_BUDGET`,
and `AI_CHAT_MAX_ROUNDS` to override the defaults of 20 persisted messages,
six tool attempts, and eight request rounds. Invalid arguments consume an
attempt. The last request in a turn disables tools to allow a final answer.

```sh
python -m pytest -q tests/test_chat.py
python -m pytest -q
```

Use the dedicated local test database for mocked integration tests. To run
the live smoke test, configure `OPENAI_API_KEY` and a separate local database:

```sh
createdb navan_stage6_smoke
export CHAT_SMOKE_DATABASE_URL='postgresql://localhost/navan_stage6_smoke'
python -m scripts.smoke_chat
```

The live test seeds synthetic local records, reads the approved chat prompt
and policy from Render in read-only mode, and prints tool names, answers,
timings, and token counts. It checks that a chat request to approve the
pending recommendation leaves both the recommendation and tier unchanged.

## Stage 7B Browser demo

Open the Flask root URL to use the approved interface. Flask serves the
template and JavaScript on the same origin as the API. The first visit
creates a signed workspace and opens its newest conversation or creates one
when the workspace is empty. The browser remembers the selected conversation
and reloads company and message state from the server.

```sh
flask --app app run
```

For a streamed Gunicorn run, keep this demo on one process with spare
request threads. Leave the browser and API on the same origin and disable
proxy buffering for `text/event-stream` responses.

```sh
gunicorn --workers 1 --worker-class gthread --threads 8 --timeout 300 \
  --bind "0.0.0.0:${PORT:-8000}" 'app:create_app()'
```

The `/api/conversations` and `/api/companies` routes read workspace state.
Conversation-scoped routes submit chat, leads, customer activity, retry
requests, and explicit evaluation decisions. POST requests require JSON
and the `X-Navan-Request: 1` header. Origins are checked where supplied.
Set a stable `SECRET_KEY` and enable secure cookies for HTTPS environments.

Operation endpoints stream `progress`, followed by a single `result` or
`error` event. Use the latest conversation response for persisted state.
Closing a stream does not cancel the operation while the worker is running.
After a connection interruption, refresh the conversation before taking any
action. The application never automatically resubmits the original mutation.

The demo limits JSON bodies to 32 KiB, runs one operation per conversation,
and permits two operations per workspace and four per process. Prepared
message retry tokens expire after 15 minutes and are held in the serving
process. They retry a message write without running the business action
again. Active operation state and prepared messages do not survive worker
restart. Routing all browser requests to the single worker avoids splitting
this process-local state.

Run the HTTP integration tests with the same isolated local PostgreSQL
configuration used in earlier stages:

```sh
python -m pytest -q tests/test_http_api.py
python -m pytest -q
```

### Output correction

In evaluation workflows, research, prediction, and decision can each make
one tool-free correction request after a structurally valid result fails
deterministic validation. The request includes the original task context,
the invalid result, and the first failed rule with its field path. The same
validator checks the corrected result, and research correction cannot add
source URLs to a signal. A failed correction ends the affected stage.
Previously completed stages are not repeated. Transport, refusal, and
schema parsing failures follow the existing failure path.

The browser shows a correction progress event without exposing diagnostics.
Server logs and execution metadata record the initial rule, correction
outcome, and added latency.

## Final business policy

The final 19 November 2026 policy and eight matching prompt seeds are in
`seeds/business_prediction_policy.txt` and `seeds/prompts/`. Before deploying
this implementation, ensure the active `business_policy` row and all eight
`prompts` rows match these files. The workflows load the stored text directly.
The final six-signal outputs use the existing JSONB columns. Older research
records may still show the earlier signals in history. No data cleanup or
schema migration is required for the policy change.

The backend computes travel scale, coverage, weighted intensity and eligible
tier targets. One admitted core travel signal can support a provisional
prediction. Customer downgrades require at least three completed months,
known qualifying eligibility inputs and Medium confidence without material
eligibility uncertainty. Adoption remains contextual. Activity metrics such as
total GBV and average monthly bookings may be left blank as unknown. Both directions
require explicit approval and the existing freshness checks.

Run the revised tests and optional live smoke cases with isolated local PostgreSQL:

```sh
python -m pytest -q tests/test_revised_policy.py
createdb navan_policy_revised_smoke
export REVISED_POLICY_SMOKE_DATABASE_URL='postgresql://localhost/navan_policy_revised_smoke'
python -m scripts.smoke_revised_policy seed
python -m scripts.smoke_revised_policy google
python -m scripts.smoke_revised_policy gitlab
python -m scripts.smoke_revised_policy customer
```

The smoke script refuses nonlocal databases and refuses to overwrite a
different policy or prompt set. The three live cases require an OpenAI key.
