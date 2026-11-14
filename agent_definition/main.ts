/**
 * AgentDefinition: every option, in one runnable demo (Agent SDK 0.3.309)
 *
 * An AgentDefinition describes a subagent the built-in Agent (formerly Task) tool can run.
 * This demo sets EVERY field at least once, across four agents:
 *
 *   coordinator  main-thread agent (selected with options.agent). Delegates, never reads.
 *   code-scout   read-only explorer with its own MCP tool, memory and an observer.
 *   doc-writer   tool-less writer that inherits the main model.
 *   auditor      background observer attached to code-scout. It watches, never acts.
 *
 * Field                              | Set on       | What it does
 * -----------------------------------|--------------|-----------------------------------------------
 * description              (required)| all          | When to use this agent (the main agent reads it to pick)
 * prompt                   (required)| all          | The agent's system prompt
 * tools                              | all          | Allow-list of tools. Omit = inherit every parent tool
 * disallowedTools                    | code-scout   | Deny-list, applied after `tools`
 * model                              | all          | Alias ('haiku'), full id, or 'inherit' (= main model)
 * mcpServers                         | code-scout   | MCP servers for this agent: a name from options.mcpServers, or an inline config
 * criticalSystemReminder_EXPERIMENTAL| coordinator  | Extra reminder pinned to the system prompt
 * skills                             | doc-writer   | Skill names to preload into the agent's context
 * initialPrompt                      | coordinator  | First user turn, ONLY when the agent is the main thread
 * maxTurns                           | all          | Cap on agentic turns (API round trips)
 * autoCompactWindow                  | code-scout   | Token count at which a subagent compacts its own context
 * background                         | code-scout   | true = fire-and-forget; the caller does not wait
 * omitClaudeMd                       | code-scout   | Run without CLAUDE.md files; the delegation prompt is all it gets
 * memory                             | code-scout   | Auto-load agent memory: 'user' | 'project' | 'local'
 * effort                             | most         | Reasoning effort: 'low' ... 'max', or a number
 * permissionMode                     | coordinator, | How tool calls are approved: 'default' | 'acceptEdits' |
 *                                    | code-scout   | 'plan' | 'dontAsk' | 'bypassPermissions' | 'auto'
 * observer                           | code-scout   | Agent type spawned as a background observer whenever this agent runs
 * observerMessage                    | code-scout   | Extra text appended to each activity digest sent to the observer
 *
 * Run: npx tsx agent_definition/main.ts
 */
import {
  createSdkMcpServer,
  query,
  tool,
  type AgentDefinition,
  type SDKMessage,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createLogger } from "../lib/sdk_parser.ts";

process.loadEnvFile();

// Project rule (CLAUDE.md): keep costs low with Haiku.
const MODEL = "claude-haiku-5-5";

// The subagents read this repo, so point cwd at the project root.
const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// ---------------------------------------------------------------------------
// 0. An in-process MCP server, so `mcpServers` has something real to attach.
//    Tools from it are named mcp__<server>__<tool>.
// ---------------------------------------------------------------------------
const repoStats = createSdkMcpServer({
  name: "repo-stats",
  version: "1.0.0",
  tools: [
    tool(
      "count_lines",
      "Count the lines in one file of this repo. Path is relative to the repo root.",
      { path: z.string() },
      async ({ path }) => {
        try {
          const lines = readFileSync(resolve(REPO_ROOT, path), "utf8").split("\n").length;
          return { content: [{ type: "text", text: `${path}: ${lines} lines` }] };
        } catch (err) {
          return { content: [{ type: "text", text: `Cannot read ${path}: ${err}` }], isError: true };
        }
      },
    ),
  ],
});
const COUNT_LINES = "mcp__repo-stats__count_lines";

// ---------------------------------------------------------------------------
// 1. Agent definitions
// ---------------------------------------------------------------------------
const agents: Record<string, AgentDefinition> = {
  // Main-thread agent. options.agent = "coordinator" makes the session itself run as it.
  coordinator: {
    description: "Coordinator that plans the study note and delegates every step.",
    prompt: `You are a coordinator. You never read files yourself: delegate all reading
to code-scout and all writing to doc-writer. Subagents cannot see this conversation,
so put everything they need in the Agent prompt. Always set run_in_background to false.
Plan: (1) call code-scout once per folder, all calls in ONE message so they run in parallel;
(2) pass every report to doc-writer; (3) reply with doc-writer's note unchanged.`,
    // The main agent only needs the delegation tool.
    tools: ["Agent"],
    model: MODEL,
    maxTurns: 10,
    effort: "low",
    // 'dontAsk' denies any tool call that is not pre-approved, instead of prompting.
    // Good for unattended scripts: nothing can hang waiting for a human.
    permissionMode: "dontAsk",
    // Auto-submitted as the first user turn, ONLY because this agent is the main thread
    // (options.agent). It is prepended to the query() prompt. In the run output you can
    // see its effect: the coordinator's first line lists the folders.
    initialPrompt: "Start by listing the folders you will delegate, one per line.",
    criticalSystemReminder_EXPERIMENTAL:
      "Never call Read, Glob or Grep yourself. Your final reply is the doc-writer's note, unchanged.",
  },

  "code-scout": {
    description:
      "Read-only explorer. Use to inspect ONE folder of this repo and report what the example in it demonstrates.",
    prompt: `You are a code scout. You only read code, never change it.
Look at the folder you are given. Use count_lines on its main file. Report:
- Purpose: one sentence on what concept the example demonstrates
- Key APIs: the SDK functions/options it relies on
- Size: the line count of the main file
Keep the report under 120 words. Report facts from the code only.`,
    tools: ["Read", "Glob", "Grep", COUNT_LINES],
    // Belt and braces: even if `tools` were widened later, these stay off.
    disallowedTools: ["Bash", "Write", "Edit"],
    // Refers by name to options.mcpServers["repo-stats"]. An inline config
    // ({ "name": { type: "stdio", command: ... } }) would also work.
    mcpServers: ["repo-stats"],
    model: "haiku",
    maxTurns: 8,
    effort: "low",
    // Its tools are all allowed in options.allowedTools; anything else is denied.
    permissionMode: "dontAsk",
    // Compact this subagent's context early. It only lowers the inherited window.
    // Minimum is 100_000. Below that the WHOLE `agents` map is silently ignored
    // (the session falls back to the default agent with every tool). Verified in 0.3.309.
    autoCompactWindow: 100_000,
    // false: the coordinator waits for the report. true would return at once with a
    // "running in the background" placeholder and deliver the report later.
    background: false,
    // CLAUDE.md here only names the model, which is already set, so skip loading it.
    omitClaudeMd: true,
    // Loads .claude/agent-memory-local/code-scout/ (creates the folder on first run).
    // 'local' is personal and should not be committed; 'project' is the shared one.
    memory: "local",
    // Spawns `auditor` in the background whenever code-scout runs. It gets read-only
    // activity digests. In this one-shot script its reports did not show up in the
    // message stream, so treat it as a config example.
    observer: "auditor",
    observerMessage: "Flag any attempt to modify files or read outside the given folder.",
  },

  "doc-writer": {
    description: "Writer with no tools. Use LAST to turn the scouts' reports into the final study note.",
    prompt: `You are a technical writer preparing study notes for the Claude Certified Associate
exam. Turn the reports you are given into a short markdown note: a one-line intro, one bullet
per example (include its line count), then a "Takeaway" line. Under 200 words.`,
    tools: [], // pure text transformation, nothing to read or run
    // Skill names to preload, e.g. ["study-notes"]. This repo defines no skills, so it is empty.
    skills: [],
    model: "inherit", // same model as the main thread
    maxTurns: 2,
  },

  // Never called by the coordinator. The SDK starts it whenever code-scout runs.
  auditor: {
    description: "Observer that watches code-scout's activity and reports policy problems.",
    prompt: `You audit another agent. You receive digests of what it did.
Report briefly whether it stayed read-only and inside its assigned folder.`,
    tools: [], // reports go through the ObserverReport tool the SDK provides
    model: "haiku",
    maxTurns: 3,
    effort: "low",
  },
};

// The coordinator's prompt carries the plan; this is just the specific request.
const prompt = `Write a study note on these two examples in this repo:
  - coordinator_agent_basic/
  - dynamic_selection/`;

// ---------------------------------------------------------------------------
// 2. Logging that makes the delegation tree visible
// ---------------------------------------------------------------------------
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const blue = (s: string) => `\x1b[34m${s}\x1b[0m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[0m`;

// tool_use_id of each Agent call -> subagent_type, so nested messages can be labelled.
const agentCalls = new Map<string, string>();
const short = (s: string, n = 160) => (s.length > n ? `${s.slice(0, n)}…` : s).replace(/\s+/g, " ");

function format(message: SDKMessage): string | null | undefined {
  if (message.type === "system" && message.subtype === "task_started") {
    if (message.tool_use_id && message.subagent_type) agentCalls.set(message.tool_use_id, message.subagent_type);
    return blue(`[task_started] ${message.subagent_type ?? "?"} — "${message.description}"`);
  }
  if (message.type === "system" && message.subtype === "task_notification") {
    const u = message.usage;
    const stats = u ? ` tokens=${u.total_tokens} tool_uses=${u.tool_uses} ${u.duration_ms}ms` : "";
    return blue(`[task_done] ${message.status}${stats} — ${short(message.summary)}`);
  }

  if (message.type === "assistant") {
    const parent = message.parent_tool_use_id;
    const who = parent ? magenta(`[sub:${agentCalls.get(parent) ?? "?"}]`) : "[main]";
    const lines: string[] = [];
    for (const block of message.message.content) {
      if (block.type === "tool_use") {
        if (block.name === "Agent" || block.name === "Task") {
          const input = block.input as { subagent_type?: string; description?: string };
          agentCalls.set(block.id, input.subagent_type ?? "general-purpose");
          lines.push(`${who} ${blue("delegates →")} ${input.subagent_type}: "${input.description}"`);
        } else {
          lines.push(`${who} tool ${block.name} ${dim(short(JSON.stringify(block.input), 100))}`);
        }
      } else if (block.type === "text" && block.text.trim()) {
        lines.push(`${who} ${short(block.text)}`);
      }
    }
    return lines.join("\n") || null;
  }

  if (message.type === "user" && typeof message.message.content !== "string") {
    const lines: string[] = [];
    for (const block of message.message.content) {
      if (block.type !== "tool_result") continue;
      const body = Array.isArray(block.content)
        ? block.content.map((b) => ("text" in b ? b.text : "")).join(" ")
        : String(block.content ?? "");
      const sub = agentCalls.get(block.tool_use_id);
      if (sub && !message.parent_tool_use_id) {
        lines.push(`[main] ${blue(`← report from ${sub}:`)} ${dim(short(body, 200))}`);
      } else if (message.parent_tool_use_id) {
        // Results of a subagent's own tool calls (Read, count_lines, ...)
        lines.push(dim(`[sub:${agentCalls.get(message.parent_tool_use_id) ?? "?"}] result ${short(body, 100)}`));
      }
    }
    return lines.join("\n") || null;
  }

  return undefined; // fall back to the shared formatter (init, result, ...)
}

const logger = createLogger({ name: "agent_definition", format });

// ---------------------------------------------------------------------------
// 3. Run
// ---------------------------------------------------------------------------
logger.note(`prompt: ${prompt}`);

let finalText = "";
for await (const message of query({
  prompt,
  options: {
    model: MODEL,
    cwd: REPO_ROOT,
    agents,
    agent: "coordinator", // run the main thread AS this AgentDefinition
    mcpServers: { "repo-stats": repoStats },
    // Session tool pool. Each agent's `tools` picks a subset of it.
    tools: ["Agent", "Read", "Glob", "Grep"],
    allowedTools: ["Agent", "Read", "Glob", "Grep", COUNT_LINES],
    maxTurns: 10,
  },
})) {
  logger.log(message);
  if (message.type === "result" && message.subtype === "success") finalText = message.result;
}

console.log("\n================ FINAL NOTE ================\n");
console.log(finalText);
if (logger.file) console.log(dim(`\nlog: ${logger.file}`));
