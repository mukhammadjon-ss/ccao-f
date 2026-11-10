/**
 * Built-in Task tool demo (Agent SDK)
 *
 * In the earlier coordinator examples we hand-rolled a `delegate` tool and ran
 * each subagent ourselves. The Agent SDK ships that capability built in:
 *
 *   - Define subagents in `options.agents` (description + prompt + tools + model).
 *   - Allow the built-in delegation tool. It is called `Agent` in current SDKs
 *     (it was called `Task` before; `Task` still appears in docs and type comments).
 *   - The main agent decides when to delegate by calling
 *       Agent({ subagent_type, description, prompt })
 *     and the SDK runs the subagent in its own fresh context window.
 *   - Only the subagent's final report comes back to the main agent as the tool_result.
 *
 * What this demo shows on the console:
 *   [main]          messages from the coordinator (parent_tool_use_id === null)
 *   [sub:<type>]    messages produced inside a subagent (parent_tool_use_id = the Agent call)
 *   [task_started]  / [task_done]  lifecycle events the SDK emits for every subagent run
 *
 * Run: npx tsx task_tool/main.ts
 */
import { query, type AgentDefinition, type SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createLogger } from "../lib/sdk_parser.ts";

process.loadEnvFile();

// Project rule (CLAUDE.md): keep costs low with Haiku.
const MODEL = "claude-haiku-5-5";

// The subagents read this repo, so point cwd at the project root.
const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// ---------------------------------------------------------------------------
// 1. Subagent definitions
//    `description` is what the main agent reads to decide WHO to delegate to.
//    `prompt` is the subagent's system prompt. `tools` scopes what it may do.
// ---------------------------------------------------------------------------
const agents: Record<string, AgentDefinition> = {
  "code-scout": {
    description:
      "Read-only explorer. Use to inspect ONE folder of this repo and report what the example in it demonstrates.",
    prompt: `You are a code scout. You only read code, never change it.
Look at the files in the folder you are given and report:
- Purpose: one sentence on what concept the example demonstrates
- Key APIs: the SDK functions/options it relies on
- Notable: one interesting design choice
Keep the report under 120 words. Report facts from the code only.`,
    tools: ["Read", "Glob", "Grep"],
    model: MODEL,
    maxTurns: 8,
  },
  "doc-writer": {
    description:
      "Writer with no tools. Use LAST to turn the scouts' reports into the final study note.",
    prompt: `You are a technical writer preparing study notes for the
Claude Certified Associate exam. Turn the reports you are given into a short markdown
note: a one-line intro, then one bullet per example, then a "Takeaway" line comparing
hand-rolled delegation with the built-in Task/Agent tool. Under 250 words.

Facts about the built-in Task/Agent tool (this note itself was produced with it):
- Subagents are declared in options.agents with a description, system prompt, tools and model.
- The main agent calls Agent({ subagent_type, description, prompt }); the SDK runs the loop.
- Each subagent gets a fresh context; only its final report returns as the tool_result.
- Several Agent calls in one message run in parallel; the SDK emits task_started / task_notification.`,
    tools: [], // pure text transformation, nothing to read or run
    model: MODEL,
    maxTurns: 2,
  },
};

// ---------------------------------------------------------------------------
// 2. Coordinator prompt
//    The prompt tells the main agent not to read files itself, so it delegates.
// ---------------------------------------------------------------------------
const prompt = `You are a coordinator. Do not read files yourself; delegate all reading.
Produce a study note comparing these two examples in this repo:
  - coordinator_agent_basic/   (hand-rolled delegate tool)
  - dynamic_selection/

Steps:
1. In a SINGLE message, call the Agent tool twice in parallel with subagent_type "code-scout",
   one call per folder. Give each scout the folder path in its prompt. Set run_in_background to false.
2. When both reports are back, call the Agent tool once with subagent_type "doc-writer",
   pasting both reports into its prompt (it cannot see this conversation). Set run_in_background to false.
3. Reply with the doc-writer's note unchanged.`;

// ---------------------------------------------------------------------------
// 3. Logging that makes the delegation tree visible
// ---------------------------------------------------------------------------
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const blue = (s: string) => `\x1b[34m${s}\x1b[0m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[0m`;

// tool_use_id of each Agent call -> subagent_type, so nested messages can be labelled.
const agentCalls = new Map<string, string>();
const short = (s: string, n = 160) => (s.length > n ? `${s.slice(0, n)}…` : s).replace(/\s+/g, " ");

function format(message: SDKMessage): string | null | undefined {
  // Subagent lifecycle events
  if (message.type === "system" && message.subtype === "task_started") {
    return blue(`[task_started] ${message.subagent_type ?? "?"} — "${message.description}" (task ${message.task_id})`);
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
          const input = block.input as { subagent_type?: string; description?: string; prompt?: string };
          agentCalls.set(block.id, input.subagent_type ?? "general-purpose");
          lines.push(`${who} ${blue("delegates →")} ${input.subagent_type}: "${input.description}"`);
          lines.push(dim(`        prompt: ${short(input.prompt ?? "")}`));
        } else {
          lines.push(`${who} tool ${block.name} ${dim(short(JSON.stringify(block.input), 100))}`);
        }
      } else if (block.type === "text" && block.text.trim()) {
        // The coordinator's final answer is printed in full below; keep the stream compact.
        lines.push(`${who} ${parent ? short(block.text) : block.text}`);
      }
    }
    return lines.join("\n") || null;
  }

  if (message.type === "user" && typeof message.message.content !== "string") {
    // Tool results flowing back. For Agent calls this IS the subagent's final report.
    const lines: string[] = [];
    for (const block of message.message.content) {
      if (block.type !== "tool_result") continue;
      const sub = agentCalls.get(block.tool_use_id);
      if (!sub || message.parent_tool_use_id) continue; // skip subagents' own Read/Glob results
      const body = Array.isArray(block.content)
        ? block.content.map((b) => ("text" in b ? b.text : "")).join(" ")
        : String(block.content ?? "");
      lines.push(`[main] ${blue(`← report from ${sub}:`)} ${dim(short(body, 300))}`);
    }
    return lines.join("\n") || null;
  }

  return undefined; // fall back to the shared formatter (init, result, ...)
}

const logger = createLogger({ name: "task_tool", format });

// ---------------------------------------------------------------------------
// 4. Run
// ---------------------------------------------------------------------------
logger.note(`prompt: ${prompt}`);

let finalText = "";
for await (const message of query({
  prompt,
  options: {
    model: MODEL,
    cwd: REPO_ROOT,
    agents,
    // `tools` is the session's tool pool. Subagents pick from it via their own `tools`
    // list, so Read/Glob/Grep must be here for the scouts. The coordinator could use them
    // too; the prompt is what tells it to delegate instead. The init message lists the
    // delegation tool under its wire name, "Task".
    tools: ["Agent", "Read", "Glob", "Grep"],
    allowedTools: ["Agent", "Read", "Glob", "Grep"],
    disallowedTools: ["Bash", "Write", "Edit"],
    maxTurns: 10,
  },
})) {
  logger.log(message);
  if (message.type === "result" && message.subtype === "success") finalText = message.result;
}

console.log("\n================ FINAL NOTE ================\n");
console.log(finalText);
if (logger.file) console.log(dim(`\nlog: ${logger.file}`));
