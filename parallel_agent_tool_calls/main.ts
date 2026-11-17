/**
 * Parallel agents: a mock-exam builder with blind verification (Agent SDK 0.3.309)
 *
 * The coordinator runs TWO waves of parallel subagents, then one sequential step:
 *
 *   Wave 1 (parallel)  4 x question-writer   one per exam topic, each reads its own folders
 *   Wave 2 (parallel)  4 x blind-verifier    answers one question set WITHOUT the answer key
 *   Final  (single)    1 x exam-assembler    keeps / fixes / drops questions where key and
 *                                            verifier disagree
 *
 * How parallelism happens: the model puts several Agent tool_use blocks in ONE assistant
 * message, and the SDK runs them concurrently. Calls spread over several messages run
 * one after another. The timing table at the end shows which of the two you got.
 *
 * Run: npm run parallel   (or: npx tsx parallel_agent_tool_calls/main.ts)
 */
import { query, type AgentDefinition, type SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { createLogger } from "../lib/sdk_parser.ts";

process.loadEnvFile();

// Project rule (CLAUDE.md): keep costs low with Haiku.
const MODEL = "claude-haiku-5-5";
const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// ---------------------------------------------------------------------------
// 1. Agent definitions
//    Only 3 definitions, but question-writer and blind-verifier each run 4 times
//    at once: one definition, several parallel instances.
// ---------------------------------------------------------------------------
const READ_ONLY = {
  tools: ["Read", "Glob", "Grep"],
  disallowedTools: ["Write", "Edit", "Bash"],
  model: "haiku",
  effort: "low",
  permissionMode: "dontAsk",
  omitClaudeMd: true, // CLAUDE.md only names the model, which is already set
} satisfies Partial<AgentDefinition>;

const agents: Record<string, AgentDefinition> = {
  coordinator: {
    description: "Builds a verified mock exam from this repo's examples using parallel agents.",
    prompt: `You build a mock exam. You never read files yourself.

WAVE 1: in ONE message, call question-writer 4 times, one per topic:
  1. coordinator patterns   -> coordinator_agent_basic/, coordinator_refactor/
  2. tool use & stop_reason -> stop_reason/, task_tool/
  3. agent definitions      -> agent_definition/, dynamic_selection/
  4. task decomposition     -> narrow_task_decomposition/, research_partitioning/
Each prompt must name the topic and its folders (subagents cannot see this chat).

WAVE 2: split each writer's output into QUESTIONS and ANSWER KEY.
In ONE message, call blind-verifier 4 times. Give each verifier ONLY the
questions of one set, never the key, plus the folders of that topic.

FINAL: call exam-assembler once with all questions, all keys and all verifier
answers, labelled by set number. Reply with its output unchanged.

Always set run_in_background to false. If a subagent fails, note it for the
assembler instead of retrying.`,
    tools: ["Agent"],
    model: MODEL,
    maxTurns: 8,
    effort: "low",
    permissionMode: "dontAsk",
    criticalSystemReminder_EXPERIMENTAL:
      "All calls of a wave go in the SAME message. Never pass an answer key to blind-verifier.",
  },

  "question-writer": {
    ...READ_ONLY,
    description: "Writes 3 exam questions on ONE topic, grounded in the given repo folders.",
    prompt: `Read the main files of the folders you are given. Write 3 multiple-choice
questions (A-D) that test understanding, not trivia. Each must be answerable from the code.
Output exactly this, nothing else:
QUESTIONS
1. <question> A) ... B) ... C) ... D) ...
2. ...
3. ...
ANSWER KEY
1. <letter> because <one line citing file:line>
2. ...
3. ...`,
    maxTurns: 6,
  },

  "blind-verifier": {
    ...READ_ONLY,
    description: "Answers exam questions WITHOUT the key, using the repo as evidence.",
    prompt: `Answer each question you are given. Read the repo to check your answer.
For each, one line: "<n>. <letter> (confidence: high|low) - <one-line reason>".
If two options seem correct, or none is, write "<n>. AMBIGUOUS - <why>".`,
    maxTurns: 6,
  },

  "exam-assembler": {
    description: "Merges questions, keys and verifier answers into the final exam. Use LAST, once.",
    prompt: `You get question sets, their answer keys, and a blind verifier's answers.
Compare each key with the verifier's answer:
- Match: keep the question.
- Mismatch or AMBIGUOUS: rewrite it to be unambiguous, or drop it. Say which.
Output markdown: "## Mock exam" (numbered questions, grouped by topic),
"## Answer key", then "## Quality report" with kept / fixed / dropped counts
and one line per fix or drop.`,
    tools: [],
    model: "haiku",
    maxTurns: 2,
  },
};

const prompt = "Build the mock exam for the Claude Certified Associate exam from this repo.";

// ---------------------------------------------------------------------------
// 2. Timing: record when each subagent starts and ends, relative to the run start
// ---------------------------------------------------------------------------
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const blue = (s: string) => `\x1b[34m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[0m`;
const short = (s: string, n = 140) => (s.length > n ? `${s.slice(0, n)}…` : s).replace(/\s+/g, " ");

const t0 = Date.now();
const at = () => `+${((Date.now() - t0) / 1000).toFixed(1).padStart(5)}s`;

interface Run {
  type: string;
  description: string;
  wave: number; // index of the coordinator message that launched it
  start?: number;
  end?: number;
  durationMs?: number;
  status?: string;
}
const runs = new Map<string, Run>(); // tool_use_id -> Run
// The SDK emits ONE event per content block, so a single API message with 4 Agent
// calls arrives as 4 events sharing message.id. A wave = one API message id.
const waveOf = new Map<string, number>(); // API message id -> wave number

function format(message: SDKMessage): string | null | undefined {
  // The coordinator's messages: group Agent calls by the API message that carries them.
  if (message.type === "assistant" && !message.parent_tool_use_id) {
    const lines: string[] = [];
    for (const block of message.message.content) {
      if (block.type === "tool_use" && (block.name === "Agent" || block.name === "Task")) {
        let wave = waveOf.get(message.message.id);
        if (wave === undefined) {
          wave = waveOf.size + 1;
          waveOf.set(message.message.id, wave);
          lines.push(green(`${at()} [main] WAVE ${wave} begins (message ${message.message.id.slice(-8)})`));
        }
        const input = block.input as { subagent_type?: string; description?: string };
        runs.set(block.id, { type: input.subagent_type ?? "?", description: input.description ?? "", wave });
        lines.push(`${at()} [main] ${blue("delegates ->")} ${input.subagent_type}: "${input.description}"`);
      } else if (block.type === "text" && block.text.trim()) {
        lines.push(`${at()} [main] ${short(block.text)}`);
      }
    }
    return lines.join("\n") || null;
  }

  if (message.type === "system" && message.subtype === "task_started") {
    const run = message.tool_use_id ? runs.get(message.tool_use_id) : undefined;
    if (run) run.start = Date.now();
    return blue(`${at()} [start] ${message.subagent_type ?? "?"} - "${message.description}"`);
  }

  if (message.type === "system" && message.subtype === "task_notification") {
    const run = message.tool_use_id ? runs.get(message.tool_use_id) : undefined;
    if (run) {
      run.end = Date.now();
      run.status = message.status;
      run.durationMs = message.usage?.duration_ms;
    }
    const u = message.usage;
    const stats = u ? dim(` tokens=${u.total_tokens} tools=${u.tool_uses} ${u.duration_ms}ms`) : "";
    return blue(`${at()} [done]  ${run?.type ?? "?"} ${message.status}`) + stats;
  }

  // Subagent chatter: show tool calls only, so the parallel interleaving stays readable.
  if (message.type === "assistant" && message.parent_tool_use_id) {
    const who = magenta(`[${runs.get(message.parent_tool_use_id)?.type ?? "sub"}]`);
    const tools = message.message.content
      .filter((b) => b.type === "tool_use")
      .map((b) => `${at()} ${who} ${b.name} ${dim(short(JSON.stringify(b.input), 90))}`);
    return tools.join("\n") || null;
  }
  if (message.type === "user") return null; // tool results: too noisy here

  return undefined; // init, result, ... -> shared formatter
}

function timingReport(): string {
  const sec = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
  const lines = ["", "================ TIMING ================"];
  const byWave = new Map<number, Run[]>();
  for (const r of runs.values()) byWave.set(r.wave, [...(byWave.get(r.wave) ?? []), r]);

  for (const [w, list] of [...byWave].sort((a, b) => a[0] - b[0])) {
    const done = list.filter((r) => r.start && r.end);
    lines.push(`Wave ${w}: ${list.length} x ${[...new Set(list.map((r) => r.type))].join(", ")}`);
    for (const r of done) {
      lines.push(`  ${r.type.padEnd(16)} ${sec(r.start! - t0).padStart(7)} -> ${sec(r.end! - t0).padStart(7)}  ${r.status}`);
    }
    if (done.length > 1) {
      const wall = Math.max(...done.map((r) => r.end!)) - Math.min(...done.map((r) => r.start!));
      const sum = done.reduce((s, r) => s + (r.durationMs ?? r.end! - r.start!), 0);
      // Peak concurrency: the most subagents running at the same instant. Each subagent
      // starts as soon as its tool_use block has streamed, so calls with long prompts
      // start ~1-2s apart and a fast one may finish before the last one starts.
      const peak = Math.max(...done.map((r) => done.filter((o) => o.start! <= r.start! && o.end! > r.start!).length));
      lines.push(
        `  wall ${sec(wall)} vs sum ${sec(sum)} -> speed-up x${(sum / wall).toFixed(1)}  ` +
          `peak concurrency ${peak}/${done.length} ` +
          (peak > 1 ? green("PARALLEL") : "\x1b[31mSEQUENTIAL\x1b[0m"),
      );
    }
  }
  lines.push(`Total run: ${sec(Date.now() - t0)}`);
  return lines.join("\n");
}

const logger = createLogger({ name: "parallel_agents", format });

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
    agent: "coordinator", // the main thread runs AS the coordinator definition
    tools: ["Agent", "Read", "Glob", "Grep"],
    allowedTools: ["Agent", "Read", "Glob", "Grep"],
    maxTurns: 8,
  },
})) {
  logger.log(message);
  if (message.type === "result" && message.subtype === "success") finalText = message.result;
}

console.log("\n================ MOCK EXAM ================\n");
console.log(finalText);
logger.note(timingReport());
if (logger.file) console.log(dim(`\nlog: ${logger.file}`));
