/**
 * Quality criteria: a mock-exam builder whose coordinator has an explicit GOAL and
 * measurable QUALITY CRITERIA (Agent SDK 0.3.309)
 *
 * Where the goal and criteria take effect:
 *   1. They are defined once (section 0) and put into the coordinator, writer and assembler prompts.
 *   2. The assembler grades each question against them and writes a JSON report.
 *   3. The coordinator reads that report and runs ONE repair round if coverage (C5) fails.
 *   4. After the run, the code checks the report again and exits 1 if the criteria are not met.
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
 * Run: npx tsx quality_criteria/main.ts
 */
import {
  query,
  type AgentDefinition,
  type SDKMessage,
} from "@anthropic-ai/claude-agent-sdk";
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
// ---------------------------------------------------------------------------
// 0. Goal and quality criteria: ONE source of truth.
//    They are injected into every prompt (subagents cannot see the coordinator's
//    chat), and the same numbers are checked again in code after the run.
// ---------------------------------------------------------------------------
const TOPICS = [
  { name: "coordinator patterns", folders: ["coordinator_agent_basic/", "coordinator_refactor/"] },
  { name: "tool use & stop_reason", folders: ["stop_reason/", "task_tool/"] },
  { name: "agent definitions", folders: ["agent_definition/", "dynamic_selection/"] },
  { name: "task decomposition", folders: ["narrow_task_decomposition/", "research_partitioning/"] },
];
const PER_TOPIC = 3;
const MIN_PER_TOPIC = 2; // kept questions a topic needs after verification
const MIN_TOTAL = 10;

const GOAL = `Produce a ${TOPICS.length * PER_TOPIC}-question multiple-choice mock exam \
(${PER_TOPIC} per topic) for the Claude Certified Associate exam, in which every question \
has exactly one correct answer that can be proven from this repo's code.`;

const CRITERIA = [
  "C1 Single answer: exactly one option (A-D) is correct; the other three are plausible but wrong.",
  "C2 Grounded: every answer-key line cites a real file:line inside that topic's folders.",
  "C3 Understanding: asks why / what happens if, never trivia (names, counts, line numbers).",
  "C4 Verified: a blind verifier, without the key, picks the same letter.",
  `C5 Coverage: at least ${MIN_PER_TOPIC} kept questions per topic and at least ${MIN_TOTAL} in total.`,
];
const CRITERIA_TEXT = CRITERIA.map((c) => `- ${c}`).join("\n");
const TOPIC_LIST = TOPICS.map(
  (t, i) => `  ${i + 1}. ${t.name.padEnd(24)} -> ${t.folders.join(", ")}`,
).join("\n");

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
    description:
      "Builds a verified mock exam from this repo's examples using parallel agents.",
    prompt: `You build a mock exam. You never read files yourself.

GOAL
${GOAL}

QUALITY CRITERIA (the exam is done only when all of them hold)
${CRITERIA_TEXT}

TOPICS
${TOPIC_LIST}

WAVE 1: in ONE message, call question-writer ${TOPICS.length} times, one per topic.
Each prompt must contain the topic, its folders and criteria C1-C3 word for word
(subagents cannot see this chat).

WAVE 2: split each writer's output into QUESTIONS and ANSWER KEY.
In ONE message, call blind-verifier ${TOPICS.length} times. Give each verifier ONLY the
questions of one set, never the key, plus the folders of that topic.

ASSEMBLE: call exam-assembler once with all questions, all keys and all verifier
answers, labelled with the exact topic names above.

QUALITY GATE: read the assembler's JSON report. If C5 fails, run ONE repair round:
question-writer for the failing topics only (one message, ask for ${PER_TOPIC} NEW questions),
then blind-verifier for them (one message), then exam-assembler again with ALL
material. Never more than one repair round.

Reply with the last assembler output unchanged. Always set run_in_background to
false. If a subagent fails, note it for the assembler instead of retrying.`,
    tools: ["Agent"],
    model: MODEL,
    maxTurns: 12, // 3 normal waves + up to 3 repair waves + final reply
    effort: "low",
    permissionMode: "dontAsk",
    criticalSystemReminder_EXPERIMENTAL:
      "All calls of a wave go in the SAME message. Never pass an answer key to blind-verifier. " +
      "Check the quality report against C5 before you reply.",
  },

  "question-writer": {
    ...READ_ONLY,
    description:
      "Writes 3 exam questions on ONE topic, grounded in the given repo folders.",
    prompt: `Read the main files of the folders you are given. Write ${PER_TOPIC} multiple-choice
questions (A-D). Every question must meet these criteria:
${CRITERIA.slice(0, 3).map((c) => `- ${c}`).join("\n")}
Before you answer, re-check each question against them and replace any that fails.
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
    description:
      "Answers exam questions WITHOUT the key, using the repo as evidence.",
    prompt: `Answer each question you are given. Read the repo to check your answer.
For each, one line: "<n>. <letter> (confidence: high|low) - <one-line reason>".
If two options seem correct, or none is, write "<n>. AMBIGUOUS - <why>".`,
    maxTurns: 6,
  },

  "exam-assembler": {
    description:
      "Merges questions, keys and verifier answers into the final exam. Use LAST, once.",
    prompt: `You get question sets, their answer keys, and a blind verifier's answers.

GOAL
${GOAL}

Judge every question against these criteria:
${CRITERIA_TEXT}
- Key matches verifier and C1-C3 hold: keep.
- Mismatch, AMBIGUOUS, or C1-C3 broken: rewrite it to be unambiguous (fixed), or drop it.
A fixed question counts as kept. If a topic got a repair set, keep the best ${PER_TOPIC} per topic.

Output markdown: "## Mock exam" (numbered questions, grouped by topic),
"## Answer key", "## Quality report" (one line per fix or drop, naming the
criterion that failed), and END with exactly one fenced json block:
\`\`\`json
{"topics": {"<exact topic name>": {"kept": 0, "fixed": 0, "dropped": 0}},
 "total_kept": 0, "criteria_met": true, "failed": ["C5: <why>"]}
\`\`\`
Use these exact topic names: ${TOPICS.map((t) => JSON.stringify(t.name)).join(", ")}.`,
    tools: [],
    model: "haiku",
    maxTurns: 2,
  },
};

const prompt =
  "Build the mock exam for the Claude Certified Associate exam from this repo.";

// ---------------------------------------------------------------------------
// 2. Timing: record when each subagent starts and ends, relative to the run start
// ---------------------------------------------------------------------------
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const blue = (s: string) => `\x1b[34m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[0m`;
const short = (s: string, n = 140) =>
  (s.length > n ? `${s.slice(0, n)}…` : s).replace(/\s+/g, " ");

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
      if (
        block.type === "tool_use" &&
        (block.name === "Agent" || block.name === "Task")
      ) {
        let wave = waveOf.get(message.message.id);
        if (wave === undefined) {
          wave = waveOf.size + 1;
          waveOf.set(message.message.id, wave);
          lines.push(
            green(
              `${at()} [main] WAVE ${wave} begins (message ${message.message.id.slice(-8)})`,
            ),
          );
        }
        const input = block.input as {
          subagent_type?: string;
          description?: string;
        };
        runs.set(block.id, {
          type: input.subagent_type ?? "?",
          description: input.description ?? "",
          wave,
        });
        lines.push(
          `${at()} [main] ${blue("delegates ->")} ${input.subagent_type}: "${input.description}"`,
        );
      } else if (block.type === "text" && block.text.trim()) {
        lines.push(`${at()} [main] ${short(block.text)}`);
      }
    }
    return lines.join("\n") || null;
  }

  if (message.type === "system" && message.subtype === "task_started") {
    const run = message.tool_use_id ? runs.get(message.tool_use_id) : undefined;
    if (run) run.start = Date.now();
    return blue(
      `${at()} [start] ${message.subagent_type ?? "?"} - "${message.description}"`,
    );
  }

  if (message.type === "system" && message.subtype === "task_notification") {
    const run = message.tool_use_id ? runs.get(message.tool_use_id) : undefined;
    if (run) {
      run.end = Date.now();
      run.status = message.status;
      run.durationMs = message.usage?.duration_ms;
    }
    const u = message.usage;
    const stats = u
      ? dim(` tokens=${u.total_tokens} tools=${u.tool_uses} ${u.duration_ms}ms`)
      : "";
    return (
      blue(`${at()} [done]  ${run?.type ?? "?"} ${message.status}`) + stats
    );
  }

  // Subagent chatter: show tool calls only, so the parallel interleaving stays readable.
  if (message.type === "assistant" && message.parent_tool_use_id) {
    const who = magenta(
      `[${runs.get(message.parent_tool_use_id)?.type ?? "sub"}]`,
    );
    const tools = message.message.content
      .filter((b) => b.type === "tool_use")
      .map(
        (b) =>
          `${at()} ${who} ${b.name} ${dim(short(JSON.stringify(b.input), 90))}`,
      );
    return tools.join("\n") || null;
  }
  if (message.type === "user") return null; // tool results: too noisy here

  return undefined; // init, result, ... -> shared formatter
}

function timingReport(): string {
  const sec = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
  const lines = ["", "================ TIMING ================"];
  const byWave = new Map<number, Run[]>();
  for (const r of runs.values())
    byWave.set(r.wave, [...(byWave.get(r.wave) ?? []), r]);

  for (const [w, list] of [...byWave].sort((a, b) => a[0] - b[0])) {
    const done = list.filter((r) => r.start && r.end);
    lines.push(
      `Wave ${w}: ${list.length} x ${[...new Set(list.map((r) => r.type))].join(", ")}`,
    );
    for (const r of done) {
      lines.push(
        `  ${r.type.padEnd(16)} ${sec(r.start! - t0).padStart(7)} -> ${sec(r.end! - t0).padStart(7)}  ${r.status}`,
      );
    }
    if (done.length > 1) {
      const wall =
        Math.max(...done.map((r) => r.end!)) -
        Math.min(...done.map((r) => r.start!));
      const sum = done.reduce(
        (s, r) => s + (r.durationMs ?? r.end! - r.start!),
        0,
      );
      // Peak concurrency: the most subagents running at the same instant. Each subagent
      // starts as soon as its tool_use block has streamed, so calls with long prompts
      // start ~1-2s apart and a fast one may finish before the last one starts.
      const peak = Math.max(
        ...done.map(
          (r) =>
            done.filter((o) => o.start! <= r.start! && o.end! > r.start!)
              .length,
        ),
      );
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
    maxTurns: 12,
  },
})) {
  logger.log(message);
  if (message.type === "result" && message.subtype === "success")
    finalText = message.result;
}

// ---------------------------------------------------------------------------
// 4. Verify the quality criteria in code: never trust the model's own "criteria_met".
// ---------------------------------------------------------------------------
function qualityCheck(text: string): { ok: boolean; report: string } {
  const lines = ["", "================ QUALITY GATE ================", `Goal: ${GOAL}`];
  const blocks = [...text.matchAll(/```json\s*([\s\S]*?)```/g)];
  const raw = blocks.at(-1)?.[1];
  if (!raw) return { ok: false, report: [...lines, "FAIL: no JSON quality report found"].join("\n") };

  let data: { topics?: Record<string, { kept?: number }>; total_kept?: number; criteria_met?: boolean };
  try {
    data = JSON.parse(raw);
  } catch {
    return { ok: false, report: [...lines, "FAIL: quality report is not valid JSON"].join("\n") };
  }

  let ok = true;
  let total = 0;
  for (const t of TOPICS) {
    const kept = data.topics?.[t.name]?.kept ?? 0;
    total += kept;
    const pass = kept >= MIN_PER_TOPIC;
    ok &&= pass;
    lines.push(`  ${t.name.padEnd(24)} kept ${kept}/${PER_TOPIC}  ${pass ? green("PASS") : "\x1b[31mFAIL\x1b[0m"}`);
  }
  const totalPass = total >= MIN_TOTAL;
  ok &&= totalPass;
  lines.push(`  total kept ${total} (min ${MIN_TOTAL})  ${totalPass ? green("PASS") : "\x1b[31mFAIL\x1b[0m"}`);
  if (data.total_kept !== undefined && data.total_kept !== total)
    lines.push(dim(`  note: assembler claimed total_kept=${data.total_kept}, per-topic sum is ${total}`));
  if (data.criteria_met !== ok)
    lines.push(dim(`  note: assembler claimed criteria_met=${data.criteria_met}, code says ${ok}`));
  lines.push(ok ? green("Criteria met") : "\x1b[31mCriteria NOT met\x1b[0m");
  return { ok, report: lines.join("\n") };
}

console.log("\n================ MOCK EXAM ================\n");
console.log(finalText);
const gate = qualityCheck(finalText);
logger.note(gate.report);
if (!gate.ok) process.exitCode = 1;
logger.note(timingReport());
if (logger.file) console.log(dim(`\nlog: ${logger.file}`));
