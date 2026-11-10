import type Anthropic from "@anthropic-ai/sdk";
import type { Tracer } from "./tracer.ts";
import type { AgentRole } from "./types.ts";

export const errMsg = (err: unknown) =>
  err instanceof Error ? err.message : String(err);

export function firstText(content: Anthropic.ContentBlock[]): string {
  const block = content.find(
    (b): b is Anthropic.TextBlock => b.type === "text",
  );
  return block?.text ?? "";
}

/** Reject truncated or empty responses instead of silently storing "". */
export function completeText(
  tracer: Tracer,
  r: Anthropic.Message,
  role: AgentRole,
): string {
  if (r.stop_reason === "max_tokens") {
    throw tracer.fail(
      role,
      "hit max_tokens before finishing (thinking counts against max_tokens)",
    );
  }
  const text = firstText(r.content).trim();
  if (!text) {
    throw tracer.fail(role, `returned no text (stop_reason=${r.stop_reason})`);
  }
  return text;
}

/** Strip markdown code fences if the model adds them anyway, then parse. */
export function parseJson(raw: string): unknown {
  let text = raw.trim();
  if (text.startsWith("```")) {
    text = text
      .split("\n")
      .filter((line) => !line.startsWith("```"))
      .join("\n")
      .trim();
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    // The raw text is already in the trace (llm_response); keep this short.
    throw new Error(`invalid JSON (${errMsg(err)})`);
  }
}

export const stringList = (v: unknown): string[] =>
  Array.isArray(v)
    ? v.filter((x): x is string => typeof x === "string" && x.trim() !== "")
    : [];

export const normalize = (s: string) =>
  s.toLowerCase().replace(/\s+/g, " ").trim();

export const indent = (s: string, pad = "        ") =>
  s.replace(/\n/g, `\n${pad}`);

export const BAR = "=".repeat(60);
