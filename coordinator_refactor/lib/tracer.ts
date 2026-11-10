// ---------------------------------------------------------------------------
// Observability layer
//
// Every model call (coordinator and spokes) goes through Tracer.call, which
// writes a JSONL trace to logs/<run_id>.jsonl:
//   run_start / llm_request / llm_response / llm_error / agent_error /
//   tool_call / tool_result / loop_decision / coordinator_nudge / run_end
// Events are appended as they happen, so a crash still leaves a usable trace.
// ---------------------------------------------------------------------------

import Anthropic from "@anthropic-ai/sdk";
import { appendFileSync, mkdirSync } from "node:fs";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { LOG_DIR } from "./config.ts";
import { errMsg } from "./text.ts";
import type { AgentRole, UsageTotals } from "./types.ts";

export class AgentCallError extends Error {
  constructor(
    readonly role: AgentRole,
    message: string,
  ) {
    super(`${role}: ${message}`);
    this.name = "AgentCallError";
  }
}

export class Tracer {
  readonly runId = `${new Date().toISOString().replace(/[:.]/g, "-")}-${randomUUID().slice(0, 8)}`;
  readonly file: string;
  readonly usage = new Map<AgentRole, UsageTotals>();
  private seq = 0;
  private callSeq = 0;
  private systemLogged = new Set<AgentRole>();

  constructor(private readonly client: Anthropic) {
    mkdirSync(LOG_DIR, { recursive: true });
    this.file = path.join(LOG_DIR, `${this.runId}.jsonl`);
  }

  event(type: string, data: Record<string, unknown> = {}): void {
    const line = JSON.stringify({
      ts: new Date().toISOString(),
      run_id: this.runId,
      seq: ++this.seq,
      type,
      ...data,
    });
    appendFileSync(this.file, line + "\n");
  }

  private totals(role: AgentRole): UsageTotals {
    let t = this.usage.get(role);
    if (!t) {
      t = { calls: 0, errors: 0, input_tokens: 0, output_tokens: 0, ms: 0 };
      this.usage.set(role, t);
    }
    return t;
  }

  /**
   * The single choke point for every model call. Logs the exact input
   * (system prompt once per role, then messages from `fromMessage`),
   * the full response, usage, latency and request id — or the error.
   */
  async call(
    role: AgentRole,
    params: Anthropic.MessageCreateParamsNonStreaming,
    opts: { label?: string; fromMessage?: number } = {},
  ): Promise<Anthropic.Message> {
    const callId = ++this.callSeq;
    const totals = this.totals(role);
    totals.calls++;
    const firstForRole = !this.systemLogged.has(role);
    this.systemLogged.add(role);

    this.event("llm_request", {
      call_id: callId,
      role,
      label: opts.label,
      model: params.model,
      max_tokens: params.max_tokens,
      system: firstForRole ? params.system : undefined,
      tools: firstForRole
        ? params.tools?.map((t) => ("name" in t ? t.name : t.type))
        : undefined,
      message_count: params.messages.length,
      messages: params.messages.slice(opts.fromMessage ?? 0),
    });

    const t0 = Date.now();
    try {
      const r = await this.client.messages.create(params);
      const ms = Date.now() - t0;
      totals.ms += ms;
      totals.input_tokens += r.usage.input_tokens;
      totals.output_tokens += r.usage.output_tokens;
      this.event("llm_response", {
        call_id: callId,
        role,
        label: opts.label,
        ms,
        stop_reason: r.stop_reason,
        usage: r.usage,
        request_id: (r as { _request_id?: string | null })._request_id ?? null,
        content: r.content,
      });
      return r;
    } catch (err) {
      const ms = Date.now() - t0;
      totals.ms += ms;
      totals.errors++;
      const api = err instanceof Anthropic.APIError ? err : null;
      this.event("llm_error", {
        call_id: callId,
        role,
        label: opts.label,
        ms,
        error: errMsg(err),
        status: api?.status ?? null,
        error_type: api?.type ?? null,
        request_id: api?.requestID ?? null,
      });
      console.log(`  [ERROR] ${role} call failed: ${errMsg(err)}`);
      throw new AgentCallError(role, errMsg(err));
    }
  }

  /** Record a failure detected after a successful API call (bad output, truncation). */
  fail(role: AgentRole, message: string): AgentCallError {
    this.totals(role).errors++;
    this.event("agent_error", { role, error: message });
    console.log(`  [ERROR] ${role}: ${message}`);
    return new AgentCallError(role, message);
  }
}
