import type { SDKMessage } from "@anthropic-ai/claude-agent-sdk";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const dim = (s: string) => `\x1b[2m${s}\x1b[0m`;
const red = (s: string) => `\x1b[31m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[0m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[0m`;

const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

export function formatMessage(message: SDKMessage): string | null {
  switch (message.type) {
    case "system":
      if (message.subtype === "init") {
        return dim(`[init] model=${message.model} cwd=${message.cwd} tools=${message.tools.join(",")}`);
      }
      return null;

    case "assistant":
      return message.message.content
        .map((block) => {
          if (block.type === "text") return `${cyan("[text]")} ${block.text}`;
          if (block.type === "thinking") return dim(`[thinking] ${block.thinking}`);
          if (block.type === "tool_use") return `${magenta("[tool_use]")} ${block.name} ${JSON.stringify(block.input)}`;
          return null;
        })
        .filter(Boolean)
        .join("\n") || null;

    case "user": {
      const content = message.message.content;
      if (typeof content === "string") return null;
      return content
        .map((block) => {
          if (block.type !== "tool_result") return null;
          const body = Array.isArray(block.content)
            ? block.content.map((b) => ("text" in b ? b.text : "")).join("\n")
            : String(block.content ?? "");
          const tag = block.is_error ? red("[tool_error]") : green("[tool_result]");
          return `${tag} ${body.trim()}`;
        })
        .filter(Boolean)
        .join("\n") || null;
    }

    // Only emitted when the query runs with includePartialMessages: true
    case "stream_event":
      if (message.event.type === "message_delta") {
        return yellow(bold(`[stop_reason] ${message.event.delta.stop_reason}`));
      }
      return null;

    case "result": {
      const stats = `subtype=${message.subtype} stop_reason=${message.stop_reason} turns=${message.num_turns} ` +
        `cost=$${message.total_cost_usd.toFixed(4)} duration=${message.duration_ms}ms`;
      return message.is_error ? red(bold(`[result] ${stats}`)) : bold(`[result] ${stats}`);
    }

    default:
      return null;
  }
}

export interface LoggerOptions {
  /** Directory for log files. Default: `<dir of the entry script>/logs`. */
  dir?: string;
  /** Prefix for the log file name, e.g. "weather" -> weather_2026-10-23_14-05-09.log */
  name?: string;
  /** Print formatted lines to the console. Default: true */
  console?: boolean;
  /** Write formatted lines to a log file. Default: true */
  file?: boolean;
  /** Also write every raw SDK message as JSON to the log file. Default: false */
  raw?: boolean;
  /** Override or extend formatting; return undefined to fall back to formatMessage. */
  format?: (message: SDKMessage) => string | null | undefined;
}

export interface Logger {
  log(message: SDKMessage): void;
  /** Log a free-form line (e.g. the prompt) to console and file. */
  note(text: string): void;
  /** Path of this run's log file, or null when file logging is off. */
  readonly file: string | null;
}

// e.g. 2026-10-23_14-05-09 (no colons, they're awkward in filenames)
const timestamp = () => new Date().toISOString().slice(0, 19).replace("T", "_").replace(/:/g, "-");

function defaultLogDir(): string {
  const entry = process.argv[1];
  return join(entry ? dirname(resolve(entry)) : process.cwd(), "logs");
}

// One log file per logger, created lazily on first write.
export function createLogger(options: LoggerOptions = {}): Logger {
  const { console: toConsole = true, file: toFile = true, raw = false } = options;
  let file: string | null = null;

  const write = (text: string) => {
    if (!toFile) return;
    if (!file) {
      const dir = options.dir ? resolve(options.dir) : defaultLogDir();
      mkdirSync(dir, { recursive: true });
      file = join(dir, `${options.name ? `${options.name}_` : ""}${timestamp()}.log`);
    }
    appendFileSync(file, stripAnsi(text) + "\n");
  };

  const emit = (line: string) => {
    if (toConsole) console.log(line);
    write(line);
  };

  return {
    log(message) {
      if (raw) write(JSON.stringify(message));
      const custom = options.format?.(message);
      const line = custom === undefined ? formatMessage(message) : custom;
      if (line) emit(line);
    },
    note(text) {
      emit(bold(`[note] ${text}`));
    },
    get file() {
      return file;
    },
  };
}

let defaultLogger: Logger | null = null;

/** Shortcut using a default logger (logs next to the entry script). */
export function logMessage(message: SDKMessage): void {
  (defaultLogger ??= createLogger()).log(message);
}
