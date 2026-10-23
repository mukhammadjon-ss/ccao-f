import type { SDKMessage } from "@anthropic-ai/claude-agent-sdk";

const bold = (s: string) => `\x1b[1m${s}\x1b[0m`;
const cyan = (s: string) => `\x1b[36m${s}\x1b[0m`;
const green = (s: string) => `\x1b[32m${s}\x1b[0m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[0m`;
const magenta = (s: string) => `\x1b[35m${s}\x1b[0m`;

export function formatMessage(message: SDKMessage): string | null {
  switch (message.type) {
    case "assistant":
      return message.message.content
        .map((block) => {
          if (block.type === "text") return `${cyan("[text]")} ${block.text}`;
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
          return `${green("[tool_result]")} ${body.trim()}`;
        })
        .filter(Boolean)
        .join("\n") || null;
    }

    case "stream_event":
      if (message.event.type === "message_delta") {
        return yellow(bold(`[stop_reason] ${message.event.delta.stop_reason}`));
      }
      return null;

    case "result":
      return bold(`[result] stop_reason=${message.stop_reason} turns=${message.num_turns}`);

    default:
      return null;
  }
}

export function logMessage(message: SDKMessage): void {
  const line = formatMessage(message);
  if (line) console.log(line);
}
