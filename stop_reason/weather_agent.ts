import { query } from "@anthropic-ai/claude-agent-sdk";
import { logMessage } from "../lib/sdk_parser.ts";

const prompt =
  "What's the weather in Tashkent and London right now? " +
  "Use curl with https://wttr.in/<city>?format=3";

for await (const message of query({
  prompt,
  options: {
    tools: ["Bash"],
    allowedTools: ["Bash(curl:*)"],
    includePartialMessages: true,
  },
})) {
  logMessage(message);
}
