// Untrusted input handling — the resume is written by the candidate.
// The matching instruction for the models lives in prompts/shared.ts.

/** Wrap data in a tag, stripping any tags that could close it early. */
export function tag(name: string, body: string): string {
  const clean = body.replace(
    /<\/?\s*(job_posting|resume|findings|scope)\s*>/gi,
    "",
  );
  return `<${name}>\n${clean}\n</${name}>`;
}

export const documents = (jobPosting: string, resume: string) =>
  `${tag("job_posting", jobPosting)}\n\n${tag("resume", resume)}`;
