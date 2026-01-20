# Bun Development Skill

Claude Code skill for Bun/TypeScript development in trading-fitness.

**← [Back to trading-fitness](../../../CLAUDE.md)**

## Package Documentation

- [core-bun/CLAUDE.md](../../../packages/core-bun/CLAUDE.md)

## Triggers

- TypeScript file changes in `packages/core-bun/`
- API or async I/O development
- bun commands

## Guidelines

### Package Management

```bash
bun add <package>      # Add dependency
bun add -d <pkg>       # Add dev dependency
bun install            # Install all deps
```

### Running

```bash
bun run <file.ts>      # Run TypeScript file
bun --hot <file.ts>    # Run with hot reload
```

### Testing

```bash
bun test               # Run all tests
bun test <pattern>     # Run matching tests
```

### Linting

```bash
bun run lint           # Run biome
bunx @biomejs/biome check --fix  # Direct biome
```

### Logging

Use pino with NDJSON output:

```typescript
import pino from "pino";

const logger = pino({
  transport: {
    target: "pino/file",
    options: { destination: "./logs/core-bun.jsonl" },
  },
});

logger.info({ package: "core-bun" }, "Operation started");
```
