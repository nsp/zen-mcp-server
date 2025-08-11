# Consensus Tool - Multi-Model Perspective Gathering

**Get diverse expert opinions from multiple AI models on technical proposals and decisions**

The `consensus` tool orchestrates multiple AI models to provide diverse perspectives on your proposals, enabling structured decision-making through for/against analysis and multi-model expert opinions.

## Thinking Mode

**Default is `medium` (8,192 tokens).** Use `high` for complex architectural decisions or `max` for critical strategic choices requiring comprehensive analysis.

## Model Recommendation

Consensus tool uses extended reasoning models by default, making it ideal for complex decision-making scenarios that benefit from multiple perspectives and deep analysis.

## How It Works

The consensus tool orchestrates multiple AI models to provide diverse perspectives on your proposals:

1. **Assign stances**: Each model can take a specific viewpoint (supportive, critical, or neutral)
2. **Gather opinions**: Models analyze your proposal from their assigned perspective with built-in common-sense guardrails
3. **Synthesize results**: Claude combines all perspectives into a balanced recommendation
4. **Natural language**: Use simple descriptions like "supportive", "critical", or "against" - the tool handles synonyms automatically

## Example Prompts

**For/Against Analysis:**
```
Use zen consensus with flash taking a supportive stance and pro being critical to evaluate whether 
we should migrate from REST to GraphQL for our API
```

**Multi-Model Technical Decision:**
```
Get consensus from o3, flash, and pro on our new authentication architecture. Have o3 focus on 
security implications, flash on implementation speed, and pro stay neutral for overall assessment
```

**Natural Language Stance Assignment:**
```
Use consensus tool with gemini being "for" the proposal and grok being "against" to debate 
whether we should adopt microservices architecture
```

```
I want to work on module X and Y, unsure which is going to be more popular with users of my app. 
Get a consensus from gemini supporting the idea for implementing X, grok opposing it, and flash staying neutral
```

## Key Features

- **Stance steering**: Assign specific perspectives (for/against/neutral) to each model with intelligent synonym handling
- **Custom stance prompts**: Provide specific instructions for how each model should approach the analysis
- **Ethical guardrails**: Models will refuse to support truly bad ideas regardless of assigned stance
- **Unknown stance handling**: Invalid stances automatically default to neutral with warning
- **Natural language support**: Use terms like "supportive", "critical", "oppose", "favor" - all handled intelligently
- **Sequential processing**: Reliable execution avoiding MCP protocol issues
- **Focus areas**: Specify particular aspects to emphasize (e.g., 'security', 'performance', 'user experience')
- **File context support**: Include relevant files for informed decision-making
- **Image support**: Analyze architectural diagrams, UI mockups, or design documents
- **Conversation continuation**: Build on previous consensus analysis with additional rounds
- **Web search capability**: Enhanced analysis with current best practices and documentation

## Tool Parameters (Workflow Tool)

**IMPORTANT**: Consensus is a workflow tool that requires step-by-step execution. The first call must include:

### Required Parameters for Step 1:
- `step`: The exact question or proposal that ALL models will evaluate (required)
- `step_number`: Always `1` for the first call (required)
- `total_steps`: Number of models to consult (typically equals length of models array) (required)  
- `next_step_required`: Set to `true` to continue workflow (required)
- `findings`: Your initial analysis of the proposal (required)
- `models`: List of model configurations with stance and custom instructions (required)

### Optional Parameters:
- `relevant_files`: Context files for informed analysis (absolute paths)
- `images`: Visual references like diagrams or mockups (absolute paths)
- `model`: Model to use for the workflow orchestration (default: auto-selected)

### Subsequent Steps:
After step 1, continue with `step_number: 2, 3, etc.` until `total_steps` is reached. Each step processes one model's response.

## Model Configuration Examples

**Basic For/Against:**
```json
[
    {"model": "flash", "stance": "for"},
    {"model": "pro", "stance": "against"}
]
```

**Custom Stance Instructions:**
```json
[
    {"model": "o3", "stance": "for", "stance_prompt": "Focus on implementation benefits and user value"},
    {"model": "flash", "stance": "against", "stance_prompt": "Identify potential risks and technical challenges"}
]
```

**Neutral Analysis:**
```json
[
    {"model": "pro", "stance": "neutral"},
    {"model": "o3", "stance": "neutral"}
]
```

## Usage Examples

### **Correct Tool Call Example:**

```json
{
  "step": "Should we migrate from REST to GraphQL for our API?",
  "step_number": 1,
  "total_steps": 3,
  "next_step_required": true,
  "findings": "GraphQL could provide better flexibility for our frontend teams and reduce over-fetching, but introduces complexity and learning curve. Need to evaluate trade-offs between developer experience, performance, and maintenance overhead.",
  "models": [
    {"model": "gemini-2.5-flash", "stance": "for"},
    {"model": "gemini-2.0-flash-001", "stance": "against"}, 
    {"model": "gemini-2.5-pro", "stance": "neutral"}
  ]
}
```

### **Natural Language Usage (Claude interprets):**

**Architecture Decision:**
```
"Get consensus from pro and o3 on whether to use microservices vs monolith for our e-commerce platform"
```

**Technology Migration:**
```
"Use consensus with flash supporting and pro opposing to evaluate migrating from MySQL to PostgreSQL"
```

**Feature Priority:**
```
"Get consensus from multiple models on whether to prioritize mobile app vs web dashboard development first"
```

**With Visual Context:**
```
"Use consensus to evaluate this new UI design mockup - have flash support it and pro be critical"
```

## Best Practices

- **Provide detailed context**: Include project constraints, requirements, and background
- **Use balanced stances**: Mix supportive and critical perspectives for thorough analysis
- **Specify focus areas**: Guide models to emphasize relevant aspects (security, performance, etc.)
- **Include relevant files**: Provide code, documentation, or specifications for context
- **Build on discussions**: Use continuation for follow-up analysis and refinement
- **Leverage visual context**: Include diagrams, mockups, or design documents when relevant

## Ethical Guardrails

The consensus tool includes built-in ethical safeguards:
- Models won't support genuinely harmful proposals regardless of assigned stance
- Unknown or invalid stances automatically default to neutral
- Warning messages for potentially problematic requests
- Focus on constructive technical decision-making

## When to Use Consensus vs Other Tools

- **Use `consensus`** for: Multi-perspective analysis, structured debates, major technical decisions
- **Use `chat`** for: Open-ended discussions and brainstorming
- **Use `thinkdeep`** for: Extending specific analysis with deeper reasoning
- **Use `analyze`** for: Understanding existing systems without debate