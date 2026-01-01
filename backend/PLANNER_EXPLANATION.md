# Planner Architecture Explanation

## Current Implementation

The planner supports **two modes** for converting user prompts into execution plans:

### Mode 1: Rule-Based (v1) - **Currently Active by Default**

**How it works:**
- Uses **hardcoded patterns and keywords** to detect user intent
- Fast pattern matching with regex
- No API calls, works offline
- Deterministic results

**Example:**
```python
# User types: "change shirt to black"
# Planner detects:
# - Keyword "change" → edit operation
# - Pattern "change (shirt)" → extracts target="shirt"
# - Creates plan with inpaint operation targeting "shirt"
```

**Pros:**
- ✅ Fast (milliseconds)
- ✅ Free (no API costs)
- ✅ Works offline
- ✅ Deterministic (same input = same output)
- ✅ No dependencies

**Cons:**
- ❌ Limited to predefined patterns
- ❌ Can't understand complex/nuanced prompts
- ❌ Requires updating code for new patterns

### Mode 2: LLM-Based (v2) - **Optional, Intelligent**

**How it works:**
- Uses an **AI language model** to understand user intent
- Model analyzes the prompt and generates a structured plan
- Supports multiple LLM backends:
  - **Ollama** (local, free) - Recommended for privacy
  - **OpenAI API** (cloud, paid)
  - **Transformers** (local models, free but slower)

**Example:**
```python
# User types: "make the photo look like it's from the 1980s with retro colors"
# LLM understands:
# - This is an image edit
# - Needs style transformation
# - Should apply retro color grading
# - Creates plan with appropriate operations
```

**Pros:**
- ✅ Understands complex, nuanced prompts
- ✅ Handles variations in wording
- ✅ Can infer context and intent
- ✅ Adapts to new scenarios without code changes

**Cons:**
- ❌ Slower (seconds, not milliseconds)
- ❌ May require API keys (OpenAI) or local setup (Ollama)
- ❌ Non-deterministic (same input might give different outputs)
- ❌ Requires LLM to be available

## Configuration

### Default Behavior (Rule-Based)
```python
planner = Planner()  # Uses rule-based by default
```

### Enable LLM Mode
```python
# Option 1: Always use LLM (if available)
planner = Planner(mode="llm")

# Option 2: Auto (try LLM, fallback to rules)
planner = Planner(mode="auto")

# Option 3: Force rule-based
planner = Planner(mode="rule")
```

### Using Ollama (Recommended for Local LLM)

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. The planner will automatically detect Ollama if running

### Using OpenAI API

1. Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
2. Planner will use OpenAI automatically

## When to Use Which?

### Use Rule-Based When:
- ✅ You want fast responses
- ✅ You don't want API costs
- ✅ Prompts follow predictable patterns
- ✅ You need deterministic results
- ✅ Working offline

### Use LLM-Based When:
- ✅ Users type natural, varied language
- ✅ You need to understand complex intent
- ✅ Prompts are creative/unpredictable
- ✅ You want the system to "think" about the request
- ✅ You have LLM access (Ollama or API)

## Current Status

**Default: Rule-Based (v1)**
- Fast, reliable, works everywhere
- Good for most use cases
- Easy to extend with new patterns

**LLM Support: Available but Optional**
- Automatically detects if LLM is available
- Falls back to rule-based if LLM fails
- Can be enabled per-request or globally

## Future Improvements

1. **Hybrid Approach**: Use LLM for complex prompts, rules for simple ones
2. **Learning**: Track which patterns work best
3. **Custom Models**: Fine-tune a small model for planning tasks
4. **Caching**: Cache LLM responses for common prompts



