"""Private runtime-context keys for narrowly scoped middleware audit recorders."""

LOOP_DETECTION_RECORDER_CONTEXT_KEY = "__run_loop_detection_recorder"

# Tool-promotion events share the same parent-loop recorder proxy as loop
# detection: both are middleware:* journal appends that, for native task-tool
# subagents, must be forwarded to the loop that owns the RunJournal rather than
# invoked on the isolated subagent loop. A distinct key keeps the promotion
# path's recorder lookup independent of loop detection so either can evolve
# without silently coupling to the other's presence.
TOOL_PROMOTION_RECORDER_CONTEXT_KEY = "__run_tool_promotion_recorder"
