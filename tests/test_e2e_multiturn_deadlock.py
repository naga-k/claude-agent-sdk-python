"""End-to-end test for multi-turn deadlock fix (issue #558).

This test uses the REAL Claude CLI to verify that multi-turn conversations
with ClaudeSDKClient work correctly and don't deadlock.

The scenario:
1. First query asks Claude a simple question
2. Second query is sent immediately after — with the old bounded buffer,
   this could deadlock because _read_messages() blocks on unconsumed
   messages, preventing control protocol routing
3. With the fix (unbounded buffer), both turns complete successfully

Requires:
- Claude CLI installed and accessible
- ANTHROPIC_API_KEY set in environment
- Network access to Anthropic API

Usage:
    python -m pytest tests/test_e2e_multiturn_deadlock.py -v -s
    # Or run directly:
    python tests/test_e2e_multiturn_deadlock.py
"""

import os
import shutil

import anyio
import pytest

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)

# Skip unless explicitly opted in — these tests require network + API access
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_E2E_TESTS"),
    reason="Set RUN_E2E_TESTS=1 to run e2e tests (requires network + API key)",
)


def _cli_available() -> bool:
    """Check if Claude CLI is installed."""
    return shutil.which("claude") is not None


class TestE2EMultiTurnDeadlock:
    """End-to-end tests for multi-turn conversation deadlock fix."""

    def test_multi_turn_conversation_completes(self):
        """Two-turn conversation completes without deadlocking.

        This is the basic sanity check: send two queries sequentially
        and verify both produce ResultMessages.
        """
        if not _cli_available():
            pytest.skip("Claude CLI not installed")

        async def _test():
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model="claude-sonnet-4-5-20250929",
                max_turns=2,
                system_prompt="Respond very briefly in 1-2 sentences.",
            )

            async with ClaudeSDKClient(options=options) as client:
                # First turn
                await client.query("What is 2+2? Answer in one word.")
                first_messages = []
                with anyio.fail_after(30):
                    async for msg in client.receive_response():
                        first_messages.append(msg)

                assert any(isinstance(m, ResultMessage) for m in first_messages), (
                    "First turn should produce a ResultMessage"
                )

                # Second turn — this is where issue #558 would deadlock
                await client.query("What is 3+3? Answer in one word.")
                second_messages = []
                with anyio.fail_after(30):
                    async for msg in client.receive_response():
                        second_messages.append(msg)

                assert any(isinstance(m, ResultMessage) for m in second_messages), (
                    "Second turn should produce a ResultMessage (no deadlock)"
                )

        anyio.run(_test)

    def test_multi_turn_with_tool_use(self):
        """Multi-turn conversation where first turn uses tools.

        Tool use generates additional messages (tool_use, tool_result)
        that exercise the buffer more heavily.
        """
        if not _cli_available():
            pytest.skip("Claude CLI not installed")

        async def _test():
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model="claude-sonnet-4-5-20250929",
                max_turns=3,
                system_prompt="Respond very briefly.",
                allowed_tools=["Bash"],
            )

            async with ClaudeSDKClient(options=options) as client:
                # First turn — triggers tool use
                await client.query("Run: echo hello")
                first_messages = []
                with anyio.fail_after(60):
                    async for msg in client.receive_response():
                        first_messages.append(msg)

                assert any(isinstance(m, ResultMessage) for m in first_messages)

                # Second turn — should not deadlock even after tool use messages
                await client.query("What did the command output? One word answer.")
                second_messages = []
                with anyio.fail_after(30):
                    async for msg in client.receive_response():
                        second_messages.append(msg)

                assert any(isinstance(m, ResultMessage) for m in second_messages), (
                    "Second turn after tool use should complete without deadlock"
                )

        anyio.run(_test)

    def test_three_turn_conversation(self):
        """Three-turn conversation to stress-test buffer handling.

        If the deadlock is latent, it's more likely to manifest over
        multiple turns as unconsumed messages accumulate.
        """
        if not _cli_available():
            pytest.skip("Claude CLI not installed")

        async def _test():
            options = ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model="claude-sonnet-4-5-20250929",
                max_turns=2,
                system_prompt="Respond with exactly one word.",
            )

            async with ClaudeSDKClient(options=options) as client:
                prompts = [
                    "Say 'alpha'",
                    "Say 'beta'",
                    "Say 'gamma'",
                ]

                for i, prompt in enumerate(prompts):
                    await client.query(prompt)
                    messages = []
                    with anyio.fail_after(30):
                        async for msg in client.receive_response():
                            messages.append(msg)

                    assert any(isinstance(m, ResultMessage) for m in messages), (
                        f"Turn {i + 1} should produce a ResultMessage"
                    )

        anyio.run(_test)


if __name__ == "__main__":
    """Run directly: python tests/test_e2e_multiturn_deadlock.py"""
    os.environ["RUN_E2E_TESTS"] = "1"
    pytest.main([__file__, "-v", "-s"])
