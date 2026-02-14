"""Tests for message buffer deadlock fix (issue #558).

Reproduces the deadlock scenario where:
1. _read_messages() handles BOTH control routing AND message buffering
2. When the message buffer fills, _read_messages() blocks on send()
3. This prevents it from reading ANY transport data, including control messages
4. Control requests time out, and the system deadlocks

The fix (math.inf unbounded buffer) prevents _read_messages() from ever
blocking on send(), keeping control message routing alive.
"""

import asyncio
import json
import math
from unittest.mock import AsyncMock, Mock

import anyio
import pytest

from claude_agent_sdk._internal.query import Query


def _make_init_response(request_id: str) -> dict:
    """Build a control_response for an initialize request."""
    return {
        "type": "control_response",
        "response": {
            "request_id": request_id,
            "subtype": "success",
            "commands": [],
            "output_style": "default",
        },
    }


class TestMessageBufferDeadlock:
    """Test that an unbounded message buffer prevents deadlocks."""

    def test_bounded_buffer_blocks_producer(self):
        """Prove bounded anyio buffer blocks send() when full.

        This demonstrates the core mechanism behind issue #558:
        anyio.create_memory_object_stream with a finite max_buffer_size
        blocks the sender when the buffer is full and no consumer is draining.
        """

        async def _test():
            send_stream, recv_stream = anyio.create_memory_object_stream[dict](
                max_buffer_size=5
            )

            sent_count = 0

            async def producer():
                nonlocal sent_count
                for i in range(10):
                    await send_stream.send({"i": i})
                    sent_count += 1

            async with anyio.create_task_group() as tg:
                tg.start_soon(producer)
                # Give producer time to fill the buffer and block
                await asyncio.sleep(0.1)

                # Producer should be stuck after 5 sends
                assert sent_count == 5, (
                    f"Expected producer to block after 5 sends, but sent {sent_count}"
                )

                # Drain to unblock
                for _ in range(10):
                    await recv_stream.receive()

        anyio.run(_test)

    def test_unbounded_buffer_never_blocks_producer(self):
        """Prove math.inf buffer never blocks send().

        This is the key property of the fix: _read_messages() can always
        push messages into the buffer without waiting for a consumer.
        """

        async def _test():
            send_stream, recv_stream = anyio.create_memory_object_stream[dict](
                max_buffer_size=math.inf
            )

            sent_count = 0

            async def producer():
                nonlocal sent_count
                for i in range(200):
                    await send_stream.send({"i": i})
                    sent_count += 1

            async with anyio.create_task_group() as tg:
                tg.start_soon(producer)
                await asyncio.sleep(0.1)

                # ALL messages sent immediately — no blocking
                assert sent_count == 200

                for _ in range(200):
                    await recv_stream.receive()

        anyio.run(_test)

    def test_bounded_buffer_blocks_control_message_routing(self):
        """Prove bounded buffer causes _read_messages() to block, preventing
        control protocol messages from being processed.

        This is the exact deadlock from issue #558:
        1. Transport emits 110 regular messages (more than buffer=100)
        2. Nobody consumes from the buffer (simulates receive_response()
           having stopped after a ResultMessage)
        3. _read_messages() blocks on send() at message 101
        4. A control_response that comes AFTER message 100 is never read
        5. The pending control request that needs that response times out

        With unbounded buffer, _read_messages() never blocks, so it reads
        the control_response and the pending request resolves.
        """

        async def _test():
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.close = AsyncMock()
            mock_transport.end_input = AsyncMock()
            mock_transport.is_ready = Mock(return_value=True)

            written_messages: list[str] = []

            async def track_write(data: str):
                written_messages.append(data)

            mock_transport.write = AsyncMock(side_effect=track_write)

            # We'll send a control request with a known request_id,
            # and place its control_response AFTER 110 regular messages.
            # With bounded buffer (100), _read_messages blocks before
            # reaching the control_response.
            pending_request_id: str | None = None

            async def mock_read_messages():
                nonlocal pending_request_id
                # Wait for init request
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if written_messages:
                        break

                # Handle initialization
                for msg_str in written_messages:
                    try:
                        msg = json.loads(msg_str.strip())
                        if (
                            msg.get("type") == "control_request"
                            and msg.get("request", {}).get("subtype") == "initialize"
                        ):
                            yield _make_init_response(msg["request_id"])
                            break
                    except (json.JSONDecodeError, KeyError):
                        pass

                # Emit 110 regular messages — these fill the buffer
                for i in range(110):
                    yield {
                        "type": "system",
                        "subtype": "task_notification",
                        "task_id": f"task_{i}",
                        "status": "completed",
                        "summary": f"Task {i} done",
                    }

                # Now wait for a control request to appear (e.g., interrupt)
                # and emit its response. With bounded buffer, _read_messages()
                # is blocked on send() above and will never reach this yield.
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    for msg_str in written_messages:
                        try:
                            msg = json.loads(msg_str.strip())
                            if (
                                msg.get("type") == "control_request"
                                and msg.get("request", {}).get("subtype") == "interrupt"
                            ):
                                pending_request_id = msg["request_id"]
                                yield {
                                    "type": "control_response",
                                    "response": {
                                        "request_id": pending_request_id,
                                        "subtype": "success",
                                    },
                                }
                                return
                        except (json.JSONDecodeError, KeyError):
                            pass

            mock_transport.read_messages = mock_read_messages

            # Use a BOUNDED buffer (old behavior) by patching
            original_init = Query.__init__

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                # Override with bounded buffer
                self._message_send, self._message_receive = (
                    anyio.create_memory_object_stream[dict](max_buffer_size=100)
                )

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(Query, "__init__", patched_init)

                q = Query(transport=mock_transport, is_streaming_mode=True)
                await q.start()
                await q.initialize()

                # Don't consume any messages — simulates receive_response()
                # having stopped after ResultMessage.

                # Wait for buffer to fill (110 messages, buffer=100)
                await asyncio.sleep(0.5)

                # Now try to send an interrupt control request.
                # _read_messages() is blocked on send(), so it can't read
                # the control_response → this should time out.
                with pytest.raises(Exception, match="Control request timeout"):
                    await q._send_control_request({"subtype": "interrupt"}, timeout=1.0)

                await q.close()

        anyio.run(_test)

    def test_unbounded_buffer_allows_control_message_routing(self):
        """Prove unbounded buffer keeps control message routing alive.

        Same scenario as above, but with math.inf buffer (the fix).
        _read_messages() never blocks, so it reads the control_response
        and the interrupt request succeeds.
        """

        async def _test():
            mock_transport = AsyncMock()
            mock_transport.connect = AsyncMock()
            mock_transport.close = AsyncMock()
            mock_transport.end_input = AsyncMock()
            mock_transport.is_ready = Mock(return_value=True)

            written_messages: list[str] = []

            async def track_write(data: str):
                written_messages.append(data)

            mock_transport.write = AsyncMock(side_effect=track_write)

            async def mock_read_messages():
                # Wait for init request
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if written_messages:
                        break

                # Handle initialization
                for msg_str in written_messages:
                    try:
                        msg = json.loads(msg_str.strip())
                        if (
                            msg.get("type") == "control_request"
                            and msg.get("request", {}).get("subtype") == "initialize"
                        ):
                            yield _make_init_response(msg["request_id"])
                            break
                    except (json.JSONDecodeError, KeyError):
                        pass

                # Emit 110 regular messages
                for i in range(110):
                    yield {
                        "type": "system",
                        "subtype": "task_notification",
                        "task_id": f"task_{i}",
                        "status": "completed",
                        "summary": f"Task {i} done",
                    }

                # Wait for interrupt control request and respond
                for _ in range(200):
                    await asyncio.sleep(0.01)
                    for msg_str in written_messages:
                        try:
                            msg = json.loads(msg_str.strip())
                            if (
                                msg.get("type") == "control_request"
                                and msg.get("request", {}).get("subtype") == "interrupt"
                            ):
                                yield {
                                    "type": "control_response",
                                    "response": {
                                        "request_id": msg["request_id"],
                                        "subtype": "success",
                                    },
                                }
                                return
                        except (json.JSONDecodeError, KeyError):
                            pass

            mock_transport.read_messages = mock_read_messages

            # Use the real Query (which has math.inf buffer from the fix)
            q = Query(transport=mock_transport, is_streaming_mode=True)
            await q.start()
            await q.initialize()

            # Don't consume messages — buffer absorbs all 110

            # Wait for messages to arrive
            await asyncio.sleep(0.5)

            # Send interrupt — with unbounded buffer, _read_messages()
            # is NOT blocked, so it reads the control_response.
            # This should succeed (not timeout).
            await q._send_control_request({"subtype": "interrupt"}, timeout=3.0)

            # If we get here, the fix works — control routing stayed alive
            await q.close()

        anyio.run(_test)

    def test_query_class_uses_unbounded_buffer(self):
        """Verify the Query class is configured with math.inf buffer size."""
        transport = AsyncMock()
        q = Query(transport=transport, is_streaming_mode=True)
        stats = q._message_send.statistics()
        assert stats.max_buffer_size == math.inf, (
            f"Expected unbounded buffer (math.inf), got {stats.max_buffer_size}"
        )
