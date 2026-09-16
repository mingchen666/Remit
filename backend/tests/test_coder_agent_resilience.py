"""Regression tests for coder retries and resumable delivery finalization."""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config.setting import ApiType
from app.core.agents.agent import _message_tokens
from app.core.agents.coder_agent import (
    CoderAgent,
    CoderAgentRunError,
    CoderAgentUnavailableError,
)
from app.core.llm.types import StandardResponse, ToolCall


def _tool_response(call_id: str, code: str) -> StandardResponse:
    return StandardResponse(
        content="running",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="execute_code",
                arguments=json.dumps({"code": code}),
            )
        ],
    )


def _make_agent(
    *,
    max_retries: int = 2,
    max_chat_turns: int = 10,
    max_code_executions: int = 8,
) -> CoderAgent:
    model = MagicMock()
    model.api_type = ApiType.OPENAI_CHAT
    interpreter = MagicMock()
    interpreter.language = "matlab"
    interpreter.backend_name = "MATLAB test double"
    interpreter.notebook_serializer = MagicMock()
    interpreter.execute_code = AsyncMock()
    interpreter.get_created_images = AsyncMock(return_value=[])
    return CoderAgent(
        task_id="task-test",
        model=model,
        work_dir=".",
        max_retries=max_retries,
        max_chat_turns=max_chat_turns,
        max_code_executions=max_code_executions,
        code_interpreter=interpreter,
    )


class CoderAgentResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_tool_arguments_have_bounded_retries(self) -> None:
        for arguments in ('{}', '[]', '{', '{"code":null}', '{"code":42}', '{"code":" "}'):
            with self.subTest(arguments=arguments):
                agent = _make_agent()
                agent._inject_user_notes = AsyncMock()
                agent._chat = AsyncMock(return_value=StandardResponse(tool_calls=[
                    ToolCall(id="bad", name="execute_code", arguments=arguments)
                ]))
                with patch("app.core.agents.coder_agent.redis_manager.publish_message", new=AsyncMock()):
                    with self.assertRaisesRegex(CoderAgentRunError, "连续 3 次"):
                        await agent.run("eda", "eda")
                self.assertEqual(agent._chat.await_count, 3)
                agent.code_interpreter.execute_code.assert_not_awaited()
                self.assertEqual(agent.current_code_executions, 0)

    async def test_truncated_empty_tool_is_retried_before_execution(self) -> None:
        agent = _make_agent()
        agent.model.max_tokens = 4096
        agent._inject_user_notes = AsyncMock()
        agent._chat = AsyncMock(side_effect=[
            StandardResponse(finish_reason="max_tokens", tool_calls=[
                ToolCall(id="bad", name="execute_code", arguments="{}")
            ]),
            _tool_response("good", "print(1)"),
            StandardResponse(content="done"),
        ])
        agent.code_interpreter.execute_code.return_value = ("1", False, "")
        with patch("app.core.agents.coder_agent.redis_manager.publish_message", new=AsyncMock()):
            result = await agent.run("eda", "eda")
        self.assertEqual(result.code_response, "done")
        agent.code_interpreter.execute_code.assert_awaited_once_with("print(1)")
        self.assertEqual(agent._chat.await_args_list[1].kwargs["max_tokens"], 16384)

    def test_tool_arguments_are_counted_toward_context_budget(self) -> None:
        small = _message_tokens({"role": "assistant", "content": "ok"})
        with_code = _message_tokens(
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "execute_code",
                            "arguments": json.dumps({"code": "x" * 3000}),
                        },
                    }
                ],
            }
        )
        self.assertGreater(with_code, small + 500)

    def test_missing_interpreter_raises_explicit_runtime_error(self) -> None:
        agent = _make_agent()
        agent.code_interpreter = None

        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            agent._require_interpreter()

    async def test_transport_failure_is_not_returned_as_completed_code(self) -> None:
        agent = _make_agent(max_retries=2)
        agent._chat = AsyncMock(side_effect=ConnectionError("provider offline"))

        with (
            patch(
                "app.core.agents.coder_agent.redis_manager.publish_message",
                new=AsyncMock(),
            ),
            patch("app.core.agents.coder_agent.asyncio.sleep", new=AsyncMock()),
        ):
            with self.assertRaisesRegex(CoderAgentUnavailableError, "provider offline"):
                await agent.run("finish ques1", "ques1")
        self.assertEqual(agent._chat.await_count, 1)

    async def test_successful_execution_resets_consecutive_error_budget(self) -> None:
        agent = _make_agent(max_retries=2)
        agent._chat = AsyncMock(
            side_effect=[
                _tool_response("call-1", "bad1"),
                _tool_response("call-2", "good1"),
                _tool_response("call-3", "bad2"),
                _tool_response("call-4", "good2"),
                StandardResponse(content="done"),
            ]
        )
        agent.code_interpreter.execute_code.side_effect = [
            ("", True, "first MATLAB error"),
            ("first result", False, ""),
            ("", True, "second MATLAB error"),
            ("second result", False, ""),
        ]

        with patch(
            "app.core.agents.coder_agent.redis_manager.publish_message",
            new=AsyncMock(),
        ):
            result = await agent.run("finish ques1", "ques1")

        self.assertEqual(result.code_response, "done")

    async def test_chat_turn_budget_is_per_run_not_whole_workflow(self) -> None:
        agent = _make_agent(max_chat_turns=1)
        agent._chat = AsyncMock(
            side_effect=[
                StandardResponse(content="first"),
                StandardResponse(content="second"),
            ]
        )

        with patch(
            "app.core.agents.coder_agent.redis_manager.publish_message",
            new=AsyncMock(),
        ):
            first = await agent.run("eda", "eda")
            second = await agent.run("ques1 repair", "ques1")

        self.assertEqual(first.code_response, "first")
        self.assertEqual(second.code_response, "second")

    async def test_execution_budget_forces_summary_without_an_extra_tool_call(self) -> None:
        agent = _make_agent(max_code_executions=2, max_chat_turns=10)
        calls = 0

        async def respond(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _tool_response("call-1", "good1")
            if calls == 2:
                return _tool_response("call-2", "good2")
            if kwargs["tools"]:
                return _tool_response("call-3", "good3")
            return StandardResponse(content="summary from existing results")

        agent._chat = AsyncMock(side_effect=respond)
        agent.code_interpreter.execute_code.side_effect = [
            ("one", False, ""),
            ("two", False, ""),
        ]

        with patch(
            "app.core.agents.coder_agent.redis_manager.publish_message",
            new=AsyncMock(),
        ):
            result = await agent.run("finish ques1", "ques1")

        self.assertEqual(result.code_response, "summary from existing results")
        self.assertEqual(agent.code_interpreter.execute_code.await_count, 2)
        self.assertEqual(agent._chat.await_args_list[-1].kwargs["tools"], [])
        self.assertEqual(agent._chat.await_args_list[-1].kwargs["tool_choice"], "none")

    async def test_run_can_apply_a_smaller_repair_execution_budget(self) -> None:
        agent = _make_agent(max_code_executions=8, max_chat_turns=10)

        async def respond(**kwargs):
            if kwargs["tools"]:
                return _tool_response("repair-call", "inspect_or_repair()")
            return StandardResponse(content="repair budget exhausted")

        agent._chat = AsyncMock(side_effect=respond)
        agent.code_interpreter.execute_code.return_value = ("ok", False, "")

        with patch(
            "app.core.agents.coder_agent.redis_manager.publish_message",
            new=AsyncMock(),
        ):
            result = await agent.run(
                "repair only",
                "ques1",
                max_code_executions=2,
            )

        self.assertEqual(result.code_response, "repair budget exhausted")
        self.assertEqual(agent.code_interpreter.execute_code.await_count, 2)
        self.assertEqual(agent.max_code_executions, 8)

    async def test_warns_model_to_persist_contract_before_last_execution(self) -> None:
        agent = _make_agent(max_code_executions=3, max_chat_turns=10)
        histories: list[list[dict]] = []

        async def respond(**kwargs):
            histories.append([dict(item) for item in kwargs["history"]])
            if len(histories) <= 2:
                return _tool_response(f"call-{len(histories)}", "work()")
            return StandardResponse(content="files persisted")

        agent._chat = AsyncMock(side_effect=respond)
        agent.code_interpreter.execute_code.return_value = ("ok", False, "")

        with patch(
            "app.core.agents.coder_agent.redis_manager.publish_message",
            new=AsyncMock(),
        ):
            result = await agent.run("solve and persist", "ques1")

        self.assertEqual(result.code_response, "files persisted")
        self.assertTrue(
            any(
                "下一次执行必须优先" in str(item.get("content", ""))
                for item in histories[1]
            )
        )

    async def test_each_subtask_starts_with_fresh_model_history(self) -> None:
        agent = _make_agent()
        captured: list[list[dict]] = []

        async def finish(**kwargs):
            captured.append([dict(item) for item in kwargs["history"]])
            return StandardResponse(content="done")

        agent._chat = AsyncMock(side_effect=finish)
        with patch(
            "app.core.agents.coder_agent.redis_manager.publish_message",
            new=AsyncMock(),
        ):
            await agent.run("first prompt", "ques1")
            await agent.run("second prompt", "ques2")

        self.assertEqual(len(captured), 2)
        self.assertNotIn("first prompt", json.dumps(captured[1], ensure_ascii=False))
        self.assertIn("second prompt", json.dumps(captured[1], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
