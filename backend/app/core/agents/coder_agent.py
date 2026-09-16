"""编码 Agent：让模型写代码、驱动执行后端、按错误反思自愈。

主循环语义：模型每轮要么调用 ``execute_code`` 工具推进任务，
要么返回纯文本宣告完成；连续失败计入反思预算，
模型服务故障与代码执行故障分别记账、分别兜底。
"""

import asyncio
import json
from typing import Any

from app.config.setting import settings
from app.core.activity import publish_activity
from app.core.agents.agent import Agent
from app.core.functions import get_coder_tools
from app.core.llm.llm import LLM
from app.core.prompts.shared import get_reflection_prompt
from app.core.prompts.coder import get_coder_prompt
from app.core.structured_output import (
    configured_output_budget,
    expanded_output_budget,
    response_was_truncated,
)
from app.schemas.A2A import CoderToWriter
from app.schemas.response import InterpreterMessage, SystemMessage
from app.services.redis_manager import redis_manager
from app.tools.base_interpreter import BaseCodeInterpreter
from app.utils.common_utils import get_current_files
from app.utils.log_util import logger


class CoderAgentRunError(RuntimeError):
    """编码阶段未能完成；质量门不得把它当成有效产出验收。"""


class CoderAgentUnavailableError(CoderAgentRunError):
    """模型服务在多次瞬断重试后仍不可用。"""


class CoderCodeExecutionError(CoderAgentRunError):
    """生成的代码持续执行失败，期间没有任何一次成功运行。"""


class CoderAgentBudgetError(CoderAgentRunError):
    """代码手达到硬轮次或执行次数上限，防止成功调用掩盖失控循环。"""


class CoderAgent(Agent):
    """以工具调用循环推进编码任务的 Agent。"""

    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,
        max_chat_turns: int | None = settings.MAX_CHAT_TURNS,
        max_retries: int | None = settings.MAX_RETRIES,
        max_code_executions: int | None = settings.MAX_CODE_EXECUTIONS_PER_RUN,
        code_interpreter: BaseCodeInterpreter | None = None,
        context_window: int = 128000,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        language = code_interpreter.language if code_interpreter else "python"
        super().__init__(
            task_id,
            model,
            context_window,
            cancel_event=cancel_event,
            system_prompt=get_coder_prompt(language),
        )
        self.work_dir = work_dir
        self.max_chat_turns = max_chat_turns or settings.MAX_CHAT_TURNS or 20
        self.max_retries = max_retries or settings.MAX_RETRIES or 3
        self.max_code_executions = (
            max_code_executions or settings.MAX_CODE_EXECUTIONS_PER_RUN
        )
        self.current_chat_turns = 0
        self.current_code_executions = 0
        self.is_first_run = True
        self.code_interpreter = code_interpreter

    # ---- 主循环 ----

    async def run(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        prompt: str,
        subtask_title: str,
        max_code_executions: int | None = None,
    ) -> CoderToWriter:
        """推进一个编码子任务直到模型宣告完成。

        Args:
            prompt: 子任务说明。
            subtask_title: 小节标题，用于 notebook 分段与输出归集。
            max_code_executions: 本轮可选的更小执行上限，用于只需修补现有
                产物的断点续跑；不能突破 Agent 的全局安全上限。

        Returns:
            编码结论与产出图片清单。

        Raises:
            CoderAgentUnavailableError: 模型服务连续不可达。
            CoderCodeExecutionError: 代码连续执行失败。
        """
        logger.info(f"{self.__class__.__name__}:开始:执行子任务: {subtask_title}")
        interpreter = self._require_interpreter()
        # 每个小问使用独立模型上下文；运行状态由 notebook/checkpoint 承接，
        # 避免前一问的大段代码把后续提示膨胀到上下文极限。
        self.chat_history = []
        self.current_token_count = 0
        self.is_first_run = True
        # 预算按单次调用计，跨小问 / 修复尝试不共享。
        self.current_chat_turns = 0
        self.current_code_executions = 0
        execution_limit = self.max_code_executions
        if max_code_executions is not None:
            execution_limit = max(1, min(max_code_executions, execution_limit))
        interpreter.add_section(subtask_title)
        interpreter.notebook_serializer.add_markdown_segmentation_to_notebook(
            "以下代码与输出属于该工作流节点，可按此标题在 notebook 中定位。",
            subtask_title,
        )

        # 统一 OpenAI 工具格式：Provider 各自转换，
        # 备用模型跨协议切换时工具形状才不会失配
        tools = get_coder_tools(interpreter.language, anthropic=False)

        await self._prime_history(prompt)

        retry_count = 0
        last_error = ""
        last_source = ""

        while True:
            if self.current_code_executions >= execution_limit:
                return await self._finalize_at_execution_limit(
                    interpreter, subtask_title, execution_limit
                )
            self._enforce_budget(retry_count, last_error, last_source)
            self.current_chat_turns += 1
            await self._inject_user_notes()
            await publish_activity(
                self.task_id,
                f"代码手正在思考下一步（{subtask_title} 第 {self.current_chat_turns} 轮）",
                category="llm",
            )

            try:
                response = await self._call_model(tools)
            except CoderAgentRunError:
                raise
            except Exception as exc:
                # LLM.chat 已拥有网络重试和备用模型切换权；这里再次重试会把
                # 4 次网关请求乘成 12 次，因此只负责转换为可续跑的阶段错误。
                message = (
                    f"代码手模型服务在内部重试后仍不可用：{exc}。"
                    "已生成文件均已保留，可从当前节点续跑。"
                )
                logger.error(message)
                raise CoderAgentUnavailableError(message) from exc

            if not response.tool_calls:
                # 没有工具调用 = 模型宣告任务完成
                logger.info("没有工具调用，任务完成")
                await publish_activity(
                    self.task_id,
                    f"{subtask_title} 的代码工作完成，进入质量检查",
                    category="gate",
                )
                return CoderToWriter(
                    code_response=response.content,
                    created_images=await interpreter.get_created_images(subtask_title),
                )

            outcome = await self._handle_tool_call(response, interpreter)
            if outcome == "ok":
                retry_count, last_error, last_source = 0, "", ""
                remaining_executions = (
                    execution_limit - self.current_code_executions
                )
                if 0 < remaining_executions <= 2:
                    # 不能等预算归零后才要求总结：质量报告等契约文件必须由
                    # execute_code 真正落盘。提前保留最后一到两次调用，让模型
                    # 把当前内核中的真实中间结果持久化并回读，而不是在最终文本
                    # 中“算完了”却因缺文件再次进入整轮重试。
                    await self.append_chat_history(
                        {
                            "role": "user",
                            "content": (
                                f"本轮只剩 {remaining_executions} 次 execute_code。"
                                "停止新增探索和改进模型；下一次执行必须优先把当前"
                                "内核中的真实结果写入提示中要求的全部必需交付文件，"
                                "并在同一次执行末尾回读、检查文件存在性与 JSON/CSV"
                                "结构。若证据不足，按协议如实写失败或人工复核状态，"
                                "不得把落盘动作留到预算耗尽后的文字总结。"
                            ),
                        }
                    )
                continue
            if outcome is not None:
                # outcome 是 (错误详情)，走反思路径
                retry_count += 1
                last_source, last_error = "execution", outcome
                await self._notify("代码手反思纠正错误", "error")
                await publish_activity(
                    self.task_id,
                    f"代码报错，正在自动修复（{retry_count}/{self.max_retries or '∞'}）",
                    category="repair",
                    detail=outcome[:160],
                )
                await self.append_chat_history(
                    {
                        "role": "user",
                        "content": get_reflection_prompt(outcome, self._last_code),
                    }
                )

    # ---- 步骤拆分 ----

    def _require_interpreter(self) -> BaseCodeInterpreter:
        if self.code_interpreter is None:
            raise RuntimeError("code_interpreter is not initialized")
        return self.code_interpreter

    async def _prime_history(self, prompt: str) -> None:
        """首次运行时铺垫系统提示与数据清单，再注入本轮任务。"""
        if self.is_first_run:
            logger.info("首次运行，添加系统提示和数据集文件信息")
            self.is_first_run = False
            await self._ensure_system_prompt()
        dataset_files = get_current_files(self.work_dir, "data")
        await self.append_chat_history(
            {"role": "user", "content": f"可用数据文件清单：{dataset_files}"}
        )
        await self.append_chat_history({"role": "user", "content": prompt})

    def _enforce_budget(
        self, retry_count: int, last_error: str, last_source: str
    ) -> None:
        """两类预算耗尽时抛出对应异常；轮次预算共用通用异常。"""
        if retry_count >= self.max_retries:
            if last_source == "model":
                message = (
                    f"代码手模型服务连接连续失败 {self.max_retries} 次：{last_error}。"
                    "已生成文件均已保留，可从当前节点续跑。"
                )
                error_type: type[CoderAgentRunError] = CoderAgentUnavailableError
            else:
                message = (
                    f"代码手连续执行失败 {self.max_retries} 次：{last_error}。"
                    "已生成文件均已保留，可从当前节点续跑。"
                )
                error_type = CoderCodeExecutionError
            logger.error(message)
            raise error_type(message)

        if self.current_chat_turns >= self.max_chat_turns:
            logger.error(f"超过最大聊天次数: {self.max_chat_turns}")
            raise CoderAgentBudgetError(
                f"代码手达到最大对话轮次 {self.max_chat_turns}，任务尚未完成；"
                "已保留当前产物和 checkpoint。"
            )

    async def _notify(self, content: str, level: str) -> None:
        await redis_manager.publish_message(
            self.task_id,
            SystemMessage(content=content, type=level),  # type: ignore[arg-type]
        )

    async def _call_model(
        self, tools: list[dict], tool_choice: str = "auto"
    ) -> Any:
        """只返回完整且可执行的响应，协议重试最多三次。"""
        budget = configured_output_budget(self.model)
        for attempt in range(3):
            response = await self._chat(
                history=self.chat_history,
                tools=tools,
                tool_choice=tool_choice,
                agent_name=self.__class__.__name__,
                max_tokens=budget,
            )
            error = ""
            if response_was_truncated(response, budget):
                error = "模型输出达到上限被截断"
                budget = expanded_output_budget(budget)
            elif response.tool_calls:
                if len(response.tool_calls) != 1 or not tools:
                    error = "当前响应必须只包含一个允许的工具调用"
                else:
                    try:
                        self._validated_code(response.tool_calls[0])
                    except ValueError as exc:
                        error = str(exc)
            elif not isinstance(response.content, str) or not response.content.strip():
                error = "模型未返回代码或有效总结"
            if not error:
                return response
            logger.warning(f"代码手响应校验失败 ({attempt + 1}/3): {error}")
            if attempt == 2:
                raise CoderAgentRunError(
                    f"代码手连续 3 次响应不完整或参数无效：{error}；可从当前节点续跑"
                )
            # 不把不完整工具调用放入历史，避免悬空 tool_call 或执行半段代码。
            await self.append_chat_history({
                "role": "user",
                "content": (
                    f"上次响应未执行：{error}。请缩短输出并完整重发。"
                    + ('只调用一次 execute_code，参数为含非空字符串 code 的 JSON 对象。'
                       if tools else "工具已禁用，请基于已有结果给出完整总结。")
                ),
            })
            await publish_activity(self.task_id, "模型响应不完整，正在重新生成", category="repair")

    @staticmethod
    def _validated_code(tool_call: Any) -> str:
        """校验不可信工具参数，避免缺字段或错误类型进入执行器。"""
        if tool_call.name != "execute_code":
            raise ValueError("只允许 execute_code 工具")
        if not isinstance(tool_call.arguments, str) or len(tool_call.arguments) > 1_000_000:
            raise ValueError("工具参数必须为不超过 1000000 字符的 JSON 字符串")
        try:
            arguments = json.loads(tool_call.arguments)
        except (ValueError, RecursionError) as exc:
            raise ValueError("工具参数不是有效 JSON") from exc
        code = arguments.get("code") if isinstance(arguments, dict) else None
        if not isinstance(code, str) or not code.strip():
            raise ValueError("execute_code 缺少非空字符串 code 参数")
        return code

    async def _finalize_at_execution_limit(
        self,
        interpreter: BaseCodeInterpreter,
        subtask_title: str,
        execution_limit: int,
    ) -> CoderToWriter:
        """达到执行上限后，基于已有结果完成总结而不再运行代码。"""
        logger.warning(
            f"代码执行已达到本轮上限 {execution_limit}，"
            "禁用工具并要求模型收口"
        )
        await self._inject_user_notes()
        await self.append_chat_history(
            {
                "role": "user",
                "content": (
                    f"你已用完本轮 {execution_limit} 次代码执行预算。"
                    "禁止继续调用工具。请仅依据已有代码输出和已生成文件，"
                    "立即给出本节点的最终结论；明确关键结果、产物路径以及"
                    "仍未完成或无法验证的事项，不得虚构结果。"
                ),
            }
        )
        await publish_activity(
            self.task_id,
            f"{subtask_title} 已达到代码执行上限，正在整理已有结果",
            category="gate",
        )
        try:
            response = await self._call_model([], tool_choice="none")
        except CoderAgentRunError:
            raise
        except Exception as exc:
            message = (
                f"代码手整理已有结果时模型服务不可用：{exc}。"
                "已生成文件均已保留，可从当前节点续跑。"
            )
            logger.error(message)
            raise CoderAgentUnavailableError(message) from exc

        if response.tool_calls:
            raise CoderAgentBudgetError(
                f"代码手达到本轮 {execution_limit} 次代码执行上限后"
                "仍请求执行工具；已保留当前产物和 checkpoint。"
            )

        await publish_activity(
            self.task_id,
            f"{subtask_title} 已基于现有结果完成整理，进入质量检查",
            category="gate",
        )
        return CoderToWriter(
            code_response=response.content,
            created_images=await interpreter.get_created_images(subtask_title),
        )

    _last_code: str = ""

    async def _handle_tool_call(
        self, response: Any, interpreter: BaseCodeInterpreter
    ) -> str | None:
        """执行模型请求的工具调用。

        Returns:
            ``"ok"`` 表示执行成功；错误详情字符串表示需要反思；
            ``None`` 表示非 execute_code 调用（忽略）。
        """
        tool_call = response.tool_calls[0]
        code = self._validated_code(tool_call)
        if tool_call.name != "execute_code":
            logger.info(f"忽略非代码工具调用: {tool_call.name}")
            return None

        if self.current_code_executions >= self.max_code_executions:
            raise CoderAgentBudgetError(
                f"代码手达到单节点 {self.max_code_executions} 次代码执行上限；"
                "禁止通过成功调用重置总预算，请拆分计算或改用低复杂度方案。"
            )
        self.current_code_executions += 1

        logger.info(f"调用工具: {tool_call.name}")
        await self._notify(f"代码手调用{tool_call.name}工具", "info")

        self._last_code = code
        await redis_manager.publish_message(
            self.task_id,
            InterpreterMessage(
                input={
                    "code": code,
                    "language": interpreter.language,
                    "backend": interpreter.backend_name,
                },
            ),
        )
        await self.append_chat_history(self._assistant_history_entry(response))

        await publish_activity(
            self.task_id,
            f"正在执行 {interpreter.backend_name} 代码…",
            category="code",
        )
        output_text, failed, error_detail = await interpreter.execute_code(code)

        tool_reply = {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": "execute_code",
            "content": error_detail if failed else output_text,
        }
        await self.append_chat_history(tool_reply)

        if failed:
            logger.warning(f"代码执行错误: {error_detail}")
            return error_detail

        await publish_activity(
            self.task_id, "代码执行成功，继续下一步", category="code"
        )
        return "ok"
