import {
	explainModelingSubmissionFailure,
	normalizeCompTemplate,
	submitModelingTask,
} from "@/apis/submitModelingApi";
import { beforeEach, describe, expect, it, vi } from "vitest";

const http = vi.hoisted(() => ({ post: vi.fn() }));

vi.mock("@/utils/request", () => ({ default: http }));

function submittedFormData(): FormData {
	return http.post.mock.calls.at(-1)?.[1] as FormData;
}

describe("建模任务计算环境", () => {
	beforeEach(() => {
		http.post.mockReset();
		http.post.mockResolvedValue({ data: { task_id: "task-1" } });
	});

	it("显式选择 Python 时发送项目级后端", async () => {
		await submitModelingTask({
			ques_all: "建立预测模型",
			execution_backend: "python",
		});

		expect(submittedFormData().get("execution_backend")).toBe("python");
	});

	it("未选择时仍以 MATLAB 为默认值", async () => {
		await submitModelingTask({ ques_all: "建立预测模型" });

		expect(submittedFormData().get("execution_backend")).toBe("matlab");
	});
});

describe("建模任务论文模板", () => {
	it.each([
		["国赛", "CHINA"],
		["美赛", "AMERICAN"],
		["CHINA", "CHINA"],
		["AMERICAN", "AMERICAN"],
	])("将界面值 %s 转换为后端线协议 %s", (input, expected) => {
		expect(normalizeCompTemplate(input)).toBe(expected);
	});

	it("拒绝未知模板值，避免向后端发送必然失败的请求", () => {
		expect(() => normalizeCompTemplate("未知模板")).toThrow("不支持的论文模板");
	});
});

describe("建模任务提交错误", () => {
	it("展示后端返回的具体字段校验错误", () => {
		const error = {
			isAxiosError: true,
			response: {
				status: 422,
				data: {
					detail: [
						{
							loc: ["body", "comp_template"],
							msg: "Input should be 'CHINA' or 'AMERICAN'",
						},
					],
				},
			},
		};

		expect(explainModelingSubmissionFailure(error)).toContain(
			"comp_template：Input should be 'CHINA' or 'AMERICAN'",
		);
	});
});
