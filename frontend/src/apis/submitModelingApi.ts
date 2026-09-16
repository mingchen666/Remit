import request from "@/utils/request";
import axios from "axios";

/** 单张赛题插图的识图结论 */
export interface ProblemFigureInsight {
	index: number;
	page_number: number;
	kind: string;
	figure_type: string;
	title: string;
	transcription: string;
	readable_values: string[];
	modeling_relevance: string;
	carries_information: boolean;
}

export interface ProblemPdfParseResult {
	filename: string;
	text: string;
	page_count: number;
	char_count: number;
	/** 识别出建模信息的插图数量 */
	figure_count: number;
	vision_status: "completed" | "partial" | "failed" | "skipped" | "disabled";
	vision_error: string;
	figures: ProblemFigureInsight[];
}

export type ExecutionBackend = "matlab" | "python";
export type CompTemplate = "CHINA" | "AMERICAN";

const COMP_TEMPLATE_ALIASES: Readonly<Record<string, CompTemplate>> = {
	国赛: "CHINA",
	美赛: "AMERICAN",
	CHINA: "CHINA",
	AMERICAN: "AMERICAN",
};

/** 将界面展示值转换为后端稳定线协议，未知值在发起请求前直接拒绝。 */
export function normalizeCompTemplate(value: string): CompTemplate {
	const normalized = COMP_TEMPLATE_ALIASES[value.trim()];
	if (!normalized) {
		throw new Error(`不支持的论文模板：${value}`);
	}
	return normalized;
}

/** 从任务提交失败响应中提取可操作信息，避免把协议错误误报为 API Key 错误。 */
export function explainModelingSubmissionFailure(error: unknown): string {
	if (!axios.isAxiosError(error)) {
		return error instanceof Error ? error.message : "任务提交失败，请稍后重试";
	}
	const detail = error.response?.data?.detail;
	if (typeof detail === "string" && detail.trim()) {
		return detail.trim();
	}
	if (Array.isArray(detail)) {
		const issues = detail.flatMap((item) => {
			if (item === null || typeof item !== "object") return [];
			const message = Reflect.get(item, "msg");
			const location = Reflect.get(item, "loc");
			if (typeof message !== "string") return [];
			const field = Array.isArray(location) ? location.at(-1) : null;
			return [typeof field === "string" ? `${field}：${message}` : message];
		});
		if (issues.length) return `提交内容校验失败：${issues.join("；")}`;
	}
	return error.response
		? `任务提交接口返回 ${error.response.status}`
		: "无法连接 Remit 后端，请检查服务状态";
}

export type ModelingSubmission = Readonly<{
	ques_all: string;
	user_requirements?: string;
	comp_template?: CompTemplate;
	format_output?: string;
	execution_backend?: ExecutionBackend;
}>;

export type ModelingTaskReceipt = {
	task_id: string;
	status: string;
};

/** 解析赛题 PDF，并返回可直接交给建模流程的完整文本。 */
export function parseProblemPdf(file: File) {
	const formData = new FormData();
	formData.append("file", file);

	// 识图会额外调用多模态模型，比纯文本解析慢得多，超时必须放宽
	return request.post<ProblemPdfParseResult>("/parse-problem-pdf", formData, {
		timeout: 300000,
	});
}

/**
 * 提交数学建模任务
 * @param problem 问题描述
 * @param files 上传的数据文件
 */
export function submitModelingTask(
	problem: ModelingSubmission,
	files?: File[],
) {
	const formData = new FormData();
	const fields = {
		ques_all: problem.ques_all,
		user_requirements: problem.user_requirements ?? "",
		comp_template: problem.comp_template ?? "CHINA",
		format_output: problem.format_output ?? "LaTeX",
		execution_backend: problem.execution_backend ?? "matlab",
	};
	for (const [name, value] of Object.entries(fields)) {
		formData.set(name, value);
	}

	for (const file of files ?? []) {
		formData.append("files", file);
	}

	return request.post<ModelingTaskReceipt>("/modeling", formData, {
		timeout: 30000,
	});
}
