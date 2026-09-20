import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "bun:test";

const handlers = new Map();
const tools = [];
const entries = [];
const messages = [];
const requests = [];
const statusUpdates = [];
const widgets = [];
let layaConfig = {};
let failChangePreflight = false;
const extension = (await import("./laya.ts")).default;
const originalFetch = globalThis.fetch;
const extensionDirectory = fileURLToPath(new URL(".", import.meta.url));
const ui = {
	notify() {},
	setStatus(key, text) {
		statusUpdates.push({ key, text });
	},
	setWidget(key, content, options) {
		widgets.push({ key, content, options });
	},
};
const context = {
	hasUI: false,
	cwd: extensionDirectory,
	ui,
	getSystemPrompt: () => [],
};
const z = {
	enum: values => values,
	object: schema => schema,
	string: () => ({ describe: () => ({}) }),
};

async function emit(event, payload) {
	let result;
	for (const handler of handlers.get(event) ?? []) {
		const next = await handler(payload, context);
		if (next !== undefined) result = next;
	}
	return result;
}

test("Laya control plane enforces its decision policy", async () => {
	try {
		globalThis.fetch = async (url, init) => {
			if (String(url).endsWith("/health")) return Response.json({ status: "ok", device: "test" });
			const body = JSON.parse(init.body);
			requests.push(body);
			if (failChangePreflight && body.state.input.includes("write path=broken.ts")) {
				throw new Error("simulated preflight outage");
			}
			const synthesis = body.state.input.includes("design.md") ? 0.9 : 0.1;
			const destructive = body.state.input.includes("rm -rf");
			return Response.json({
				answers: {
					probe: { choice: "ready" },
					retrieval: { choice: synthesis > 0.5 ? "explore" : "none" },
					synthesis: { noul: synthesis },
					destructive_op: { noul: destructive ? 0.95 : 0.1 },
					sensitive_data: { noul: 0.1 },
					data_loss_risk: { noul: destructive ? 0.95 : 0.1 },
					observable_behavior: { noul: 0.1 },
					regression_risk: { noul: 0.1 },
				},
			});
		};

		extension({
			zod: z,
			on(event, handler) {
				const registered = handlers.get(event) ?? [];
				registered.push(handler);
				handlers.set(event, registered);
			},
			registerTool(tool) {
				tools.push(tool);
			},
			appendEntry(type, data) {
				entries.push({ type, data });
			},
			sendMessage(message, options) {
				messages.push({ message, options });
			},
			pi: {
				settings: {
					getGlobalSettings: () => ({ laya: layaConfig }),
					getProjectSettings: () => ({}),
				},
			},
			logger: { debug() {}, info() {}, warn() {}, error() {} },
		});

		await emit("session_start", {});
		context.hasUI = true;
		await emit("input", { text: "Explain this setting without looking through the repository." });
		await emit("turn_start", {});
		const preparation = await emit("before_agent_start", {
			prompt: "Explain this setting without looking through the repository.",
			systemPrompt: [],
		});
		assert.equal(preparation, undefined, "routine turn preparation must not add a system prompt");

		const broadSearch = await emit("tool_call", {
			toolName: "glob",
			toolCallId: "glob-1",
			input: { path: "**/*.ts" },
		});
		assert.equal(broadSearch?.block, true, "a no-discovery decision must hold a broad search once");
		assert.equal(
			messages.at(-1)?.message.details.phase,
			"discovery:glob-1",
			"the transcript must explain the optimization that changed execution",
		);
		assert.match(messages.at(-1)?.message.content ?? "", /Skipped broad discovery/);
		assert.equal(
			statusUpdates.at(-1)?.text,
			"Discovery optimized",
			"the footer must mirror the current Laya decision",
		);
		assert.match(messages.at(-1)?.message.content ?? "", /broad discovery need/);
		assert.deepEqual(
			widgets.at(-1),
			{
				key: "laya-activity",
				content: [
					"Laya · Skipped broad discovery",
					"Laya's turn plan found no broad discovery need, so this search was held until the agent establishes one.",
				],
				options: { placement: "aboveEditor" },
			},
			"the current Laya decision must live in an integrated editor widget",
		);

		await emit("tool_call", { toolName: "read", toolCallId: "read-1", input: { path: "laya_decide.ts:1-5" } });
		await emit("tool_result", {
			toolName: "read",
			toolCallId: "read-1",
			isError: false,
			content: [{ type: "text", text: "first range" }],
		});
		await emit("tool_call", { toolName: "read", toolCallId: "read-2", input: { path: "laya_decide.ts:1-10" } });
		await emit("tool_result", {
			toolName: "read",
			toolCallId: "read-2",
			isError: false,
			content: [{ type: "text", text: "wider range" }],
		});

		const rewritten = await emit("context", {
			messages: [
				{
					role: "toolResult",
					toolName: "read",
					toolCallId: "read-1",
					content: [{ type: "text", text: "first range" }],
				},
				{
					role: "toolResult",
					toolName: "read",
					toolCallId: "read-2",
					content: [{ type: "text", text: "wider range" }],
				},
			],
		});
		assert.equal(
			rewritten.messages[0].content[0].text,
			"[laya] Superseded by a later unchanged read; use that result.",
		);

		await emit("input", { text: "Write design.md describing the site's style and design choices." });
		await emit("turn_start", {});
		await emit("before_agent_start", {
			prompt: "Write design.md describing the site's style and design choices.",
			systemPrompt: [],
		});
		for (const [index, selector] of ["20-24", "25-29", "30-34"].entries()) {
			const id = `synthesis-read-${index}`;
			await emit("tool_call", { toolName: "read", toolCallId: id, input: { path: `laya_decide.ts:${selector}` } });
			await emit("tool_result", {
				toolName: "read",
				toolCallId: id,
				isError: false,
				content: [{ type: "text", text: selector }],
			});
		}
		await emit("turn_end", {
			message: {
				role: "assistant",
				content: [],
				usage: { input: 10, output: 2, cacheRead: 3, cacheWrite: 1, totalTokens: 16 },
			},
		});
		assert.equal(
			entries.at(-1)?.data.synthesisReadCount,
			3,
			"economy telemetry must retain synthesis evidence usage",
		);
		const fourthRead = await emit("tool_call", {
			toolName: "read",
			toolCallId: "synthesis-read-4",
			input: { path: "laya_decide.ts:35-39" },
		});
		assert.equal(fourthRead?.block, true, "a synthesis run must stop rereading the same unchanged file");

		await emit("turn_end", {
			message: {
				role: "assistant",
				content: [],
				usage: { input: 10, output: 2, cacheRead: 3, cacheWrite: 1, totalTokens: 16 },
			},
		});
		assert.equal(
			entries.at(-1)?.data.synthesisReadCount,
			3,
			"economy telemetry must retain synthesis evidence usage",
		);
		assert.equal(tools.length, 2, "the Laya analysis and decision tools remain registered");

		layaConfig = { policyMode: "enforce" };
		await emit("input", { text: "Fix the observable UI behavior in src/app.ts." });
		await emit("turn_start", {});
		await emit("before_agent_start", { prompt: "Fix the observable UI behavior in src/app.ts.", systemPrompt: [] });
		const editCall = await emit("tool_call", {
			toolName: "edit",
			toolCallId: "edit-1",
			input: { path: "src/app.ts" },
		});
		assert.equal(editCall, undefined, "a low-risk edit preflight must allow the operation");
		const editResult = await emit("tool_result", {
			toolName: "edit",
			toolCallId: "edit-1",
			isError: false,
			content: [{ type: "text", text: "updated component" }],
		});
		assert.match(
			editResult.content[0].text,
			/narrowest relevant check/,
			"a Laya test-worthiness result must reach the agent before it can finish",
		);
		assert.ok(
			requests.some(request => Object.hasOwn(request.questions, "observable_behavior")),
			"state-changing work must invoke Laya's combined change preflight",
		);
		assert.equal(messages.at(-1)?.message.details.phase, "verification-required:edit-1");
		assert.equal(widgets.at(-1)?.content[0], "Laya · Targeted verification required");
		await emit("turn_end", {
			message: {
				role: "assistant",
				content: [],
				usage: { input: 10, output: 2, cacheRead: 3, cacheWrite: 1, totalTokens: 16 },
			},
		});
		assert.equal(
			messages.at(-1)?.message.details.phase,
			"verification-pending",
			"a missing post-change check must remain visible after the turn",
		);
		await emit("agent_end", { willContinue: false });
		assert.equal(
			messages.at(-1)?.message.customType,
			"laya-verification-enforcement",
			"enforce mode must schedule a hidden correction turn",
		);
		assert.equal(messages.at(-1)?.options.deliverAs, "nextTurn");
		assert.equal(messages.at(-1)?.options.triggerTurn, true);
		assert.ok(
			entries.some(entry => entry.type === "laya-verification" && entry.data.event === "enforced"),
			"the mutation ledger must preserve enforcement",
		);
		await emit("tool_call", {
			toolName: "bash",
			toolCallId: "verify-edit-1",
			input: { command: "bun test src/app.test.ts" },
		});
		await emit("tool_result", {
			toolName: "bash",
			toolCallId: "verify-edit-1",
			isError: false,
			content: [{ type: "text", text: "1 pass" }],
		});
		assert.equal(messages.at(-1)?.message.details.phase, "verification-satisfied:verify-edit-1");
		const messagesBeforeSettledEnd = messages.length;
		await emit("agent_end", { willContinue: false });
		assert.equal(
			messages.length,
			messagesBeforeSettledEnd,
			"a successful post-mutation verification must clear enforce mode",
		);

		await emit("input", { text: "Write a small source file." });
		await emit("before_agent_start", { prompt: "Write a small source file.", systemPrompt: [] });
		failChangePreflight = true;
		const outageCall = await emit("tool_call", {
			toolName: "write",
			toolCallId: "write-outage",
			input: { path: "broken.ts" },
		});
		failChangePreflight = false;
		assert.equal(outageCall, undefined, "an unavailable Laya preflight must fail open");
		const outage = entries.find(entry => entry.type === "laya-failure" && entry.data.label === "change preflight");
		assert.match(
			outage?.data.stack ?? "",
			/simulated preflight outage/,
			"the session trace must retain the preflight stack",
		);

		await emit("input", { text: "Remove the abandoned build directory." });
		await emit("before_agent_start", { prompt: "Remove the abandoned build directory.", systemPrompt: [] });
		const destructiveCall = await emit("tool_call", {
			toolName: "bash",
			toolCallId: "bash-dangerous",
			input: { command: "rm -rf ./src" },
		});
		assert.equal(destructiveCall?.block, true, "a Laya safety preflight must hold destructive work");
		assert.equal(messages.at(-1)?.message.details.phase, "held:bash-dangerous");
		layaConfig = { enabled: true, economyEnabled: false };
		await emit("input", { text: "Explain this setting without looking through the repository." });
		await emit("before_agent_start", {
			prompt: "Explain this setting without looking through the repository.",
			systemPrompt: [],
		});
		const unrestrictedSearch = await emit("tool_call", {
			toolName: "glob",
			toolCallId: "glob-economy-off",
			input: { path: "**/*.ts" },
		});
		assert.equal(unrestrictedSearch, undefined, "disabling economy guards must allow broad exploration");

		layaConfig = { enabled: false };
		const disabledGuard = await emit("tool_call", {
			toolName: "glob",
			toolCallId: "glob-disabled",
			input: { path: "**/*.ts" },
		});
		assert.equal(disabledGuard, undefined, "the master switch must bypass Laya tool interception");
		const disabledTool = await tools[0].execute("disabled-analysis", { mode: "routing", text: "test" });
		assert.match(
			disabledTool.content[0].text,
			/disabled in settings/,
			"the master switch must disable Laya analysis tools",
		);
	} finally {
		globalThis.fetch = originalFetch;
	}
});
