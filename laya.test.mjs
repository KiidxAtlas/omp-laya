import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { test } from "bun:test";

const handlers = new Map();
const tools = [];
const commands = new Map();
const entries = [];
const messages = [];
const requests = [];
const statusUpdates = [];
const widgets = [];
const notifications = [];
const modelResolutions = [];
const providers = new Map();
const systemOneRequests = [];
const currentModel = {
	provider: "test",
	id: "current",
	reasoning: true,
	thinking: { efforts: ["low", "medium", "high"] },
};
const smolModel = {
	provider: "test",
	id: "smol",
	reasoning: true,
	thinking: { efforts: ["low", "medium", "high"] },
};
const slowModel = {
	provider: "test",
	id: "slow",
	reasoning: true,
	thinking: { efforts: ["low", "medium", "high"] },
};
const roleModels = { "@smol": smolModel, "@slow": slowModel };
let contextUsage;
let layaConfig = {};
let failChangePreflight = false;
let healthUnavailable = false;
const originalFetch = globalThis.fetch;
const extensionDirectory = fileURLToPath(new URL(".", import.meta.url));
const originalLayaPython = process.env.LAYA_PYTHON;
process.env.LAYA_PYTHON = extensionDirectory;
const extension = (await import("./laya.ts")).default;
if (originalLayaPython === undefined) delete process.env.LAYA_PYTHON;
else process.env.LAYA_PYTHON = originalLayaPython;
const ui = {
	notify(message, type) {
		notifications.push({ message, type });
	},
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
	model: currentModel,
	models: {
		current: () => currentModel,
		resolve(spec) {
			modelResolutions.push(spec);
			return roleModels[spec];
		},
	},
	getContextUsage: () => contextUsage,
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

async function prepare(prompt) {
	await emit("input", { text: prompt });
	await emit("turn_start", {});
	return emit("before_agent_start", { prompt, systemPrompt: [] });
}

test("Laya control plane enforces its decision policy", async () => {
	try {
		globalThis.fetch = async (url, init) => {
			if (String(url).endsWith("/health"))
				return Response.json({ status: healthUnavailable ? "starting" : "ok", device: "test" });
			const body = JSON.parse(init.body);
			if (String(url).endsWith("/v1/systemone")) {
				systemOneRequests.push(body);
				return Response.json({
					answers: {
						judge: {
							choice: "3",
							confidence: 0.9,
							probabilities: { "0": 0.01, "1": 0.02, "2": 0.07, "3": 0.9 },
						},
						correctness: { noul: 0.8, confidence: 0.8 },
					},
					usage: { input_tokens: 9, output_tokens: 0 },
				});
			}
			const requestText = String(body.state.input ?? "");
			requests.push(body);
			if (failChangePreflight && requestText.includes("write path=broken.json")) {
				throw new Error("simulated preflight outage");
			}
			const summary = requestText.includes("design.md") || requestText.includes("summary.md");
			const synthesis = summary ? 0.9 : 0.1;
			const destructive = requestText.includes("rm -rf");
			const difficulty = requestText.includes("invalid-prediction")
				? "invalid"
				: requestText.includes("hard")
					? "hard"
					: requestText.includes("easy") || requestText.includes("sensitive")
						? "easy"
						: "trivial";
			return Response.json({
				answers: {
					probe: { choice: "ready" },
					retrieval: { choice: summary ? "explore" : "none" },
					synthesis: { noul: synthesis },
					difficulty: { choice: difficulty },
					sensitive: { noul: requestText.includes("sensitive") ? 0.95 : 0.05 },
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
			registerCommand(name, command) {
				commands.set(name, command);
			},
			appendEntry(type, data) {
				entries.push({ type, data });
			},
			sendMessage(message, options) {
				messages.push({ message, options });
			},
			registerProvider(name, config) {
				providers.set(name, config);
			},
			pi: {
				settings: {
					getGlobalSettings: () => ({ laya: layaConfig }),
					getProjectSettings: () => ({}),
				},
			},
			logger: { debug() {}, info() {}, warn() {}, error() {} },
			getThinkingLevel: () => "medium",
		});
		const layaProvider = providers.get("laya-systemone");
		assert.ok(layaProvider, "the extension registers a selectable System One provider");
		assert.equal(layaProvider.models[0].id, "laya");
		assert.equal(layaProvider.api, "laya-systemone");
		const streamed = layaProvider.streamSimple(
			{
				provider: "laya-systemone",
				id: "laya",
				api: "laya-systemone",
				baseUrl: "http://127.0.0.1:8001/v1",
				reasoning: false,
			},
			{
				systemPrompt: ["Judge correctness and completeness."],
				messages: [{ role: "user", content: [{ type: "text", text: "Request: say yes. Answer: yes." }] }],
			},
		);
		const judgeEvents = [];
		for await (const event of streamed) judgeEvents.push(event);
		assert.equal(systemOneRequests.length, 1);
		assert.equal(systemOneRequests[0].questions.judge.type, "choice");
		assert.equal(systemOneRequests[0].questions.judge.instructions, "Judge correctness and completeness.");
		const longSystemStream = layaProvider.streamSimple(
			{
				provider: "laya-systemone",
				id: "laya",
				api: "laya-systemone",
				baseUrl: "http://127.0.0.1:8001/v1",
				reasoning: false,
			},
			{
				systemPrompt: ["global OMP prompt ".repeat(100)],
				messages: [{ role: "user", content: [{ type: "text", text: "Request: say yes. Answer: yes." }] }],
			},
		);
		for await (const _event of longSystemStream) {
			// Consume the provider result to exercise the request path.
		}
		assert.equal(systemOneRequests.length, 2);
		assert.equal(
			systemOneRequests[1].questions.judge.instructions,
			"Judge the assistant response in the supplied conversation for correctness, relevance, and completeness.",
		);
		assert.equal(systemOneRequests[0].questions.correctness.type, "noul");
		assert.deepEqual(systemOneRequests[0].state, [
			{ role: "user", content: "Request: say yes. Answer: yes." },
		]);
		const judgeMessage = judgeEvents.find(event => event.type === "done")?.message;
		assert.equal(JSON.parse(judgeMessage.content[0].text).score, 3);

		await emit("session_start", {});
		context.hasUI = true;
		await emit("input", { text: "Explain this setting without looking through the repository." });
		await emit("turn_start", {});
		const preparation = await emit("before_agent_start", {
			prompt: "Explain this setting without looking through the repository.",
			systemPrompt: [],
		});
		assert.equal(preparation, undefined, "routing advice must not override the active model or effort");

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
		const targetedSearch = await emit("tool_call", {
			toolName: "grep",
			toolCallId: "grep-targeted",
			input: { pattern: "knownSetting", path: "laya.ts" },
		});
		assert.equal(targetedSearch, undefined, "a no-discovery classification must still allow targeted grep");

		const numbered = (start, end) =>
			Array.from({ length: end - start + 1 }, (_, offset) => `${start + offset}:\tconst value${start + offset} = compute(${start + offset}); // padding to make realistic source`).join("\n");
		const readOutput = (body, footer) => `[laya_decide.ts#AB12]\n${body}\n\n[${footer}]`;
		const results = [
			["read-narrow", "laya_decide.ts:1-40", readOutput(numbered(1, 40), "Showing lines 1-40 of 400. Use :41 to continue")],
			["read-wide", "laya_decide.ts:1-80", readOutput(numbered(1, 80), "Showing lines 1-80 of 400. Use :81 to continue")],
			["read-body", "laya_decide.ts:100-160", readOutput(numbered(100, 160), "Showing lines 100-160 of 400. Use :161 to continue")],
			["read-summary", "laya_decide.ts", readOutput(`${numbered(100, 100)}\n…\n${numbered(160, 160)}`, "Bare code: declarations only")],
		];
		for (const [toolCallId, path, text] of results) {
			await emit("tool_call", { toolName: "read", toolCallId, input: { path } });
			await emit("tool_result", { toolName: "read", toolCallId, isError: false, content: [{ type: "text", text }] });
		}
		const testOutput = `${"✓ passing test case with a long descriptive name\n".repeat(60)}60 pass`;
		const contextMessages = [
			...results.map(([toolCallId, , text]) => ({
				role: "toolResult",
				toolName: "read",
				toolCallId,
				content: [{ type: "text", text }],
			})),
			{ role: "toolResult", toolName: "bash", toolCallId: "bash-a", content: [{ type: "text", text: testOutput }] },
			{ role: "toolResult", toolName: "bash", toolCallId: "small-a", content: [{ type: "text", text: "ok" }] },
			{ role: "toolResult", toolName: "bash", toolCallId: "bash-b", content: [{ type: "text", text: testOutput }] },
			{ role: "toolResult", toolName: "bash", toolCallId: "small-b", content: [{ type: "text", text: "ok" }] },
		];
		const rewritten = await emit("context", { messages: contextMessages });
		const rewrittenText = id => rewritten.messages.find(message => message.toolCallId === id).content[0].text;
		assert.equal(rewritten.messages.length, contextMessages.length, "pruning must never remove a tool result");
		assert.equal(rewrittenText("read-narrow"), results[0][2], "a wider read must not erase a non-identical earlier result");
		assert.equal(rewrittenText("read-body"), results[2][2], "a later elided summary read must not erase a detailed body");
		assert.match(rewrittenText("bash-a"), /identical to a later bash result/, "identical large output is elided with a pointer");
		assert.equal(rewrittenText("small-a"), "ok", "small outputs are not worth a prompt-cache rewrite");
		assert.equal(rewrittenText("bash-b"), testOutput, "the latest copy of identical output is kept");
		const rawSource = `[importantHeader]\n${numbered(1, 40)}\n[importantFooter]`;
		const rawResult = await emit("context", { messages: [
			{ role: "toolResult", toolName: "read", toolCallId: "read-narrow", content: [{ type: "text", text: rawSource }] },
			{ role: "toolResult", toolName: "read", toolCallId: "read-wide", content: [{ type: "text", text: numbered(1, 80) }] },
		] });
		assert.equal(rawResult, undefined, "bracket-only source lines must not be stripped to establish supersession");

		const duplicateRead = await emit("tool_call", {
			toolName: "read",
			toolCallId: "read-duplicate",
			input: { path: "laya_decide.ts:100-160" },
		});
		assert.equal(duplicateRead?.block, true, "an unchanged duplicate read is suppressed once");
		await emit("tool_result", { toolName: "eval", toolCallId: "eval-1", isError: false, content: [] });
		const rereadAfterEval = await emit("tool_call", {
			toolName: "read",
			toolCallId: "read-after-eval",
			input: { path: "laya_decide.ts:100-160" },
		});
		assert.equal(rereadAfterEval, undefined, "a tool that may mutate files invalidates duplicate-read suppression");

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
		assert.equal(fourthRead, undefined, "different ranges of one file remain accessible during synthesis");

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
		assert.equal(messages.at(-1)?.message.details.phase, "verification-required:edit-1");
		assert.equal(widgets.at(-1)?.content[0], "Laya · Targeted verification required");
		const remindersBeforeSecondEdit = messages.filter(item => item.message.details?.phase?.startsWith("verification-required:")).length;
		await emit("tool_call", { toolName: "edit", toolCallId: "edit-2", input: { path: "src/app.ts" } });
		await emit("tool_result", { toolName: "edit", toolCallId: "edit-2", isError: false, content: [{ type: "text", text: "updated again" }] });
		assert.equal(
			messages.filter(item => item.message.details?.phase?.startsWith("verification-required:")).length,
			remindersBeforeSecondEdit,
			"multiple pending source edits produce one reminder rather than one per edit",
		);
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
		for (const [index, command] of [
			"echo bun test",
			"tsc --noEmit | grep errors",
			"echo 'bun test'",
		].entries()) {
			const toolCallId = `not-verification-${index}`;
			await emit("tool_call", { toolName: "bash", toolCallId, input: { command } });
			await emit("tool_result", { toolName: "bash", toolCallId, isError: false, content: [{ type: "text", text: "exit 0" }] });
		}
		await emit("turn_end", { message: { role: "assistant", content: [] } });
		assert.equal(entries.filter(entry => entry.type === "laya-economy").at(-1)?.data.verificationRequired, true,
			"echoes and pipelines cannot clear a pending verification obligation");
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
		const noticesBeforeVerifiedAnswer = notifications.length;
		await emit("turn_start", {});
		await emit("turn_end", { message: { role: "assistant", content: [{ type: "text", text: "Tests passed." }] } });
		assert.equal(notifications.length, noticesBeforeVerifiedAnswer,
			"a new model iteration must not forget successful verification of the current mutation");

		await emit("tool_call", { toolName: "edit", toolCallId: "edit-after-check", input: { path: "src/app.ts" } });
		await emit("tool_result", { toolName: "edit", toolCallId: "edit-after-check", isError: false, content: [] });
		await emit("turn_start", {});
		await emit("turn_end", { message: { role: "assistant", content: [{ type: "text", text: "Tests passed." }] } });
		assert.equal(notifications.length, noticesBeforeVerifiedAnswer + 1,
			"verification before a new mutation cannot certify that mutation");

		await emit("input", { text: "Exercise a smoke scenario." });
		await emit("tool_call", { toolName: "bash", toolCallId: "smoke-unrecognized", input: { command: "node smoke.mjs" } });
		await emit("tool_result", { toolName: "bash", toolCallId: "smoke-unrecognized", isError: false, content: [{ type: "text", text: "scenario passed" }] });
		const entriesBeforeSmokeAnswer = entries.length;
		await emit("turn_start", {});
		await emit("turn_end", { message: { role: "assistant", content: [{ type: "text", text: "Verified the scenario." }] } });
		assert.ok(entries.slice(entriesBeforeSmokeAnswer).some(entry => entry.type === "laya-activity" && entry.data.phase === "verification-claim"),
			"unrecognized evidence receives a verification advisory");
		assert.equal(entries.filter(entry => entry.type === "laya-economy").at(-1)?.data.verificationLedger.verifiedVersion, undefined,
			"an unrecognized smoke is not automatically certified as a recognized check");

		await emit("input", { text: "Write a small source file." });
		await emit("before_agent_start", { prompt: "Write a small source file.", systemPrompt: [] });
		failChangePreflight = true;
		const outageCall = await emit("tool_call", {
			toolName: "write",
			toolCallId: "write-outage",
			input: { path: "broken.json" },
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

		layaConfig = {
			enabled: true,
			profile: "savings",
			modelRoutingEnabled: true,
			contextBudgetEnabled: false,
			economyEnabled: true,
		};
		async function recommendation(prompt, expectedModel, expectedEffort) {
			const result = await prepare(prompt);
			assert.equal(result?.model, undefined, "recommendations must not emit model overrides");
			assert.equal(result?.thinkingLevel, undefined, "recommendations must not emit effort overrides");
			const entry = entries.filter(entry => entry.type === "laya-routing").at(-1)?.data;
			assert.equal(entry?.advisory, true);
			assert.equal(entry?.activeModel, "test/current");
			assert.equal(entry?.activeThinkingLevel, "medium");
			assert.equal(entry?.recommendedModel, expectedModel);
			assert.equal(entry?.recommendedThinkingLevel, expectedEffort);
			return entry;
		}
		assert.equal((await recommendation("route easy", "test/smol", "low")).decision, "recommended-smol");
		await recommendation("route easy sensitive", "test/current", "high");

		layaConfig.profile = "balanced";
		await recommendation("route hard", "test/current", "high");

		layaConfig.profile = "safety-first";
		assert.equal((await recommendation("route hard", "test/slow", "high")).decision, "recommended-slow");
		await recommendation("route easy sensitive", "test/slow", "high");
		await recommendation("route trivial", "test/current", undefined);

		const resolutionsBeforeInvalid = modelResolutions.length;
		await recommendation("route invalid-prediction", undefined, undefined);
		roleModels["@slow"] = undefined;
		assert.equal((await recommendation("route hard", undefined, undefined)).decision, "unavailable");
		roleModels["@slow"] = slowModel;
		assert.ok(
			modelResolutions.slice(resolutionsBeforeInvalid).every(role => role === "@smol" || role === "@slow"),
			"routing resolves only the configured @smol and @slow roles",
		);

		layaConfig = {
			enabled: true,
			profile: "savings",
			modelRoutingEnabled: false,
			contextBudgetEnabled: true,
			economyEnabled: false,
			synthesisGuardEnabled: true,
			synthesisReadLimit: 3,
			synthesisReadBudget: 32,
		};
		contextUsage = { tokens: 800, contextWindow: 1_000, percent: 80 };
		const summaryPrompt = "Write summary.md about src/laya_decide.ts.";
		await prepare(summaryPrompt);
		for (const [index, path] of ["src/laya_decide.ts", "src/laya_decide.ts"].entries()) {
			const id = `explicit-pressure-read-${index}`;
			const blocked = await emit("tool_call", { toolName: "read", toolCallId: id, input: { path } });
			assert.equal(blocked, undefined, "a user-named path remains readable under context pressure");
			await emit("tool_result", {
				toolName: "read",
				toolCallId: id,
				isError: false,
				content: [{ type: "text", text: `explicit ${index}` }],
			});
		}
		await emit("tool_call", {
			toolName: "read",
			toolCallId: "unmentioned-pressure-read-1",
			input: { path: "src/not-mentioned.ts" },
		});
		await emit("tool_result", {
			toolName: "read",
			toolCallId: "unmentioned-pressure-read-1",
			isError: false,
			content: [{ type: "text", text: "unmentioned source" }],
		});
		const pressureLimitedRead = await emit("tool_call", {
			toolName: "read",
			toolCallId: "unmentioned-pressure-read-2",
			input: { path: "src/not-mentioned.ts" },
		});
		assert.equal(pressureLimitedRead?.block, true, "context pressure activates Savings' one-read-per-file cap");
		assert.equal(await emit("tool_call", {
			toolName: "read", toolCallId: "pressure-retry", input: { path: "src/not-mentioned.ts" },
		}), undefined, "a needed read can be reissued after the synthesis hold");
		await emit("turn_end", {
			message: {
				role: "assistant",
				content: [],
				usage: { input: 13, output: 5, cacheRead: 2, cacheWrite: 1, totalTokens: 21 },
			},
		});
		const pressureMetrics = entries.filter(entry => entry.type === "laya-economy").at(-1)?.data;
		assert.equal(pressureMetrics?.contextBudget.active, true, "the turn trace records active context-pressure budgets");
		assert.equal(pressureMetrics?.blockedOperations, 1, "the turn trace counts context-budget blocks");
		assert.equal(pressureMetrics?.providerUsage.inputTokens, 13, "provider input tokens remain actual, not cache-inclusive");
		assert.equal(pressureMetrics?.providerUsage.outputTokens, 5, "provider output tokens remain actual");
		assert.deepEqual(pressureMetrics?.contextUsage, contextUsage, "the turn trace records reported context usage");
		assert.equal(typeof pressureMetrics?.prunedTokens.estimated, "number", "pruned-token telemetry is explicitly estimated");
		assert.match(pressureMetrics?.prunedTokens.method ?? "", /characters.*4/);
		assert.ok(pressureMetrics?.layaInference.requests > 0, "the trace records Laya inference count and latency");

		contextUsage = undefined;
		layaConfig = {
			enabled: true,
			profile: "balanced",
			modelRoutingEnabled: true,
			contextBudgetEnabled: false,
			economyEnabled: true,
		};
		const layaCommand = commands.get("laya");
		await layaCommand.handler("keep-model", context);
		const bypassed = await prepare("route easy");
		assert.equal(bypassed, undefined, "the one-shot bypass leaves active model and effort alone");
		const bypassEntry = entries.filter(entry => entry.type === "laya-routing" && entry.data.decision === "bypass").at(-1);
		assert.ok(bypassEntry, "the one-shot recommendation bypass is preserved in a session entry");
		assert.equal(bypassEntry?.data.recommendedModel, undefined);
		assert.equal(bypassEntry?.data.recommendedThinkingLevel, undefined);
		assert.equal((await recommendation("route hard", "test/current", "high")).decision, "recommended-current",
			"the recommendation bypass expires after one agent turn");
		await emit("turn_end", {
			message: {
				role: "assistant",
				content: [],
				usage: { input: 7, output: 3, cacheRead: 0, cacheWrite: 0, totalTokens: 10 },
			},
		});
		await layaCommand.handler("stats", context);
		const recommendationMetrics = entries.filter(entry => entry.type === "laya-economy").at(-1)?.data.sessionMetrics;
		assert.equal(recommendationMetrics.recommendationsToSmol, 1);
		assert.equal(recommendationMetrics.recommendationsToSlow, 2);
		assert.equal(recommendationMetrics.explicitUserBypasses, 1);

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
		layaConfig = { enabled: true, serviceEnabled: true };
		healthUnavailable = true;
		await emit("session_start", {});
		healthUnavailable = false;
		assert.ok(
			entries.some(entry => entry.type === "laya-failure" && entry.data.label === "startup"),
			"an unlaunchable local service must be reported without crashing the host",
		);
	} finally {
		globalThis.fetch = originalFetch;
	}
});
