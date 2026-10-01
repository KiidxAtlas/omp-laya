import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, test } from "bun:test";
import {
	DecisionLog,
	MAX_STATE_CHARS,
	applyCalibration,
	evaluate,
	exportTraining,
	fitCalibration,
	labeledExamples,
	loadCalibration,
	makeDecision,
	questionDigest,
	redact,
	renderReport,
	saveCalibration,
} from "./decisions.ts";

const tempDirs = [];
async function tempDir() {
	const dir = await mkdtemp(join(tmpdir(), "laya-decisions-"));
	tempDirs.push(dir);
	return dir;
}
afterEach(async () => {
	await Promise.all(tempDirs.splice(0).map(dir => rm(dir, { recursive: true, force: true })));
});

const NOUL = { type: "noul", instructions: "Is this command destructive?" };
const CHOICE = { type: "choice", instructions: "Which route?", criteria: { fast: "cheap model", slow: "strong model", skip: "no model" } };
const SCORE = { type: "score", instructions: "How risky?", criteria: ["none", "low", "high"] };
const BASE_TIME = Date.parse("2026-09-01T00:00:00Z");

function noulDecision(id, p, { kind = "destructive", heuristic, question = NOUL } = {}) {
	return makeDecision({
		id,
		kind,
		mode: "shadow",
		state: `state ${id}`,
		questions: { q: question },
		answers: { q: { type: "noul", noul: p, answer_confidence: Math.max(p, 1 - p) } },
		latencyMs: 5,
		heuristic: heuristic === undefined ? undefined : { q: heuristic },
		now: new Date(BASE_TIME),
	});
}

function label(decisionId, expected, { qid = "q", source = "outcome", at = 0 } = {}) {
	return { type: "label", decisionId, qid, expected, source, ts: new Date(BASE_TIME + at).toISOString() };
}

/** `n` noul decisions: model says p (true) on all, `truthy` of them are labeled true. */
function noulSet(prefix, n, p, truthy, options) {
	const records = [];
	for (let i = 0; i < n; i++) {
		const id = `${prefix}${i}`;
		records.push(noulDecision(id, p, options?.(i)), label(id, i < truthy));
	}
	return records;
}

test("rotation keeps both generations readable and never loses the newest record", async () => {
	const file = join(await tempDir(), "nested", "decisions.jsonl");
	const log = new DecisionLog(file, { maxBytes: 600 });
	const ids = [];
	for (let i = 0; i < 12; i++) {
		const record = noulDecision(`d${i}`, 0.9);
		ids.push(record.id);
		await log.append(record);
	}
	const current = (await readFile(file, "utf8")).trim().split("\n");
	const rotated = (await readFile(`${file}.1`, "utf8")).trim().split("\n");
	assert.ok(current.length >= 1 && rotated.length >= 1);
	const read = (await log.read()).map(record => record.id);
	// Older generation first, contiguous, ending with the newest append.
	assert.deepEqual(read, ids.slice(ids.length - read.length));
	assert.equal(read.at(-1), "d11");
	// Two generations only: the oldest records were dropped with the replaced `.1`.
	assert.ok(read.length < ids.length);

	// A record larger than maxBytes on its own still lands.
	const big = makeDecision({ ...noulDecision("huge", 0.5), id: "huge", state: "x".repeat(2_000) });
	await log.append(big);
	const after = await log.read();
	assert.equal(after.at(-1).id, "huge");
	assert.equal(after.at(-2).id, "d11");
});

test("concurrent appends from separate instances never interleave lines", async () => {
	const file = join(await tempDir(), "decisions.jsonl");
	const first = new DecisionLog(file);
	const second = new DecisionLog(file);
	const appends = [];
	for (let i = 0; i < 200; i++) {
		const record = makeDecision({ ...noulDecision(`c${i}`, 0.7), id: `c${i}`, state: `${i}:`.padEnd(3_000, "y") });
		appends.push((i % 2 ? first : second).append(record));
	}
	await Promise.all(appends);
	const lines = (await readFile(file, "utf8")).split("\n").filter(Boolean);
	assert.equal(lines.length, 200);
	for (const line of lines) JSON.parse(line);
	assert.equal(new Set((await first.read()).map(record => record.id)).size, 200);
});

test("concurrent appends that trigger rotation keep a contiguous newest suffix", async () => {
	const file = join(await tempDir(), "decisions.jsonl");
	const log = new DecisionLog(file, { maxBytes: 20_000 });
	const ids = [];
	const appends = [];
	for (let i = 0; i < 100; i++) {
		ids.push(`r${i}`);
		appends.push(log.append(makeDecision({ ...noulDecision(`r${i}`, 0.7), id: `r${i}`, state: "z".repeat(3_000) })));
	}
	await Promise.all(appends);
	const read = (await log.read()).map(record => record.id);
	assert.ok(read.length > 1);
	assert.deepEqual(read, ids.slice(ids.length - read.length));
});

test("read skips blank and corrupt lines and treats missing files as empty", async () => {
	const dir = await tempDir();
	assert.deepEqual(await new DecisionLog(join(dir, "missing.jsonl")).read(), []);
	const file = join(dir, "decisions.jsonl");
	const decision = noulDecision("a", 0.9);
	await writeFile(file, `${JSON.stringify(decision)}\n\n{"type":"decision",\nnot json\n${JSON.stringify(label("a", true))}\n`);
	const records = await new DecisionLog(file).read();
	assert.deepEqual(
		records.map(record => record.type),
		["decision", "label"],
	);
});

test("user labels beat outcome labels regardless of order; latest wins within a source", () => {
	const decision = noulDecision("a", 0.9);
	const userFirst = [decision, label("a", false, { source: "user", at: 0 }), label("a", true, { at: 10_000 })];
	assert.deepEqual(
		labeledExamples(userFirst).map(e => [e.expected, e.source]),
		[[false, "user"]],
	);
	const userLast = [decision, label("a", true, { at: 10_000 }), label("a", false, { source: "user", at: 0 })];
	assert.deepEqual(
		labeledExamples(userLast).map(e => [e.expected, e.source]),
		[[false, "user"]],
	);
	// Latest timestamp wins even when it appears earlier in the log.
	const outcomes = [decision, label("a", false, { at: 5_000 }), label("a", true, { at: 1_000 })];
	assert.deepEqual(
		labeledExamples(outcomes).map(e => e.expected),
		[false],
	);
	assert.deepEqual(
		labeledExamples(userFirst, { source: "outcome" }).map(e => e.expected),
		[true],
	);
	// Unknown decisions and qids are dropped.
	assert.deepEqual(labeledExamples([decision, label("nope", true), label("a", true, { qid: "other" })]), []);
});

test("evaluate verdicts at the minN, baseline-tie, and heuristic boundaries", () => {
	// 29 < 30 examples.
	assert.equal(evaluate(noulSet("s", 29, 0.9, 15))[0].verdict, "insufficient-data");
	assert.equal(evaluate(noulSet("s", 29, 0.9, 15), { minN: 29 })[0].verdict, "no-better-than-baseline");

	// Always says true on an all-true set: accuracy 1 equals majority 1.
	const tie = evaluate(noulSet("t", 30, 0.9, 30))[0];
	assert.equal(tie.accuracy, 1);
	assert.equal(tie.majority, 1);
	assert.equal(tie.verdict, "no-better-than-baseline");

	// Balanced labels; model right on 24/30, heuristic right on all 30.
	const balanced = [];
	for (let i = 0; i < 30; i++) {
		const truth = i % 2 === 0;
		const modelRight = i < 24;
		const p = truth === modelRight ? 0.8 : 0.2;
		balanced.push(noulDecision(`b${i}`, p, { heuristic: truth }), label(`b${i}`, truth));
	}
	const withHeuristic = evaluate(balanced)[0];
	assert.equal(withHeuristic.accuracy, 0.8);
	assert.equal(withHeuristic.majority, 0.5);
	assert.equal(withHeuristic.chance, 0.5);
	assert.equal(withHeuristic.heuristic, 1);
	assert.equal(withHeuristic.verdict, "no-better-than-baseline");

	const withoutHeuristic = balanced.map(r => (r.type === "decision" ? { ...r, heuristic: undefined } : r));
	assert.equal(evaluate(withoutHeuristic)[0].verdict, "beats-baselines");
});

test("evaluate groups by kind and question version, sorted by kind then qid", () => {
	const changed = { ...NOUL, instructions: "Is this command irreversible?" };
	const records = [
		...noulSet("z", 3, 0.9, 3, () => ({ kind: "zeta" })),
		...noulSet("a", 2, 0.9, 2, () => ({ kind: "alpha" })),
		...noulSet("v", 4, 0.9, 4, () => ({ kind: "alpha", question: changed })),
	];
	const reports = evaluate(records);
	assert.deepEqual(
		reports.map(r => [r.kind, r.n]).sort(),
		[
			["alpha", 2],
			["alpha", 4],
			["zeta", 3],
		],
	);
	assert.deepEqual(
		reports.map(r => r.kind),
		["alpha", "alpha", "zeta"],
	);
	const choice = makeDecision({
		id: "c",
		kind: "route",
		mode: "on",
		state: "",
		questions: { route: CHOICE },
		answers: { route: { type: "choice", choice: "fast", probabilities: { fast: 0.6, slow: 0.3, skip: 0.1 } } },
		latencyMs: 1,
	});
	const [choiceReport] = evaluate([choice, label("c", "fast", { qid: "route" })]);
	assert.equal(choiceReport.chance, 1 / 3);
	assert.equal(choiceReport.accuracy, 1);
});

test("ECE is ~0 for a calibrated set and high for an overconfident wrong set", () => {
	// p=0.8 right 8/10 and p=0.6 right 6/10 (noul confidence is max(p, 1-p)).
	const calibrated = [...noulSet("h", 10, 0.8, 8), ...noulSet("m", 10, 0.4, 4)];
	const good = evaluate(calibrated, { minN: 1 })[0];
	assert.ok(good.ece < 1e-9, `ece ${good.ece}`);
	assert.ok(Math.abs(good.meanConfidence - 0.7) < 1e-9);

	const overconfident = noulSet("o", 20, 0.99, 0);
	const bad = evaluate(overconfident, { minN: 1 })[0];
	assert.equal(bad.accuracy, 0);
	assert.ok(bad.ece > 0.98, `ece ${bad.ece}`);
});

/** 100 noul decisions answered at 0.95 / 0.05 but right only 70% of the time. */
function overconfidentNoul() {
	const records = [];
	for (let i = 0; i < 100; i++) {
		const saysTrue = i % 2 === 0;
		const right = i % 10 < 7;
		records.push(noulDecision(`n${i}`, saysTrue ? 0.95 : 0.05), label(`n${i}`, saysTrue === right));
	}
	return records;
}

test("fitCalibration cools an overconfident noul question and applyCalibration moves it toward 0.5", () => {
	const records = overconfidentNoul();
	const calibration = fitCalibration(records, { now: new Date(BASE_TIME) });
	const entry = calibration[`q@${questionDigest(NOUL)}`];
	assert.ok(entry, JSON.stringify(calibration));
	assert.ok(entry.temperature > 1, `T ${entry.temperature}`);
	assert.ok(entry.nllAfter < entry.nllBefore);
	assert.equal(entry.n, 100);

	const raw = { q: { type: "noul", noul: 0.95, answer_confidence: 0.95 } };
	const calibrated = applyCalibration({ q: NOUL }, raw, calibration).q;
	assert.ok(calibrated.noul > 0.5 && calibrated.noul < 0.95, `${calibrated.noul}`);
	assert.equal(calibrated.answer_confidence, calibrated.noul);
	assert.equal(calibrated.calibrated, true);
	assert.equal(raw.q.noul, 0.95);
	// 70% accuracy at T=1 means the fitted p should land near 0.7.
	assert.ok(Math.abs(calibrated.noul - 0.7) < 0.01, `${calibrated.noul}`);

	assert.ok(evaluate(records, { calibration })[0].ece < evaluate(records)[0].ece);
	assert.deepEqual(fitCalibration(records.slice(0, 40)), {});
});

test("calibrated choice keeps argmax and score becomes the expected level", () => {
	const records = [];
	for (let i = 0; i < 60; i++) {
		const right = i % 2 === 0;
		records.push(
			makeDecision({
				id: `r${i}`,
				kind: "route",
				mode: "on",
				state: "",
				questions: { route: CHOICE, risk: SCORE },
				answers: {
					route: { type: "choice", choice: "fast", probabilities: { fast: 0.97, slow: 0.02, skip: 0.01 } },
					risk: { type: "score", score: 1.95, probabilities: { 0: 0.01, 1: 0.03, 2: 0.96 } },
				},
				latencyMs: 1,
			}),
			label(`r${i}`, right ? "fast" : "slow", { qid: "route" }),
			label(`r${i}`, right ? 2 : 1, { qid: "risk" }),
		);
	}
	const calibration = fitCalibration(records);
	assert.ok(calibration[`route@${questionDigest(CHOICE)}`].temperature > 1);
	assert.ok(calibration[`risk@${questionDigest(SCORE)}`].temperature > 1);

	const answers = {
		route: { type: "choice", choice: "fast", probabilities: { fast: 0.97, slow: 0.02, skip: 0.01 } },
		risk: { type: "score", score: 1.95, probabilities: { 0: 0.01, 1: 0.03, 2: 0.96 } },
	};
	const out = applyCalibration({ route: CHOICE, risk: SCORE }, answers, calibration);
	assert.equal(out.route.choice, "fast");
	assert.ok(out.route.probabilities.fast < 0.97 && out.route.probabilities.fast > out.route.probabilities.slow);
	assert.ok(Math.abs(Object.values(out.route.probabilities).reduce((a, b) => a + b) - 1) < 1e-12);
	assert.equal(out.route.answer_confidence, out.route.probabilities.fast);
	const p = out.risk.probabilities;
	assert.ok(Math.abs(out.risk.score - (p[1] + 2 * p[2])) < 1e-12);
	assert.ok(out.risk.score < 1.95);
	assert.equal(out.risk.calibrated, true);
});

test("a question whose instructions changed is not calibrated by the old entry", () => {
	const calibration = fitCalibration(overconfidentNoul());
	const changed = { ...NOUL, instructions: "Is this command irreversible?" };
	const answer = { type: "noul", noul: 0.95 };
	const out = applyCalibration({ q: changed }, { q: answer }, calibration);
	assert.equal(out.q, answer);
	assert.equal(out.q.calibrated, undefined);
});

test("calibration files round-trip; missing or corrupt files load as empty", async () => {
	const dir = await tempDir();
	const file = join(dir, "sub", "calibration.json");
	assert.deepEqual(await loadCalibration(file), {});
	const calibration = fitCalibration(overconfidentNoul(), { now: new Date(BASE_TIME) });
	await saveCalibration(file, calibration);
	assert.deepEqual(await loadCalibration(file), calibration);
	await writeFile(file, "{ not json");
	assert.deepEqual(await loadCalibration(file), {});
});

test("questionDigest ignores key order and tracks meaning", () => {
	const digest = questionDigest(CHOICE);
	assert.match(digest, /^[0-9a-f]{12}$/);
	assert.equal(
		questionDigest({ criteria: { skip: "no model", slow: "strong model", fast: "cheap model" }, instructions: "Which route?", type: "choice" }),
		digest,
	);
	assert.notEqual(questionDigest({ ...CHOICE, instructions: "Which route now?" }), digest);
	assert.notEqual(questionDigest({ ...CHOICE, criteria: { fast: "cheap model", slow: "strong model" } }), digest);
});

test("redact removes each secret kind", () => {
	const secrets = {
		openai: "sk-proj-abcdefghijklmnop1234",
		github: `ghp_${"a1B2".repeat(9)}`,
		// Assembled at runtime so secret scanners do not flag the fixture itself.
		aws: `${"AKIA"}IOSFODNN7EXAMPLE`,
		privateKey: `-----BEGIN RSA ${"PRIVATE"} KEY-----\nMIIEpAIBAAKCAQEA\nabcdef\n-----END RSA ${"PRIVATE"} KEY-----`,
		quotedPassword: 'password = "hunter2hunter2"',
		jsonKey: '"api_key": "abc123def"',
		envPassword: "DB_PASSWORD=hunter2",
		connection: "postgres://admin:s3cr3tpw@db.internal:5432/app",
		ssn: "123-45-6789",
		card: "4111 1111 1111 1111",
	};
	const leaked = {
		openai: "abcdefghijklmnop1234",
		github: "a1B2a1B2",
		aws: "IOSFODNN7EXAMPLE",
		privateKey: "MIIEpAIBAAKCAQEA",
		quotedPassword: "hunter2hunter2",
		jsonKey: "abc123def",
		envPassword: "hunter2",
		connection: "s3cr3tpw",
		ssn: "123-45-6789",
		card: "4111 1111",
	};
	for (const [kind, text] of Object.entries(secrets)) {
		const out = redact(`before ${text} after`);
		assert.ok(!out.includes(leaked[kind]), `${kind} leaked: ${out}`);
		assert.ok(out.includes("[REDACTED]"), `${kind}: ${out}`);
		assert.ok(out.startsWith("before ") && out.endsWith(" after"), `${kind}: ${out}`);
	}
	assert.equal(redact("postgres://admin:s3cr3tpw@db"), "postgres://admin:[REDACTED]@db");
	assert.equal(redact("DB_PASSWORD=hunter2 make"), "DB_PASSWORD=[REDACTED] make");
});

test("redact leaves ordinary code untouched", () => {
	const code = [
		"const password = await prompt();",
		"if (password === input) return token;",
		"type Creds = { token: string; apiKey?: string };",
		"const maxTokens = 4096; const startedAt = 1790834939096;",
		'const id = "550e8400-e29b-41d4-a716-446655440000";',
		"fetch(`https://example.com/v1/models?limit=20`);",
		"git clone git@github.com:org/repo.git && rm -rf dist",
		"const date = '2026-09-30'; const skip = items.slice(1);",
	].join("\n");
	assert.equal(redact(code), code);
});

test("makeDecision redacts before truncating and fingerprints questions", () => {
	const decision = makeDecision({
		kind: "destructive",
		mode: "on",
		// Keys run across where a truncation cut would fall; cutting first would leave unredactable fragments.
		state: `${`sk-${"k".repeat(40)} `.repeat(300)}${"x".repeat(10_000)}\nlast line`,
		questions: { q: NOUL },
		answers: { q: { type: "noul", noul: 0.2 } },
		latencyMs: 12,
	});
	assert.ok(decision.state.length <= MAX_STATE_CHARS);
	assert.ok(!decision.state.includes("k"), decision.state.slice(0, 200));
	assert.ok(decision.state.includes("[REDACTED]"));
	assert.ok(decision.state.endsWith("last line"));
	assert.equal(decision.questions.q.digest, questionDigest(NOUL));
	assert.match(decision.id, /^[0-9a-f-]{36}$/);
});

test("renderReport prints a markdown row per question", () => {
	const reports = evaluate(noulSet("t", 30, 0.9, 30));
	const lines = renderReport(reports).split("\n");
	assert.equal(lines[0], "| kind | question | n | accuracy | majority | heuristic | chance | ECE | verdict |");
	assert.equal(lines.length, 3);
	assert.equal(lines[2], `| destructive | q@${questionDigest(NOUL)} | 30 | 1.00 | 1.00 | – | 0.50 | 0.10 | no-better-than-baseline |`);
});

test("exportTraining emits one-hot gold per labeled question only", () => {
	const decision = makeDecision({
		id: "x",
		kind: "mixed",
		mode: "shadow",
		state: "the state",
		questions: { yes: NOUL, no: NOUL, route: CHOICE, list: { type: "choice", instructions: "Pick", criteria: ["a", "b"] }, risk: SCORE, unlabeled: NOUL },
		answers: {},
		latencyMs: 1,
	});
	const unlabeledDecision = noulDecision("y", 0.5);
	const rows = exportTraining([
		decision,
		unlabeledDecision,
		label("x", true, { qid: "yes" }),
		label("x", false, { qid: "no" }),
		label("x", "slow", { qid: "route" }),
		label("x", "b", { qid: "list" }),
		label("x", 1, { qid: "risk" }),
	]);
	assert.equal(rows.length, 1);
	const [row] = rows;
	assert.equal(row.state, "the state");
	assert.deepEqual(Object.keys(row.questions).sort(), ["list", "no", "risk", "route", "yes"]);
	assert.deepEqual(row.questions.risk, SCORE);
	assert.deepEqual(row.gold, {
		yes: { probabilities: { true: 1, false: 0 } },
		no: { probabilities: { true: 0, false: 1 } },
		route: { probabilities: { fast: 0, slow: 1, skip: 0 } },
		list: { probabilities: { a: 0, b: 1 } },
		risk: { probabilities: { 0: 0, 1: 1, 2: 0 } },
	});
});
