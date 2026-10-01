/**
 * Decision log, labels, evaluation and calibration for Laya answers.
 *
 * Every Laya call can be recorded as a `DecisionRecord`; outcomes or the user
 * later attach `LabelRecord`s. From the labeled pairs this module measures
 * accuracy against trivial baselines, fits per-question temperature scaling,
 * and exports supervised training rows. Pure apart from the log/calibration
 * file I/O.
 */
import { createHash, randomUUID } from "node:crypto";
import { appendFile, mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export type AnswerType = "noul" | "choice" | "score";
/** A `null` option description is the Laya/Jev convention for a catch-all option such as `other`. */
export type Question = { type: AnswerType; instructions: string; criteria?: Record<string, string | null> | string[] };
export type Answer = {
	type: AnswerType;
	noul?: number;
	choice?: string;
	score?: number;
	probabilities?: Record<string, number>;
	confidence?: number;
	answer_confidence?: number;
	[k: string]: unknown;
};
/** noul → boolean, choice → label, score → integer level. */
export type Expected = boolean | string | number;
export type DecisionRecord = {
	type: "decision";
	id: string;
	ts: string;
	kind: string;
	mode: "on" | "shadow";
	/** Already redacted and truncated by `makeDecision`. */
	state: string;
	questions: Record<string, Question & { digest: string }>;
	/** Raw answers as returned by the model (pre-calibration). */
	answers: Record<string, Answer>;
	model?: string;
	latencyMs: number;
	/** A deterministic baseline's prediction per qid, if the caller has one. */
	heuristic?: Record<string, Expected>;
	meta?: Record<string, unknown>;
};
export type LabelRecord = {
	type: "label";
	decisionId: string;
	qid: string;
	expected: Expected;
	source: "outcome" | "user";
	ts: string;
	note?: string;
};
export type LogRecord = DecisionRecord | LabelRecord;

type LabelSource = "outcome" | "user" | "any";

// ---------------------------------------------------------------------------
// Questions and records

/** JSON with object keys sorted at every depth, so equal questions hash equally regardless of key order. */
function canonicalJson(value: unknown): string {
	if (Array.isArray(value)) return `[${value.map(item => canonicalJson(item ?? null)).join(",")}]`;
	if (value !== null && typeof value === "object") {
		const entries = Object.entries(value)
			.filter(([, item]) => item !== undefined)
			.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
			.map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`);
		return `{${entries.join(",")}}`;
	}
	return JSON.stringify(value);
}

function isFiniteNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}

/** First 12 hex chars of sha256 over the canonical JSON of the question's meaning. */
export function questionDigest(q: Question): string {
	const canonical = canonicalJson({ type: q.type, instructions: q.instructions, criteria: q.criteria });
	return createHash("sha256").update(canonical).digest("hex").slice(0, 12);
}

const REDACTED = "[REDACTED]";

function passesLuhn(digits: string): boolean {
	let sum = 0;
	for (let i = 0; i < digits.length; i++) {
		let digit = digits.charCodeAt(digits.length - 1 - i) - 48;
		if (i % 2 === 1) {
			digit *= 2;
			if (digit > 9) digit -= 9;
		}
		sum += digit;
	}
	return sum % 10 === 0;
}

/**
 * Ordered so whole blocks go before their fragments; key/value forms keep the
 * key so the redacted state still reads as what it was.
 */
const REDACTIONS: Array<[RegExp, (match: string, ...groups: string[]) => string]> = [
	[/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)/g, () => REDACTED],
	// scheme://user:password@host
	[/\b([a-z][a-z0-9+.-]*:\/\/[^\s:/@]+:)[^\s@/]+@/gi, (_m, prefix) => `${prefix}${REDACTED}@`],
	// password = "value", "api_key": 'value'
	[
		/\b(\w*(?:password|passwd|pwd|secret|api[_-]?key|token)["']?\s*[=:]\s*)(["'])[^"'\n]+\2/gi,
		(_m, prefix, quote) => `${prefix}${quote}${REDACTED}${quote}`,
	],
	// Env/flag style: DB_PASSWORD=hunter2, --token=abc
	[
		/(^|[\s;&]|--?)(\w*(?:password|passwd|pwd|secret|api[_-]?key|token)=)(?![=\s"'])[^\s"';&]+/gim,
		(_m, lead, key) => `${lead}${key}${REDACTED}`,
	],
	[/\bsk-[A-Za-z0-9_-]{16,}/g, () => REDACTED],
	[/\bgh[pousr]_[A-Za-z0-9]{20,}/g, () => REDACTED],
	[/\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g, () => REDACTED],
	[/\b\d{3}-\d{2}-\d{4}\b/g, () => REDACTED],
	// Card-like digit runs; Luhn keeps timestamps and ids out.
	[/\b\d(?:[ -]?\d){12,15}\b/g, match => (passesLuhn(match.replace(/[ -]/g, "")) ? REDACTED : match)],
];

/** Replaces credentials and personal identifiers with "[REDACTED]". */
export function redact(text: string): string {
	let out = text;
	for (const [pattern, replace] of REDACTIONS) out = out.replace(pattern, replace);
	return out;
}

export const MAX_STATE_CHARS = 4_000;

/** Keeps the head and the tail: the tail usually holds the newest output. */
function truncateState(state: string): string {
	if (state.length <= MAX_STATE_CHARS) return state;
	const omitted = state.length - MAX_STATE_CHARS;
	const marker = `\n…[${omitted} chars truncated]…\n`;
	const keep = MAX_STATE_CHARS - marker.length;
	const head = Math.ceil(keep / 2);
	return state.slice(0, head) + marker + state.slice(state.length - (keep - head));
}

export function makeDecision(input: {
	kind: string;
	mode: "on" | "shadow";
	state: string;
	questions: Record<string, Question>;
	answers: Record<string, Answer>;
	model?: string;
	latencyMs: number;
	heuristic?: Record<string, Expected>;
	meta?: Record<string, unknown>;
	now?: Date;
	id?: string;
}): DecisionRecord {
	const questions: DecisionRecord["questions"] = {};
	for (const [qid, question] of Object.entries(input.questions)) {
		questions[qid] = { ...question, digest: questionDigest(question) };
	}
	const record: DecisionRecord = {
		type: "decision",
		id: input.id ?? randomUUID(),
		ts: (input.now ?? new Date()).toISOString(),
		kind: input.kind,
		mode: input.mode,
		// Redact before truncating so a cut can never leave a secret unrecognizable.
		state: truncateState(redact(input.state)),
		questions,
		answers: input.answers,
		latencyMs: input.latencyMs,
	};
	if (input.model !== undefined) record.model = input.model;
	if (input.heuristic !== undefined) record.heuristic = input.heuristic;
	if (input.meta !== undefined) record.meta = input.meta;
	return record;
}

/** The discrete answer the model gave; undefined when the answer is malformed. */
export function predicted(question: Question, answer: Answer): Expected | undefined {
	if (!answer || (answer.type !== undefined && answer.type !== question.type)) return undefined;
	switch (question.type) {
		case "noul":
			return isFiniteNumber(answer.noul) ? answer.noul >= 0.5 : undefined;
		case "choice":
			return typeof answer.choice === "string" ? answer.choice : undefined;
		case "score":
			return isFiniteNumber(answer.score) ? Math.round(answer.score) : undefined;
		default:
			return undefined;
	}
}

// ---------------------------------------------------------------------------
// Log file

const DEFAULT_MAX_BYTES = 20 * 1024 * 1024;
/** One write queue per resolved path, so separate instances cannot interleave either. */
const writeQueues = new Map<string, Promise<void>>();

function isMissing(error: unknown): boolean {
	return error instanceof Error && "code" in error && error.code === "ENOENT";
}

function isLogRecord(value: unknown): value is LogRecord {
	return (
		value !== null &&
		typeof value === "object" &&
		"type" in value &&
		(value.type === "decision" || value.type === "label")
	);
}

export class DecisionLog {
	readonly file: string;
	readonly maxBytes: number;

	constructor(file: string, options?: { maxBytes?: number }) {
		this.file = resolve(file);
		this.maxBytes = options?.maxBytes ?? DEFAULT_MAX_BYTES;
	}

	append(record: LogRecord): Promise<void> {
		const line = `${JSON.stringify(record)}\n`;
		const previous = writeQueues.get(this.file) ?? Promise.resolve();
		const run = previous.then(() => this.write(line));
		writeQueues.set(
			this.file,
			run.catch(() => {}),
		);
		return run;
	}

	private async write(line: string): Promise<void> {
		await mkdir(dirname(this.file), { recursive: true });
		let size = 0;
		try {
			size = (await stat(this.file)).size;
		} catch (error) {
			if (!isMissing(error)) throw error;
		}
		if (size > 0 && size + Buffer.byteLength(line) > this.maxBytes) {
			await rename(this.file, `${this.file}.1`);
		}
		await appendFile(this.file, line, "utf8");
	}

	/** Older generation first; blank and corrupt lines are skipped. */
	async read(): Promise<LogRecord[]> {
		const records: LogRecord[] = [];
		for (const path of [`${this.file}.1`, this.file]) {
			let text: string;
			try {
				text = await readFile(path, "utf8");
			} catch (error) {
				if (isMissing(error)) continue;
				throw error;
			}
			for (const line of text.split("\n")) {
				if (!line.trim()) continue;
				try {
					const parsed: unknown = JSON.parse(line);
					if (isLogRecord(parsed)) records.push(parsed);
				} catch {
					// A torn or hand-edited line costs one record, not the log.
				}
			}
		}
		return records;
	}
}

// ---------------------------------------------------------------------------
// Labels

export type LabeledExample = { decision: DecisionRecord; qid: string; expected: Expected; source: "outcome" | "user" };

function matchesType(question: Question, expected: Expected): boolean {
	switch (question.type) {
		case "noul":
			return typeof expected === "boolean";
		case "choice":
			return typeof expected === "string";
		case "score":
			return typeof expected === "number" && Number.isInteger(expected) && expected >= 0;
		default:
			return false;
	}
}

/** Whether `candidate` should replace `current` as the label for its (decision, qid). */
function supersedes(candidate: LabelRecord, current: LabelRecord): boolean {
	if (candidate.source !== current.source) return candidate.source === "user";
	// Equal timestamps: the later line in the log wins.
	return Date.parse(candidate.ts) >= Date.parse(current.ts);
}

/** One example per (decisionId, qid): user beats outcome; within a source the latest ts wins. */
export function labeledExamples(records: LogRecord[], options?: { source?: LabelSource }): LabeledExample[] {
	const source = options?.source ?? "any";
	const decisions = new Map<string, DecisionRecord>();
	for (const record of records) if (record.type === "decision") decisions.set(record.id, record);

	const winners = new Map<string, { label: LabelRecord; decision: DecisionRecord }>();
	for (const record of records) {
		if (record.type !== "label") continue;
		if (source !== "any" && record.source !== source) continue;
		const decision = decisions.get(record.decisionId);
		const question = decision?.questions[record.qid];
		if (!decision || !question || !matchesType(question, record.expected)) continue;
		const key = `${record.decisionId}\u0000${record.qid}`;
		const current = winners.get(key);
		if (!current || supersedes(record, current.label)) winners.set(key, { label: record, decision });
	}

	return [...winners.values()].map(({ label, decision }) => ({
		decision,
		qid: label.qid,
		expected: label.expected,
		source: label.source,
	}));
}

// ---------------------------------------------------------------------------
// Evaluation

export type QuestionReport = {
	qid: string;
	digest: string;
	kind: string;
	type: AnswerType;
	n: number;
	accuracy: number;
	majority: number;
	chance: number;
	/** Heuristic accuracy over the examples that carry `decision.heuristic[qid]`. */
	heuristic?: number;
	ece?: number;
	meanConfidence?: number;
	verdict: "insufficient-data" | "beats-baselines" | "no-better-than-baseline";
};

const DEFAULT_MIN_N = 30;
const ECE_BINS = 15;

function criteriaLabels(question: Question): string[] {
	const { criteria } = question;
	if (!criteria) return [];
	return Array.isArray(criteria) ? criteria.map(String) : Object.keys(criteria);
}

/** Option labels a question can take: noul true/false, choice labels, score levels "0".."k-1". */
function optionCount(question: Question): number {
	if (question.type === "noul") return 2;
	return criteriaLabels(question).length;
}

/** The model's confidence in its own answer, as a probability. */
function answerConfidence(question: Question, answer: Answer | undefined): number | undefined {
	if (!answer) return undefined;
	if (question.type === "noul") return isFiniteNumber(answer.noul) ? Math.max(answer.noul, 1 - answer.noul) : undefined;
	const values = Object.values(answer.probabilities ?? {}).filter(isFiniteNumber);
	return values.length ? Math.max(...values) : undefined;
}

function expectedCalibrationError(points: Array<{ confidence: number; correct: boolean }>): number {
	const bins = Array.from({ length: ECE_BINS }, () => ({ count: 0, confidence: 0, correct: 0 }));
	for (const point of points) {
		const bin = bins[Math.min(ECE_BINS - 1, Math.max(0, Math.floor(point.confidence * ECE_BINS)))];
		bin.count++;
		bin.confidence += point.confidence;
		if (point.correct) bin.correct++;
	}
	let ece = 0;
	for (const bin of bins) {
		if (!bin.count) continue;
		ece += (bin.count / points.length) * Math.abs(bin.correct / bin.count - bin.confidence / bin.count);
	}
	return ece;
}

export function evaluate(
	records: LogRecord[],
	options?: { minN?: number; source?: LabelSource; calibration?: Calibration },
): QuestionReport[] {
	const minN = options?.minN ?? DEFAULT_MIN_N;
	const groups = new Map<string, LabeledExample[]>();
	for (const example of labeledExamples(records, { source: options?.source })) {
		const { digest } = example.decision.questions[example.qid];
		const key = `${example.decision.kind}\u0000${example.qid}\u0000${digest}`;
		const group = groups.get(key);
		if (group) group.push(example);
		else groups.set(key, [example]);
	}

	const reports: QuestionReport[] = [];
	for (const examples of groups.values()) {
		const { decision, qid } = examples[0];
		const question = decision.questions[qid];
		const n = examples.length;
		let correct = 0;
		let heuristicSeen = 0;
		let heuristicCorrect = 0;
		const expectedCounts = new Map<string, number>();
		const points: Array<{ confidence: number; correct: boolean }> = [];

		for (const example of examples) {
			let answer = example.decision.answers[qid];
			if (answer && options?.calibration) {
				answer = applyCalibration({ [qid]: question }, { [qid]: answer }, options.calibration)[qid];
			}
			const isCorrect = answer !== undefined && predicted(question, answer) === example.expected;
			if (isCorrect) correct++;
			const expectedKey = JSON.stringify(example.expected);
			expectedCounts.set(expectedKey, (expectedCounts.get(expectedKey) ?? 0) + 1);
			const heuristic = example.decision.heuristic?.[qid];
			if (heuristic !== undefined) {
				heuristicSeen++;
				if (heuristic === example.expected) heuristicCorrect++;
			}
			const confidence = answerConfidence(question, answer);
			if (confidence !== undefined) points.push({ confidence, correct: isCorrect });
		}

		const accuracy = correct / n;
		const majority = Math.max(...expectedCounts.values()) / n;
		const k = optionCount(question);
		const chance = k > 0 ? 1 / k : 0;
		const heuristic = heuristicSeen ? heuristicCorrect / heuristicSeen : undefined;
		const report: QuestionReport = {
			qid,
			digest: question.digest,
			kind: decision.kind,
			type: question.type,
			n,
			accuracy,
			majority,
			chance,
			verdict:
				n < minN
					? "insufficient-data"
					: accuracy > Math.max(majority, chance, heuristic ?? 0)
						? "beats-baselines"
						: "no-better-than-baseline",
		};
		if (heuristic !== undefined) report.heuristic = heuristic;
		if (points.length) {
			report.ece = expectedCalibrationError(points);
			report.meanConfidence = points.reduce((sum, point) => sum + point.confidence, 0) / points.length;
		}
		reports.push(report);
	}

	return reports.sort(
		(a, b) => a.kind.localeCompare(b.kind) || a.qid.localeCompare(b.qid) || a.digest.localeCompare(b.digest),
	);
}

export function renderReport(reports: QuestionReport[]): string {
	const cell = (value: number | undefined) => (value === undefined ? "–" : value.toFixed(2));
	const escape = (text: string) => text.replace(/\|/g, "\\|");
	const lines = [
		"| kind | question | n | accuracy | majority | heuristic | chance | ECE | verdict |",
		"| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
	];
	for (const r of reports) {
		lines.push(
			`| ${escape(r.kind)} | ${escape(r.qid)}@${r.digest} | ${r.n} | ${cell(r.accuracy)} | ${cell(r.majority)} | ${cell(r.heuristic)} | ${cell(r.chance)} | ${cell(r.ece)} | ${r.verdict} |`,
		);
	}
	return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Calibration (temperature scaling)

/** Keyed `${qid}@${digest}`. */
export type Calibration = Record<
	string,
	{ temperature: number; n: number; nllBefore: number; nllAfter: number; fittedAt: string }
>;

const MIN_P = 1e-6;
const MAX_P = 1 - 1e-6;
const T_MIN = 0.25;
const T_MAX = 8;
const GOLDEN = (Math.sqrt(5) - 1) / 2;

const clampProbability = (p: number) => Math.min(MAX_P, Math.max(MIN_P, p));

/** p'_i ∝ p_i^(1/T), computed in log space. For two options this is sigmoid(logit(p)/T). */
function temperatureScale(probabilities: number[], temperature: number): number[] {
	const logits = probabilities.map(p => Math.log(clampProbability(p)) / temperature);
	const max = Math.max(...logits);
	const weights = logits.map(logit => Math.exp(logit - max));
	const total = weights.reduce((sum, w) => sum + w, 0);
	return weights.map(w => w / total);
}

/** The answer's distribution as parallel label/probability arrays; undefined when unusable. */
function answerDistribution(
	question: Question,
	answer: Answer | undefined,
): { labels: string[]; probabilities: number[] } | undefined {
	if (!answer) return undefined;
	if (question.type === "noul") {
		return isFiniteNumber(answer.noul) ? { labels: ["true", "false"], probabilities: [answer.noul, 1 - answer.noul] } : undefined;
	}
	const entries = Object.entries(answer.probabilities ?? {}).filter(([, p]) => isFiniteNumber(p));
	if (!entries.length) return undefined;
	return { labels: entries.map(([label]) => label), probabilities: entries.map(([, p]) => p) };
}

function meanNll(samples: Array<{ probabilities: number[]; target: number }>, temperature: number): number {
	let total = 0;
	for (const sample of samples) {
		total -= Math.log(clampProbability(temperatureScale(sample.probabilities, temperature)[sample.target]));
	}
	return total / samples.length;
}

/** NLL is convex in 1/T, hence unimodal in T: golden-section search is exact enough. */
function bestTemperature(samples: Array<{ probabilities: number[]; target: number }>): number {
	let lo = T_MIN;
	let hi = T_MAX;
	let a = hi - GOLDEN * (hi - lo);
	let b = lo + GOLDEN * (hi - lo);
	let fa = meanNll(samples, a);
	let fb = meanNll(samples, b);
	while (hi - lo > 1e-5) {
		if (fa <= fb) {
			hi = b;
			b = a;
			fb = fa;
			a = hi - GOLDEN * (hi - lo);
			fa = meanNll(samples, a);
		} else {
			lo = a;
			a = b;
			fa = fb;
			b = lo + GOLDEN * (hi - lo);
			fb = meanNll(samples, b);
		}
	}
	return (lo + hi) / 2;
}

export function fitCalibration(
	records: LogRecord[],
	options?: { minN?: number; source?: LabelSource; now?: Date },
): Calibration {
	const minN = options?.minN ?? DEFAULT_MIN_N;
	const fittedAt = (options?.now ?? new Date()).toISOString();
	const samplesByKey = new Map<string, Array<{ probabilities: number[]; target: number }>>();
	for (const { decision, qid, expected } of labeledExamples(records, { source: options?.source })) {
		const question = decision.questions[qid];
		const distribution = answerDistribution(question, decision.answers[qid]);
		if (!distribution) continue;
		// noul labels are "true"/"false"; choice labels and score levels stringify to their keys.
		const target = distribution.labels.indexOf(String(expected));
		if (target < 0) continue;
		const key = `${qid}@${question.digest}`;
		const samples = samplesByKey.get(key) ?? [];
		samples.push({ probabilities: distribution.probabilities, target });
		samplesByKey.set(key, samples);
	}

	const calibration: Calibration = {};
	for (const [key, samples] of samplesByKey) {
		if (samples.length < minN) continue;
		const temperature = bestTemperature(samples);
		const nllBefore = meanNll(samples, 1);
		const nllAfter = meanNll(samples, temperature);
		if (nllAfter < nllBefore - 1e-9) {
			calibration[key] = { temperature, n: samples.length, nllBefore, nllAfter, fittedAt };
		}
	}
	return calibration;
}

function argmax(labels: string[], probabilities: number[], preferred: unknown): string {
	const best = Math.max(...probabilities);
	const preferredIndex = labels.indexOf(String(preferred));
	if (preferredIndex >= 0 && probabilities[preferredIndex] === best) return labels[preferredIndex];
	return labels[probabilities.indexOf(best)];
}

/** Returns new answers with temperature scaling applied where an entry matches the current question digest. */
export function applyCalibration(
	questions: Record<string, Question>,
	answers: Record<string, Answer>,
	calibration: Calibration,
): Record<string, Answer> {
	const result: Record<string, Answer> = { ...answers };
	for (const [qid, answer] of Object.entries(answers)) {
		const question = questions[qid];
		if (!question) continue;
		const temperature = calibration[`${qid}@${questionDigest(question)}`]?.temperature;
		if (!isFiniteNumber(temperature) || temperature <= 0) continue;
		const distribution = answerDistribution(question, answer);
		if (!distribution) continue;
		const scaled = temperatureScale(distribution.probabilities, temperature);
		const answerConfidence = Math.max(...scaled);
		if (question.type === "noul") {
			result[qid] = { ...answer, noul: scaled[0], answer_confidence: answerConfidence, calibrated: true };
			continue;
		}
		const probabilities: Record<string, number> = {};
		distribution.labels.forEach((label, i) => {
			probabilities[label] = scaled[i];
		});
		const calibrated: Answer = { ...answer, probabilities, answer_confidence: answerConfidence, calibrated: true };
		if (question.type === "choice") calibrated.choice = argmax(distribution.labels, scaled, answer.choice);
		else calibrated.score = distribution.labels.reduce((sum, label, i) => sum + Number(label) * scaled[i], 0);
		result[qid] = calibrated;
	}
	return result;
}

/** The temperature is the only field applyCalibration reads; the rest is provenance. */
function hasUsableTemperature(entry: unknown): entry is Calibration[string] {
	return (
		entry !== null &&
		typeof entry === "object" &&
		"temperature" in entry &&
		isFiniteNumber(entry.temperature) &&
		entry.temperature > 0
	);
}

/** Missing or corrupt file → {}; entries without a usable temperature are dropped. */
export async function loadCalibration(file: string): Promise<Calibration> {
	let parsed: unknown;
	try {
		parsed = JSON.parse(await readFile(file, "utf8"));
	} catch {
		return {};
	}
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return {};
	const calibration: Calibration = {};
	for (const [key, entry] of Object.entries(parsed)) {
		if (hasUsableTemperature(entry)) calibration[key] = entry;
	}
	return calibration;
}

/** Atomic: written beside the target, then renamed over it. */
export async function saveCalibration(file: string, calibration: Calibration): Promise<void> {
	await mkdir(dirname(file), { recursive: true });
	const temp = `${file}.${process.pid}.${randomUUID()}.tmp`;
	await writeFile(temp, `${JSON.stringify(calibration, null, "\t")}\n`, "utf8");
	await rename(temp, file);
}

// ---------------------------------------------------------------------------
// Training export

export type TrainingRow = {
	state: string;
	questions: Record<string, Question>;
	gold: Record<string, { probabilities: Record<string, number> }>;
};

function goldProbabilities(question: Question, expected: Expected): Record<string, number> {
	if (question.type === "noul") return expected === true ? { true: 1, false: 0 } : { true: 0, false: 1 };
	const labels =
		question.type === "choice"
			? criteriaLabels(question)
			: Array.from({ length: Math.max(criteriaLabels(question).length, Number(expected) + 1) }, (_, i) => String(i));
	const target = String(expected);
	if (!labels.includes(target)) labels.push(target);
	const gold: Record<string, number> = {};
	for (const label of labels) gold[label] = label === target ? 1 : 0;
	return gold;
}

/** One row per labeled decision, restricted to its labeled questions, with one-hot gold distributions. */
export function exportTraining(records: LogRecord[], options?: { source?: LabelSource }): TrainingRow[] {
	const rows = new Map<string, TrainingRow>();
	for (const { decision, qid, expected } of labeledExamples(records, options)) {
		let row = rows.get(decision.id);
		if (!row) {
			row = { state: decision.state, questions: {}, gold: {} };
			rows.set(decision.id, row);
		}
		const { digest: _digest, ...question } = decision.questions[qid];
		row.questions[qid] = question;
		row.gold[qid] = { probabilities: goldProbabilities(question, expected) };
	}
	return [...rows.values()];
}
