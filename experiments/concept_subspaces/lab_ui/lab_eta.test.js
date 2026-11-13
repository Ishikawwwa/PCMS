"use strict";

const assert = require("node:assert/strict");
const {test} = require("node:test");
const {estimateFinish, summarizeUsage} = require("./lab.js");

const minute = 60_000;
const event = (time, kind, title, fields = {}) => ({at: new Date(time * minute).toISOString(), kind, title, ...fields});

test("paired estimate includes the other arm and subtracts time spent paused", () => {
  const run = {status: "running", manifest: {turns: 2, conditions: ["pcms", "mage"]}};
  const events = [event(0, "run", "Comparison arm started", {condition: "pcms"}),
    event(1, "control", "Paused at checkpoint"), event(5, "control", "Continued from checkpoint"),
    event(6, "run", "PCMS traveler turn ended", {condition: "pcms", turn: 0})];
  const eta = estimateFinish(run, events, 7 * minute);
  assert.equal(eta.finish, 12 * minute);
  assert.equal(eta.rough, true);
  assert.equal(estimateFinish({...run, status: "paused"}, events, 7 * minute), null);
});

test("diagnostic estimates use observed model-call durations and planned work", () => {
  const run = {status: "running", manifest: {planned_model_calls: {write: 2, read: 0, act: 0}}};
  const prompt = event(1, "prompt", "Polza completion request", {scope: {turn: "write"}});
  assert.deepEqual(estimateFinish(run, [prompt], 1.5 * minute), {state: "estimating"});
  const response = event(2, "response", "Polza completion response", {scope: {turn: "write"}});
  assert.equal(estimateFinish(run, [prompt, response], 2 * minute).finish, 3 * minute);
  const later = {...response, at: new Date(3 * minute).toISOString()};
  assert.deepEqual(estimateFinish(run, [prompt, response, later], 3 * minute), {state: "finishing"});
});

test("older unclosed pause intervals do not inflate the finish estimate", () => {
  const run = {status: "running", manifest: {turns: 2, conditions: ["pcms", "mage"]}};
  const events = [event(0, "run", "Comparison arm started", {condition: "pcms"}),
    event(1, "control", "Paused at checkpoint"),
    event(20, "run", "PCMS traveler turn ended", {condition: "pcms", turn: 0})];
  assert.deepEqual(estimateFinish(run, events, 21 * minute), {state: "estimating"});
});

test("benchmark ETA supports PCMS alone and missing progress", () => {
  const manifest = {benchmark: "travel", conditions: ["pcms"], workers: 2, travelers: 50,
    pilot_seconds_per_traveler: 60, started: new Date(0).toISOString()};
  const run = {status: "running", manifest};
  const now = 20 * minute;
  const progress = {splits: {all: {arms: {pcms: {travelers_scored: 20}}}}};
  assert.equal(estimateFinish(run, [], now, progress).remaining, 30 * minute);
  assert.equal(estimateFinish(run, [], now, {splits: {all: {arms: {}}}}).remaining, 25 * minute);
  assert.equal(estimateFinish(run, [], now).remaining, 25 * minute);
});

test("usage includes memory requests and retains unknown failed usage", () => {
  const usage = summarizeUsage([
    {operation: "agent", request_status: "completed", input_tokens: 10, output_tokens: 5, total_tokens: 15, cost_rub: 0.5},
    {operation: "extraction", request_status: "completed", input_tokens: 4, output_tokens: 3, total_tokens: 7, cost_rub: 0.2},
    {operation: "agent", request_status: "failed", total_tokens: null, cost_rub: null},
  ]);
  assert.equal(usage.requests, 3);
  assert.equal(usage.memoryCalls, 1);
  assert.equal(usage.failures, 1);
  assert.deepEqual(usage.totals.total_tokens, {sum: 22, known: 2});
  assert.deepEqual(usage.totals.cost_rub, {sum: 0.7, known: 2});
});
