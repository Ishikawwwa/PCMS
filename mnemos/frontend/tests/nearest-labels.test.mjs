import assert from 'node:assert/strict';
import { cameraDistanceSquared, projectCameraPoint, nearestLabelCandidates } from '../src/nearest-labels.js';

const item = (id, kind, point, screen = [400, 300]) => ({
  n: { id, kind },
  p: { x: screen[0], y: screen[1], z: point[2], distance2: cameraDistanceSquared(point) },
});

// The node with smaller z can still be farther away because it is off-axis.
const offAxis = item('off-axis', 'episode', [600, 0, -300]);
const centered = item('centered', 'episode', [0, 0, -200]);
assert(offAxis.p.z < centered.p.z);
assert(offAxis.p.distance2 > centered.p.distance2);
assert.equal(nearestLabelCandidates([offAxis, centered], 800, 600, 1)[0].n.id, 'centered');

// A nearer trace is eligible even when semantic layers are present.
const trace = item('trace', 'trace', [0, 0, -400]);
assert.equal(nearestLabelCandidates([centered, trace], 800, 600, 1)[0].n.id, 'trace');

// Reversing the orbit reverses front/back without changing world-space length.
const front = cameraDistanceSquared([0, 0, -300]);
const back = cameraDistanceSquared([0, 0, 300]);
assert(front < back);

// Off-screen nodes do not consume the visible label budget; input is unmodified.
const outside = item('outside', 'concept', [0, 0, -800], [-5, 300]);
const input = [offAxis, outside, centered, trace];
assert.deepEqual(nearestLabelCandidates(input, 800, 600, 2).map(v => v.n.id), ['trace', 'centered']);
assert.equal(input[0], offAxis);
console.log('Nearest-label checks passed: full distance, all layers, orbit, viewport, budget.');

// Exercise the actual projection used by the renderer, not just sorting.
const foreground = projectCameraPoint([0, 0, -300], 860, 650, 1);
const background = projectCameraPoint([0, 0, 300], 860, 650, 1);
assert(foreground.p > background.p, 'foreground node must appear larger');
assert(foreground.distance2 < background.distance2);
assert.equal(nearestLabelCandidates([
  { n: { id: 'back' }, p: background },
  { n: { id: 'front' }, p: foreground },
], 860, 650, 1)[0].n.id, 'front');
const painted = [foreground, background].sort((a, b) => b.z - a.z);
assert.equal(painted.at(-1), foreground, 'foreground must paint last');
console.log('Projection integration passed: size, label distance, and drawing order agree.');
