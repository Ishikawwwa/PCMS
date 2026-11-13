// Camera looks into the field from negative Z. This makes the lower/front
// part of the tilted field approach the viewer, consistently across projection,
// node size, painter order, hit testing, and label distance.
export const PERSPECTIVE_DISTANCE = 900;
export const DEPTH_SCALE = .72;
export const CAMERA_Z = -PERSPECTIVE_DISTANCE / DEPTH_SCALE;

export function cameraDistanceSquared([x, y, z]) {
  return x * x + y * y + (z - CAMERA_Z) ** 2;
}

export function projectCameraPoint([x, y, z], width, height, zoom) {
  const scale = zoom * Math.min(width / 860, height / 650) * 1.58;
  const perspective = PERSPECTIVE_DISTANCE / (PERSPECTIVE_DISTANCE + z * DEPTH_SCALE);
  return {
    x: width * .5 + x * scale * perspective,
    y: height * .51 + y * scale * perspective,
    z, p: perspective, distance2: cameraDistanceSquared([x, y, z]),
  };
}

export function nearestLabelCandidates(items, width, height, limit) {
  return items
    .filter(({ p }) => p.x >= 0 && p.x <= width && p.y >= 0 && p.y <= height)
    .sort((a, b) => a.p.distance2 - b.p.distance2 || a.n.id.localeCompare(b.n.id))
    .slice(0, limit);
}
