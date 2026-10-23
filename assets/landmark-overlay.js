// Python returns normalized [x, y, z] arrays in its mirrored camera space.
(function (root) {
  const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],
    [13,17],[0,17],[17,18],[18,19],[19,20]];
  function map(landmarks, width, height, mirroredPreview = false) {
    if (!Array.isArray(landmarks) || landmarks.length !== 21 ||
        !landmarks.every(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite))) return [];
    return landmarks.map(([x, y]) => ({x: (mirroredPreview ? x : 1 - x) * width, y: y * height}));
  }
  function draw(ctx, landmarks, width, height, mirroredPreview = false) {
    const points = map(landmarks, width, height, mirroredPreview);
    if (!points.length) return 0;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.beginPath();
    connections.forEach(([a,b]) => {ctx.moveTo(points[a].x, points[a].y); ctx.lineTo(points[b].x, points[b].y);});
    ctx.strokeStyle = '#1f3549'; ctx.lineWidth = 4; ctx.stroke();
    ctx.strokeStyle = '#dcebf5'; ctx.lineWidth = 2; ctx.stroke();
    points.forEach(({x,y}) => {
      ctx.beginPath(); ctx.arc(x,y,3.5,0,Math.PI*2);
      ctx.fillStyle = '#f7fafc'; ctx.fill();
      ctx.strokeStyle = '#416d8e'; ctx.lineWidth = 1.5; ctx.stroke();
    });
    ctx.restore();
    return points.length;
  }
  const api = {map, draw, connections};
  if (typeof module !== 'undefined') module.exports = api;
  root.SignSyncLandmarks = api;
})(globalThis);
