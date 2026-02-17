/**
 * FusionPact — Vector Math Utilities
 * Optimized similarity/distance functions for dense vectors.
 */

'use strict';

/**
 * Dot product of two vectors
 * @param {Float64Array|number[]} a
 * @param {Float64Array|number[]} b
 * @returns {number}
 */
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * (b[i] || 0);
  return s;
}

/**
 * Magnitude (L2 norm) of a vector
 * @param {Float64Array|number[]} a
 * @returns {number}
 */
function magnitude(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

/**
 * Cosine similarity between two vectors (1 = identical, -1 = opposite)
 * @param {Float64Array|number[]} a
 * @param {Float64Array|number[]} b
 * @returns {number}
 */
function cosine(a, b) {
  const d = dot(a, b);
  const m = magnitude(a) * magnitude(b);
  return m === 0 ? 0 : d / m;
}

/**
 * Euclidean distance (L2) between two vectors
 * @param {Float64Array|number[]} a
 * @param {Float64Array|number[]} b
 * @returns {number}
 */
function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - (b[i] || 0)) ** 2;
  return Math.sqrt(s);
}

/**
 * Normalize a vector to unit length
 * @param {number[]} a
 * @returns {number[]}
 */
function normalize(a) {
  const m = magnitude(a);
  return m === 0 ? a : a.map(v => v / m);
}

/**
 * Generate a random unit vector of given dimension
 * @param {number} dim
 * @returns {number[]}
 */
function random(dim) {
  return normalize(Array.from({ length: dim }, () => Math.random() * 2 - 1));
}

/**
 * Compute similarity score based on metric. Higher = more similar.
 * For euclidean, returns negative distance so higher is still better.
 * @param {Float64Array|number[]} a
 * @param {Float64Array|number[]} b
 * @param {'cosine'|'euclidean'|'dot'} metric
 * @returns {number}
 */
function score(a, b, metric) {
  switch (metric) {
    case 'cosine': return cosine(a, b);
    case 'euclidean': return -euclidean(a, b);
    case 'dot': return dot(a, b);
    default: return cosine(a, b);
  }
}

/**
 * Convert a regular array to Float64Array for better performance
 * @param {number[]} arr
 * @returns {Float64Array}
 */
function toFloat64(arr) {
  return arr instanceof Float64Array ? arr : new Float64Array(arr);
}

module.exports = { dot, magnitude, cosine, euclidean, normalize, random, score, toFloat64 };
