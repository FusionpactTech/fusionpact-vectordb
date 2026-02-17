/**
 * FusionPact — Utility Functions
 */

'use strict';

const crypto = require('crypto');

/**
 * Generate a unique document ID
 * @param {string} [prefix='fp']
 * @returns {string}
 */
function generateId(prefix = 'fp') {
  return `${prefix}_${Date.now().toString(36)}_${crypto.randomBytes(4).toString('hex')}`;
}

/**
 * Recursive character text splitter with overlap.
 * Splits text into chunks of approximately `chunkSize` characters,
 * with `overlap` characters shared between consecutive chunks.
 *
 * @param {string} text
 * @param {Object} [options]
 * @param {number} [options.chunkSize=500]
 * @param {number} [options.overlap=100]
 * @param {string[]} [options.separators]
 * @returns {Array<{text: string, index: number, charStart: number, charEnd: number}>}
 */
function chunkText(text, options = {}) {
  const chunkSize = options.chunkSize || 500;
  const overlap = options.overlap || 100;
  const separators = options.separators || ['\n\n', '\n', '. ', '; ', ', ', ' '];

  const chunks = [];
  let charOffset = 0;

  function splitRecursive(txt, sepIdx) {
    if (txt.length <= chunkSize) {
      if (txt.trim()) {
        chunks.push({ text: txt.trim(), charStart: charOffset, charEnd: charOffset + txt.length });
      }
      return;
    }

    if (sepIdx >= separators.length) {
      // Hard split as last resort
      for (let i = 0; i < txt.length; i += chunkSize - overlap) {
        const c = txt.slice(i, i + chunkSize).trim();
        if (c) {
          chunks.push({ text: c, charStart: charOffset + i, charEnd: charOffset + i + c.length });
        }
      }
      return;
    }

    const sep = separators[sepIdx];
    const parts = txt.split(sep);
    let current = '';

    for (const part of parts) {
      const candidate = current ? current + sep + part : part;
      if (candidate.length > chunkSize && current) {
        chunks.push({ text: current.trim(), charStart: charOffset, charEnd: charOffset + current.length });
        // Overlap: keep end of current chunk
        const overlapText = current.slice(-overlap);
        charOffset += current.length - overlap;
        current = overlapText + part;
      } else {
        current = candidate;
      }
    }

    if (current.trim()) {
      chunks.push({ text: current.trim(), charStart: charOffset, charEnd: charOffset + current.length });
    }
  }

  splitRecursive(text, 0);

  return chunks.map((chunk, i) => ({
    text: chunk.text,
    index: i,
    charStart: chunk.charStart,
    charEnd: chunk.charEnd,
    charCount: chunk.text.length,
    wordCount: chunk.text.split(/\s+/).length,
  }));
}

/**
 * Simple timer utility
 */
class Timer {
  constructor() { this.start = performance.now(); }
  elapsed() { return +(performance.now() - this.start).toFixed(3); }
}

module.exports = { generateId, chunkText, Timer };
