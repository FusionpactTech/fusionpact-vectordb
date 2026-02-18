/**
 * FusionPact — Audit Logger
 *
 * Tracks all database operations for compliance, debugging, and observability.
 * - Who accessed what, when
 * - Insert, query, delete, forget operations
 * - GDPR-ready: audit trail for data access and deletion
 * - Configurable retention and export
 */

'use strict';

class AuditLogger {
  /**
   * @param {Object} [options]
   * @param {boolean} [options.enabled=true]
   * @param {number} [options.maxEntries=10000] — max entries before auto-pruning oldest
   * @param {boolean} [options.logToConsole=false]
   */
  constructor(options = {}) {
    this.enabled = options.enabled !== false;
    this.maxEntries = options.maxEntries || 10000;
    this.logToConsole = options.logToConsole || false;
    /** @type {AuditEntry[]} */
    this.entries = [];
    this._idCounter = 0;
  }

  /**
   * Log an audit event
   * @param {Object} event
   * @param {string} event.action — 'insert','query','delete','create_collection','drop_collection','remember','recall','learn','forget','rag_ingest','rag_search'
   * @param {string} [event.actor] — agent ID, tenant ID, or 'system'
   * @param {string} [event.collection] — collection name
   * @param {number} [event.documentCount] — number of docs affected
   * @param {Object} [event.details] — additional context
   * @returns {AuditEntry|null}
   */
  log(event) {
    if (!this.enabled) return null;

    const entry = {
      id: ++this._idCounter,
      timestamp: Date.now(),
      isoTime: new Date().toISOString(),
      action: event.action,
      actor: event.actor || 'anonymous',
      collection: event.collection || null,
      documentCount: event.documentCount || 0,
      details: event.details || null,
      durationMs: event.durationMs || null,
    };

    this.entries.push(entry);

    // Auto-prune if over limit
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-this.maxEntries);
    }

    if (this.logToConsole) {
      console.log(`[AUDIT] ${entry.isoTime} | ${entry.action} | actor=${entry.actor} | collection=${entry.collection} | docs=${entry.documentCount}`);
    }

    return entry;
  }

  /**
   * Query audit log
   * @param {Object} [filter]
   * @param {string} [filter.action] — filter by action type
   * @param {string} [filter.actor] — filter by actor
   * @param {string} [filter.collection] — filter by collection
   * @param {number} [filter.since] — timestamp, entries after this time
   * @param {number} [filter.until] — timestamp, entries before this time
   * @param {number} [filter.limit=100]
   * @returns {AuditEntry[]}
   */
  query(filter = {}) {
    let results = this.entries;

    if (filter.action) results = results.filter(e => e.action === filter.action);
    if (filter.actor) results = results.filter(e => e.actor === filter.actor);
    if (filter.collection) results = results.filter(e => e.collection === filter.collection);
    if (filter.since) results = results.filter(e => e.timestamp >= filter.since);
    if (filter.until) results = results.filter(e => e.timestamp <= filter.until);

    const limit = filter.limit || 100;
    return results.slice(-limit);
  }

  /**
   * Get audit statistics
   * @returns {Object}
   */
  getStats() {
    const actionCounts = {};
    const actorCounts = {};

    for (const e of this.entries) {
      actionCounts[e.action] = (actionCounts[e.action] || 0) + 1;
      actorCounts[e.actor] = (actorCounts[e.actor] || 0) + 1;
    }

    return {
      totalEntries: this.entries.length,
      oldestEntry: this.entries[0]?.isoTime || null,
      newestEntry: this.entries[this.entries.length - 1]?.isoTime || null,
      actionCounts,
      actorCounts,
    };
  }

  /**
   * Export audit log as JSON (for compliance/archival)
   * @param {Object} [filter] — same as query() filter
   * @returns {string} JSON string
   */
  export(filter) {
    const entries = filter ? this.query(filter) : this.entries;
    return JSON.stringify({
      exportedAt: new Date().toISOString(),
      entryCount: entries.length,
      entries,
    }, null, 2);
  }

  /**
   * Clear all audit entries
   * @returns {{cleared: number}}
   */
  clear() {
    const count = this.entries.length;
    this.entries = [];
    return { cleared: count };
  }
}

module.exports = { AuditLogger };
