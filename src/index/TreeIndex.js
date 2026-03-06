/**
 * @fileoverview TreeIndex — Hierarchical Document Index for Reasoning-Based Retrieval
 * 
 * Builds a semantic tree structure from documents (similar to an intelligent
 * table of contents) and enables LLM-powered reasoning-based retrieval.
 * This provides the "vectorless" retrieval capability that complements
 * traditional vector search for structured professional documents.
 * 
 * Inspired by the insight that similarity ≠ relevance. For structured documents
 * (financial reports, legal filings, technical manuals), reasoning-based retrieval
 * dramatically outperforms vector similarity search.
 * 
 * @module index/TreeIndex
 * @author FusionPact Technologies Inc.
 * @license Apache-2.0
 * @see https://github.com/FusionpactTech/fusionpact-vectordb
 */

'use strict';

const { EventEmitter } = require('events');

/**
 * @typedef {Object} TreeNode
 * @property {string} nodeId - Unique node identifier
 * @property {string} title - Section title
 * @property {string} summary - AI-generated summary of this section
 * @property {number} startPage - Starting page (or chunk index)
 * @property {number} endPage - Ending page (or chunk index)
 * @property {number} level - Depth in the tree (0 = root)
 * @property {string} content - Raw text content of this section
 * @property {TreeNode[]} children - Child nodes
 * @property {Object} metadata - Additional metadata
 */

/**
 * @typedef {Object} ReasoningStep
 * @property {string} nodeId - Node being evaluated
 * @property {string} title - Node title
 * @property {string} reasoning - LLM's reasoning about this node's relevance
 * @property {number} relevanceScore - 0-1 score
 * @property {string} action - 'explore', 'retrieve', or 'skip'
 */

/**
 * @typedef {Object} TreeSearchResult
 * @property {TreeNode} node - The retrieved node
 * @property {string} content - The node's content
 * @property {number} relevanceScore - Reasoning-derived relevance score
 * @property {ReasoningStep[]} reasoningPath - Full reasoning trace
 * @property {string} citation - Human-readable citation path (e.g., "Section 3 > 3.2 Financial Data > Table 3.2.1")
 */

/**
 * @typedef {Object} LLMProvider
 * @property {function(string, Object): Promise<string>} complete - LLM completion function
 * @property {string} name - Provider name (e.g., 'ollama', 'openai', 'anthropic')
 */

class TreeIndex extends EventEmitter {
  /**
   * Create a new TreeIndex.
   * 
   * @param {Object} [config={}]
   * @param {LLMProvider} [config.llmProvider=null] - LLM provider for reasoning
   * @param {number} [config.maxTokensPerNode=20000] - Max tokens per tree node
   * @param {number} [config.maxDepth=5] - Maximum tree depth
   * @param {number} [config.maxChildrenPerNode=10] - Max children per node
   * @param {boolean} [config.generateSummaries=true] - Generate node summaries
   * 
   * @example
   * const tree = new TreeIndex({
   *   llmProvider: ollamaProvider,
   *   maxDepth: 4
   * });
   */
  constructor(config = {}) {
    super();
    this.llmProvider = config.llmProvider || null;
    this.maxTokensPerNode = config.maxTokensPerNode || 20000;
    this.maxDepth = config.maxDepth || 5;
    this.maxChildrenPerNode = config.maxChildrenPerNode || 10;
    this.generateSummaries = config.generateSummaries !== false;

    /** @private */
    this._documents = new Map();  // docId -> { tree, metadata, content }
    /** @private */
    this._nodeCount = 0;
  }

  /**
   * Index a document by building a hierarchical tree structure.
   * 
   * Analyzes the document's structure (headings, sections, paragraphs)
   * and builds a navigable tree. If an LLM provider is configured,
   * generates summaries for each node.
   * 
   * @param {string} docId - Unique document identifier
   * @param {string} content - Full document text
   * @param {Object} [options={}]
   * @param {string} [options.format='text'] - Input format: 'text', 'markdown', 'html'
   * @param {Object} [options.metadata={}] - Document metadata
   * @param {string} [options.title] - Document title (auto-detected if not provided)
   * @returns {Promise<TreeNode>} The root of the document tree
   * 
   * @example
   * const tree = await treeIndex.indexDocument('annual-report-2024', reportText, {
   *   format: 'markdown',
   *   metadata: { source: 'SEC Filing', year: 2024 }
   * });
   */
  async indexDocument(docId, content, options = {}) {
    const { format = 'text', metadata = {}, title = null } = options;

    this.emit('index:start', { docId });

    // Step 1: Parse document into sections
    const sections = this._parseDocument(content, format);

    // Step 2: Build hierarchical tree
    const tree = this._buildTree(sections, title || this._detectTitle(content));

    // Step 3: Generate summaries if LLM is available
    if (this.generateSummaries && this.llmProvider) {
      await this._generateSummaries(tree);
    }

    // Step 4: Store
    this._documents.set(docId, {
      tree,
      metadata: {
        ...metadata,
        indexedAt: new Date().toISOString(),
        nodeCount: this._countNodes(tree),
        format
      },
      content
    });

    this.emit('index:complete', { docId, nodeCount: this._countNodes(tree) });
    return tree;
  }

  /**
   * Perform reasoning-based retrieval over an indexed document.
   * 
   * The LLM navigates the tree top-down, reasoning at each level
   * about which branches are most relevant to the query, mimicking
   * how a human expert would navigate a document.
   * 
   * @param {string} docId - Document to search
   * @param {string} query - Natural language query
   * @param {Object} [options={}]
   * @param {number} [options.maxResults=5] - Maximum sections to retrieve
   * @param {number} [options.minRelevance=0.3] - Minimum relevance threshold
   * @param {boolean} [options.includeReasoning=true] - Include reasoning trace
   * @returns {Promise<TreeSearchResult[]>} Relevant sections with reasoning paths
   * 
   * @example
   * const results = await treeIndex.search('annual-report-2024', 
   *   'What were the total deferred tax assets in Q3?',
   *   { maxResults: 3 }
   * );
   * // Each result includes the content, relevance score, and full reasoning trace
   */
  async search(docId, query, options = {}) {
    const {
      maxResults = 5,
      minRelevance = 0.3,
      includeReasoning = true
    } = options;

    const doc = this._documents.get(docId);
    if (!doc) {
      throw new Error(`Document "${docId}" not indexed. Call indexDocument() first.`);
    }

    if (!this.llmProvider) {
      // Fallback: keyword-based tree traversal without LLM
      return this._keywordSearch(doc.tree, query, maxResults);
    }

    this.emit('search:start', { docId, query });

    const results = [];
    const reasoningPath = [];

    // Recursive tree search with reasoning
    await this._reasoningSearch(
      doc.tree, query, results, reasoningPath, maxResults, minRelevance, 0
    );

    // Sort by relevance
    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
    const topResults = results.slice(0, maxResults);

    if (!includeReasoning) {
      topResults.forEach(r => delete r.reasoningPath);
    }

    this.emit('search:complete', { docId, query, resultCount: topResults.length });
    return topResults;
  }

  /**
   * Search across ALL indexed documents.
   * 
   * @param {string} query - Natural language query
   * @param {Object} [options={}]
   * @param {number} [options.maxResults=10] - Max results across all docs
   * @param {string[]} [options.docIds] - Limit to specific documents
   * @returns {Promise<Array<TreeSearchResult & { docId: string }>>}
   */
  async searchAll(query, options = {}) {
    const { maxResults = 10, docIds = null } = options;
    const targetDocs = docIds || Array.from(this._documents.keys());

    const allResults = [];
    for (const docId of targetDocs) {
      const results = await this.search(docId, query, {
        maxResults: Math.ceil(maxResults / targetDocs.length),
        ...options
      });
      allResults.push(...results.map(r => ({ ...r, docId })));
    }

    allResults.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return allResults.slice(0, maxResults);
  }

  /**
   * Get the tree structure of an indexed document.
   * @param {string} docId
   * @returns {TreeNode|null}
   */
  getTree(docId) {
    const doc = this._documents.get(docId);
    return doc ? doc.tree : null;
  }

  /**
   * List all indexed documents.
   * @returns {Array<{docId: string, metadata: Object}>}
   */
  listDocuments() {
    const docs = [];
    for (const [docId, doc] of this._documents) {
      docs.push({ docId, metadata: doc.metadata });
    }
    return docs;
  }

  /**
   * Remove a document from the index.
   * @param {string} docId
   * @returns {boolean}
   */
  removeDocument(docId) {
    return this._documents.delete(docId);
  }

  /**
   * Export tree index data for persistence.
   * @returns {Object}
   */
  serialize() {
    const docs = {};
    for (const [docId, doc] of this._documents) {
      docs[docId] = {
        tree: doc.tree,
        metadata: doc.metadata,
        content: doc.content
      };
    }
    return { _version: 2, _engine: 'FusionPact', documents: docs };
  }

  /**
   * Import tree index data.
   * @param {Object} data
   */
  static deserialize(data, config = {}) {
    const index = new TreeIndex(config);
    for (const [docId, doc] of Object.entries(data.documents || {})) {
      index._documents.set(docId, doc);
    }
    return index;
  }

  // ─── Document Parsing ──────────────────────────────────

  /** @private */
  _parseDocument(content, format) {
    switch (format) {
      case 'markdown':
        return this._parseMarkdown(content);
      case 'html':
        return this._parseHTML(content);
      default:
        return this._parsePlainText(content);
    }
  }

  /** @private */
  _parseMarkdown(content) {
    const sections = [];
    const lines = content.split('\n');
    let currentSection = null;
    let buffer = [];

    for (const line of lines) {
      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);

      if (headingMatch) {
        // Save previous section
        if (currentSection) {
          currentSection.content = buffer.join('\n').trim();
          sections.push(currentSection);
        }

        const level = headingMatch[1].length;
        currentSection = {
          title: headingMatch[2].trim(),
          level,
          content: '',
          startLine: lines.indexOf(line)
        };
        buffer = [];
      } else {
        buffer.push(line);
      }
    }

    // Save last section
    if (currentSection) {
      currentSection.content = buffer.join('\n').trim();
      sections.push(currentSection);
    } else if (buffer.length > 0) {
      sections.push({
        title: 'Document',
        level: 1,
        content: buffer.join('\n').trim(),
        startLine: 0
      });
    }

    return sections;
  }

  /** @private */
  _parseHTML(content) {
    // Simple HTML heading extraction
    const sections = [];
    const headingRegex = /<h([1-6])[^>]*>(.*?)<\/h[1-6]>/gi;
    let match;
    let lastEnd = 0;

    while ((match = headingRegex.exec(content)) !== null) {
      const level = parseInt(match[1]);
      const title = match[2].replace(/<[^>]*>/g, '').trim();
      const start = match.index;

      if (sections.length > 0) {
        sections[sections.length - 1].content = this._stripHTML(
          content.substring(lastEnd, start)
        );
      }

      sections.push({
        title,
        level,
        content: '',
        startOffset: start
      });

      lastEnd = start + match[0].length;
    }

    if (sections.length > 0) {
      sections[sections.length - 1].content = this._stripHTML(
        content.substring(lastEnd)
      );
    } else {
      sections.push({
        title: 'Document',
        level: 1,
        content: this._stripHTML(content),
        startOffset: 0
      });
    }

    return sections;
  }

  /** @private */
  _parsePlainText(content) {
    // Heuristic: detect section patterns
    const lines = content.split('\n');
    const sections = [];
    let buffer = [];
    let currentTitle = 'Document';
    let currentLevel = 1;

    for (const line of lines) {
      // Detect section headers (numbered like "1.", "1.1", or ALL CAPS lines)
      const numberedHeader = line.match(/^(\d+(?:\.\d+)*)\s+(.+)$/);
      const capsHeader = line.match(/^([A-Z][A-Z\s]{3,})$/);

      if (numberedHeader || (capsHeader && line.length < 80)) {
        if (buffer.length > 0) {
          sections.push({
            title: currentTitle,
            level: currentLevel,
            content: buffer.join('\n').trim()
          });
          buffer = [];
        }

        if (numberedHeader) {
          currentTitle = numberedHeader[2].trim();
          currentLevel = numberedHeader[1].split('.').length;
        } else {
          currentTitle = capsHeader[1].trim();
          currentLevel = 1;
        }
      } else {
        buffer.push(line);
      }
    }

    if (buffer.length > 0 || sections.length === 0) {
      sections.push({
        title: currentTitle,
        level: currentLevel,
        content: buffer.join('\n').trim()
      });
    }

    return sections;
  }

  /** @private */
  _stripHTML(html) {
    return html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  }

  /** @private */
  _detectTitle(content) {
    const firstLine = content.split('\n')[0].trim();
    if (firstLine.length < 200) return firstLine;
    return 'Untitled Document';
  }

  // ─── Tree Building ──────────────────────────────────

  /** @private */
  _buildTree(sections, docTitle) {
    const root = {
      nodeId: this._nextNodeId(),
      title: docTitle,
      summary: '',
      level: 0,
      content: '',
      children: [],
      metadata: {}
    };

    if (sections.length === 0) return root;

    // Build tree from flat sections using level hierarchy
    const stack = [root];

    for (const section of sections) {
      const node = {
        nodeId: this._nextNodeId(),
        title: section.title,
        summary: '',
        level: section.level,
        content: section.content,
        children: [],
        metadata: {
          startLine: section.startLine,
          startOffset: section.startOffset,
          charCount: (section.content || '').length
        }
      };

      // Find parent: go up the stack until we find a node with lower level
      while (stack.length > 1 && stack[stack.length - 1].level >= section.level) {
        stack.pop();
      }

      stack[stack.length - 1].children.push(node);
      stack.push(node);
    }

    return root;
  }

  /** @private */
  _nextNodeId() {
    return `node_${String(this._nodeCount++).padStart(4, '0')}`;
  }

  /** @private */
  _countNodes(node) {
    let count = 1;
    for (const child of node.children || []) {
      count += this._countNodes(child);
    }
    return count;
  }

  // ─── Summary Generation ──────────────────────────────────

  /** @private */
  async _generateSummaries(node) {
    // Bottom-up: summarize children first, then use child summaries to summarize parent
    for (const child of node.children || []) {
      await this._generateSummaries(child);
    }

    if (node.content || node.children.length > 0) {
      const textForSummary = node.content
        || node.children.map(c => `${c.title}: ${c.summary}`).join('\n');

      if (textForSummary.length > 50) {
        try {
          node.summary = await this.llmProvider.complete(
            `Summarize the following section in 1-2 sentences. Section title: "${node.title}"\n\n${textForSummary.substring(0, 3000)}`,
            { maxTokens: 150 }
          );
        } catch (err) {
          node.summary = textForSummary.substring(0, 200) + '...';
        }
      }
    }
  }

  // ─── Reasoning-Based Search ──────────────────────────────────

  /** @private */
  async _reasoningSearch(node, query, results, reasoningPath, maxResults, minRelevance, depth) {
    if (results.length >= maxResults) return;
    if (depth > this.maxDepth) return;

    // For leaf nodes, evaluate relevance directly
    if (node.children.length === 0 && node.content) {
      const relevance = await this._evaluateRelevance(node, query);

      reasoningPath.push({
        nodeId: node.nodeId,
        title: node.title,
        reasoning: relevance.reasoning,
        relevanceScore: relevance.score,
        action: relevance.score >= minRelevance ? 'retrieve' : 'skip'
      });

      if (relevance.score >= minRelevance) {
        results.push({
          node: { nodeId: node.nodeId, title: node.title, level: node.level },
          content: node.content,
          relevanceScore: relevance.score,
          reasoningPath: [...reasoningPath],
          citation: this._buildCitation(node, reasoningPath)
        });
      }
      return;
    }

    // For branch nodes, reason about which children to explore
    if (node.children.length > 0) {
      const childEvaluations = await this._evaluateChildren(node, query);

      // Sort children by relevance and explore top candidates
      childEvaluations.sort((a, b) => b.score - a.score);

      for (const evaluation of childEvaluations) {
        if (evaluation.score < minRelevance) continue;
        if (results.length >= maxResults) break;

        const child = node.children.find(c => c.nodeId === evaluation.nodeId);
        if (!child) continue;

        reasoningPath.push({
          nodeId: child.nodeId,
          title: child.title,
          reasoning: evaluation.reasoning,
          relevanceScore: evaluation.score,
          action: 'explore'
        });

        await this._reasoningSearch(
          child, query, results, reasoningPath, maxResults, minRelevance, depth + 1
        );
      }
    }
  }

  /** @private */
  async _evaluateRelevance(node, query) {
    try {
      const prompt = `Given the query: "${query}"

Evaluate the relevance of this document section:
Title: ${node.title}
Content preview: ${(node.content || '').substring(0, 1500)}

Respond with ONLY a JSON object (no markdown):
{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}`;

      const response = await this.llmProvider.complete(prompt, { maxTokens: 100 });
      return JSON.parse(response.replace(/```json?|```/g, '').trim());
    } catch (err) {
      // Fallback: keyword overlap scoring
      return this._keywordRelevance(node, query);
    }
  }

  /** @private */
  async _evaluateChildren(node, query) {
    if (!this.llmProvider) {
      return node.children.map(child => ({
        nodeId: child.nodeId,
        score: this._keywordRelevance(child, query).score,
        reasoning: 'keyword match'
      }));
    }

    try {
      const childDescriptions = node.children.map((child, i) => 
        `[${i}] "${child.title}" - ${child.summary || '(no summary)'}`
      ).join('\n');

      const prompt = `Given the query: "${query}"

This document section "${node.title}" has the following subsections:
${childDescriptions}

Which subsections are most likely to contain the answer? 
Respond with ONLY a JSON array (no markdown):
[{"index": <number>, "score": <0.0-1.0>, "reasoning": "<brief>"}]`;

      const response = await this.llmProvider.complete(prompt, { maxTokens: 300 });
      const parsed = JSON.parse(response.replace(/```json?|```/g, '').trim());

      return parsed.map(item => ({
        nodeId: node.children[item.index]?.nodeId,
        score: item.score,
        reasoning: item.reasoning
      })).filter(item => item.nodeId);
    } catch (err) {
      return node.children.map(child => ({
        nodeId: child.nodeId,
        score: this._keywordRelevance(child, query).score,
        reasoning: 'fallback keyword match'
      }));
    }
  }

  // ─── Fallback Search ──────────────────────────────────

  /** @private */
  _keywordSearch(tree, query, maxResults) {
    const results = [];
    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);

    this._collectLeafNodes(tree, (node) => {
      const rel = this._keywordRelevance(node, query);
      if (rel.score > 0.1) {
        results.push({
          node: { nodeId: node.nodeId, title: node.title, level: node.level },
          content: node.content,
          relevanceScore: rel.score,
          citation: node.title
        });
      }
    });

    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return results.slice(0, maxResults);
  }

  /** @private */
  _keywordRelevance(node, query) {
    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
    const text = `${node.title} ${node.summary || ''} ${node.content || ''}`.toLowerCase();

    let matches = 0;
    for (const term of queryTerms) {
      if (text.includes(term)) matches++;
    }

    const score = queryTerms.length > 0 ? matches / queryTerms.length : 0;
    return { score, reasoning: `${matches}/${queryTerms.length} keyword matches` };
  }

  /** @private */
  _collectLeafNodes(node, callback) {
    if (node.children.length === 0) {
      callback(node);
    } else {
      for (const child of node.children) {
        this._collectLeafNodes(child, callback);
      }
    }
  }

  /** @private */
  _buildCitation(node, reasoningPath) {
    return reasoningPath
      .filter(step => step.action !== 'skip')
      .map(step => step.title)
      .join(' > ');
  }
}

module.exports = { TreeIndex };
