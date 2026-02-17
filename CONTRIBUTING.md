# Contributing to FusionPact

We're thrilled you're interested in contributing to FusionPact! Here's how to get started.

## Quick Setup

```bash
git clone https://github.com/FusionPact/fusionpact-vectordb.git
cd fusionpact-vectordb
npm install
npm test          # Run test suite
npm run demo      # Run quickstart demo
npm run bench     # Run benchmarks
```

## Ways to Contribute

### 🐛 Bug Reports
Open an issue with: steps to reproduce, expected vs actual behavior, Node.js version.

### 🚀 Feature Requests
Open an issue describing the use case and proposed API.

### 📝 Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `npm test`
5. Commit with descriptive message: `git commit -m "feat: add XYZ"`
6. Push and open a PR

### Good First Issues

Look for issues tagged `good first issue` — they're designed for newcomers!

Priority areas:
- **LangChain integration** (`src/integrations/langchain.js`)
- **LlamaIndex integration** (`src/integrations/llamaindex.js`)
- **SQLite persistence** (`src/persistence/sqlite.js`)
- **Additional embedding providers**
- **Documentation improvements**
- **Test coverage expansion**

## Code Style

- Use `'use strict'` in all files
- JSDoc comments for all public methods
- Descriptive variable names
- Error handling with meaningful messages

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
