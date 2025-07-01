# torchTextClassifiers Documentation

This directory contains the documentation generation system for torchTextClassifiers.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv sync --group docs
   ```

2. **Generate and serve documentation:**
   ```bash
   uv run python generate_docs.py --all
   ```

3. **Open your browser to http://localhost:8000**

## Commands

- `uv run python generate_docs.py --build` - Generate documentation
- `uv run python generate_docs.py --serve` - Serve on localhost:8000  
- `uv run python generate_docs.py --all` - Build and serve
- `uv run python generate_docs.py --clean` - Clean build directory
- `uv run python generate_docs.py --serve --port 8080` - Custom port

## Features

✅ **Auto-generated API docs** from docstrings  
✅ **Beautiful HTML theme** with navigation  
✅ **Architecture diagrams** and examples  
✅ **Search functionality**  
✅ **Mobile-responsive design**  
✅ **Copy-to-clipboard** code examples  
✅ **Fast local development** server  

## Documentation Structure

```
docs/
├── source/           # Sphinx source files
│   ├── conf.py      # Sphinx configuration
│   ├── index.rst    # Main documentation page
│   ├── api.rst      # Auto-generated API reference
│   ├── examples.rst # Usage examples and tutorials
│   ├── architecture.rst # Framework architecture
│   └── installation.rst # Installation guide
├── build/           # Generated documentation
│   └── html/        # HTML output
└── README.md        # This file
```

## Customization

The documentation system is highly customizable:

- **Themes:** Edit `source/conf.py` to change Sphinx theme
- **Styling:** Modify `source/_static/custom.css` for custom CSS
- **JavaScript:** Add features in `source/_static/custom.js`
- **Content:** Edit `.rst` files in `source/` directory

## Dependencies

Documentation dependencies are managed through uv and defined in `pyproject.toml`:

```toml
[dependency-groups]
docs = [
  "sphinx>=5.0.0",
  "sphinx-rtd-theme>=1.2.0",
  "sphinx-autodoc-typehints>=1.19.0",
  "sphinxcontrib-napoleon>=0.7",
  "sphinx-copybutton>=0.5.0",
  "myst-parser>=0.18.0",
  "sphinx-design>=0.3.0"
]
```

## Development Workflow

1. **Make changes** to docstrings in source code
2. **Rebuild docs** with `uv run python generate_docs.py --build`
3. **Preview changes** with `uv run python generate_docs.py --serve`
4. **Iterate** until satisfied

The documentation automatically extracts docstrings from the codebase, so keeping docstrings up-to-date will automatically improve the documentation.