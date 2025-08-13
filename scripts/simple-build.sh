#!/bin/bash
# Simple documentation build script for MINTO

echo "🚀 Building MINTO Documentation"
echo "================================"

# Build English documentation
echo "📖 Building English documentation..."
cd docs/en
uv run jupyter-book build .
echo "✅ English docs: docs/en/_build/html/index.html"
cd ../..

# Build Japanese documentation  
echo "📖 Building Japanese documentation..."
cd docs/ja
uv run jupyter-book build .
echo "✅ Japanese docs: docs/ja/_build/html/index.html"
cd ../..

# Create integrated site
echo "🔗 Creating integrated site..."
rm -rf _site
mkdir -p _site
cp -r docs/redirect/* _site/
cp -r docs/en/_build/html _site/en
cp -r docs/ja/_build/html _site/ja

echo ""
echo "🎉 Build complete!"
echo "   English:    file://$(pwd)/docs/en/_build/html/index.html"
echo "   Japanese:   file://$(pwd)/docs/ja/_build/html/index.html"
echo "   Integrated: file://$(pwd)/_site/index.html"
echo ""
echo "To serve locally: cd _site && python -m http.server 8000"