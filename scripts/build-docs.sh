#!/bin/bash

# MINTO Documentation Build Script
# Usage: ./scripts/build-docs.sh [en|ja|all|serve]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
SITE_DIR="$PROJECT_ROOT/_site"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

build_lang() {
    local lang=$1
    log "Building $lang documentation..."
    
    cd "$DOCS_DIR/$lang" || {
        error "Failed to change to $DOCS_DIR/$lang directory"
        return 1
    }
    
    if uv run jupyter-book build . 2>&1; then
        success "$lang documentation built successfully"
        echo "   Output: $DOCS_DIR/$lang/_build/html/index.html"
    else
        error "Failed to build $lang documentation"
        return 1
    fi
    
    # Check for error logs (but don't fail the build)
    ERROR_LOGS=$(find _build -type f -name "*.err.log" 2>/dev/null || true)
    if [ -n "$ERROR_LOGS" ]; then
        warning "Error logs found for $lang build (this is usually not critical):"
        while IFS= read -r log; do
            echo "[ERROR LOG] $log"
            cat "$log" 2>/dev/null || true
        done <<< "$ERROR_LOGS"
    fi
}

build_all() {
    log "Building all documentation versions..."
    
    # Build both versions
    build_lang "en"
    build_lang "ja"
    
    # Create integrated site
    log "Creating integrated site..."
    rm -rf "$SITE_DIR"
    mkdir -p "$SITE_DIR"
    
    # Copy redirect files
    cp -r "$DOCS_DIR/redirect/"* "$SITE_DIR/"
    
    # Copy built documentation
    cp -r "$DOCS_DIR/en/_build/html" "$SITE_DIR/en"
    cp -r "$DOCS_DIR/ja/_build/html" "$SITE_DIR/ja"
    
    success "Integrated site created at $SITE_DIR"
}

serve_site() {
    if [ ! -d "$SITE_DIR" ]; then
        log "Site not found. Building all versions first..."
        build_all
    fi
    
    log "Starting HTTP server at http://localhost:8000"
    log "Press Ctrl+C to stop the server"
    cd "$SITE_DIR"
    python -m http.server 8000
}

open_browser() {
    local url=$1
    log "Documentation built successfully!"
    log "To view: $url"
    
    # Try to open in browser, but don't fail if it doesn't work
    if command -v open > /dev/null 2>&1; then
        # macOS
        open "$url" 2>/dev/null || true
    elif command -v xdg-open > /dev/null 2>&1; then
        # Linux
        xdg-open "$url" 2>/dev/null || true
    fi
}

show_help() {
    echo "MINTO Documentation Build Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  en      Build English documentation only"
    echo "  ja      Build Japanese documentation only"
    echo "  all     Build both versions and create integrated site"
    echo "  serve   Build all and start HTTP server"
    echo "  clean   Clean all build artifacts"
    echo "  help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 en           # Build English docs and open in browser"
    echo "  $0 ja           # Build Japanese docs and open in browser"
    echo "  $0 serve        # Build all and serve at http://localhost:8000"
}

clean_builds() {
    log "Cleaning build artifacts..."
    rm -rf "$DOCS_DIR/en/_build"
    rm -rf "$DOCS_DIR/ja/_build"
    rm -rf "$SITE_DIR"
    success "Build artifacts cleaned"
}

# Main script logic
case "${1:-all}" in
    "en")
        build_lang "en"
        open_browser "$DOCS_DIR/en/_build/html/index.html"
        ;;
    "ja")
        build_lang "ja"
        open_browser "$DOCS_DIR/ja/_build/html/index.html"
        ;;
    "all")
        build_all
        log "To view the integrated site, run: $0 serve"
        ;;
    "serve")
        serve_site
        ;;
    "clean")
        clean_builds
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac