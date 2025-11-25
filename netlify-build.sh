#!/bin/bash
# Netlify build script - static site only, no Python needed
echo "Building static site..."

# Temporarily hide Python files to prevent Netlify from detecting them
if [ -f "runtime.txt" ]; then
  mv runtime.txt runtime.txt.hidden
fi
if [ -f "requirements.txt" ]; then
  mv requirements.txt requirements.txt.hidden
fi

mkdir -p frontend/static
echo "Build complete - static files ready"

