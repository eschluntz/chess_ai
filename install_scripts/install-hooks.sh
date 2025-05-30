#!/bin/bash

# Install Git hooks
echo "Installing Git hooks..."
cp scripts/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
echo "Pre-push hook installed successfully!"