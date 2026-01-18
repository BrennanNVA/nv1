#!/bin/bash
# Quick commit script for Nova Aetus
# Usage: ./commit.sh "commit message"
#        ./commit.sh              # Auto-generates message from changed files

# Don't exit on error - we want to handle pre-commit hook fixes
set +e

# Get commit message from argument or auto-generate
if [ -z "$1" ]; then
    # Auto-generate commit message from changed files
    CHANGED_FILES=$(git diff --cached --name-only 2>/dev/null || git diff --name-only 2>/dev/null)

    if [ -z "$CHANGED_FILES" ]; then
        echo "No changes to commit"
        exit 0
    fi

    # Count changes by type
    DOCS=$(echo "$CHANGED_FILES" | grep -E "\.md$|docs/" | wc -l)
    CODE=$(echo "$CHANGED_FILES" | grep -E "\.py$" | wc -l)
    CONFIG=$(echo "$CHANGED_FILES" | grep -E "\.toml$|\.env|config" | wc -l)

    # Generate message
    MSG_PARTS=()
    [ "$DOCS" -gt 0 ] && MSG_PARTS+=("docs: $DOCS file(s)")
    [ "$CODE" -gt 0 ] && MSG_PARTS+=("code: $CODE file(s)")
    [ "$CONFIG" -gt 0 ] && MSG_PARTS+=("config: $CONFIG file(s)")

    COMMIT_MSG="Update: $(IFS=', '; echo "${MSG_PARTS[*]}")"
else
    COMMIT_MSG="$1"
fi

# Stage all changes
git add -A

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo "No changes staged for commit"
    exit 0
fi

# Commit with message (pre-commit hooks will run and may fix files)
if ! git commit -m "$COMMIT_MSG"; then
    # If commit failed due to pre-commit fixes, re-stage and try again
    echo "Pre-commit hooks modified files, re-staging..."
    git add -A
    if ! git commit -m "$COMMIT_MSG"; then
        echo "❌ Commit failed. Please check the errors above."
        exit 1
    fi
fi

echo "✅ Committed: $COMMIT_MSG"

# Push to GitHub
echo "Pushing to GitHub..."
if git push; then
    echo "✅ Pushed to GitHub successfully"
else
    echo "⚠️  Push failed. You may need to pull first or check your remote configuration."
    echo "   Run 'git push' manually to retry."
    exit 1
fi
