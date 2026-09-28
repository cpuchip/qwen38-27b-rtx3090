#!/usr/bin/env bash
# Sweep a repo for everything since our last post before posting anything there, so a post never ignores replies it
# has not read (a reply elsewhere often answers or overtakes it). Lists issue/PR comments, PR review comments and
# threads touched since SINCE, marks the threads that
# involve us (authored, commented, mentioned), and shows the default branch's new commits. It writes a stamp that
# post_comment.sh requires to be under an hour old.
#   bash scripts/pr-gate/sweep.sh [--repo syv-ai/HyperQwen] [--since 2026-09-24T00:00:00Z] [--me cpuchip]
# SINCE defaults to our latest comment in the repo (the last 30 days), else 7 days ago.
set -euo pipefail
REPO=syv-ai/HyperQwen; ME=cpuchip; SINCE=
while [ $# -gt 0 ]; do case "$1" in
  --repo) REPO=$2; shift 2 ;; --since) SINCE=$2; shift 2 ;; --me) ME=$2; shift 2 ;;
  *) echo "sweep: unknown argument $1"; exit 2 ;; esac; done
if [ -z "$SINCE" ]; then
  month=$(date -u -d '30 days ago' +%Y-%m-%dT%H:%M:%SZ)
  SINCE=$(gh api "repos/$REPO/issues/comments?since=$month&per_page=100" --paginate \
          --jq ".[] | select(.user.login==\"$ME\") | .created_at" | sort | tail -1)
  [ -n "$SINCE" ] || SINCE=$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ)
fi
mine=$(gh api "search/issues?q=repo:$REPO+involves:$ME&per_page=100" --jq '.items[].number' | sort -n | tr '\n' ' ')
echo "== sweep of $REPO since $SINCE (* = a thread involving $ME: $mine)"
echo "-- comments on issues and PRs:"
gh api "repos/$REPO/issues/comments?since=$SINCE&per_page=100" --paginate \
  --jq ".[] | select(.created_at > \"$SINCE\") | \"\(.created_at) \(.user.login) #\(.issue_url|split(\"/\")|last) \(.html_url|split(\"#\")|last)\"" \
  | sort | while read -r t who num rest; do
      n=${num#\#}; star=" "; case " $mine " in *" $n "*) star="*" ;; esac
      echo "$star $t $who $num $rest"; done
echo "-- PR review comments:"
gh api "repos/$REPO/pulls/comments?since=$SINCE&per_page=100" --paginate \
  --jq ".[] | select(.created_at > \"$SINCE\") | \"  \(.created_at) \(.user.login) PR#\(.pull_request_url|split(\"/\")|last) \(.path)\"" | sort
echo "-- threads touched (number, state, author, title):"
gh api "repos/$REPO/issues?state=all&since=$SINCE&per_page=100&sort=updated" --paginate \
  --jq '.[] | "#\(.number) [\(if .pull_request then (if .pull_request.merged_at then "PR merged" else "PR "+.state end) else "issue "+.state end)] \(.user.login): \(.title[0:90])"' \
  | sort -t'#' -k2 -n | while read -r num rest; do
      n=${num#\#}; star=" "; case " $mine " in *" $n "*) star="*" ;; esac
      echo "$star $num $rest"; done
echo "-- default branch commits:"
gh api "repos/$REPO/commits?since=$SINCE&per_page=50" --jq '.[] | "  \(.sha[0:7]) \(.commit.author.date[0:16]) \(.commit.message|split("\n")[0][0:90])"'
STAMP="${HOME}/.cache/pr-gate/sweep-${REPO//\//_}.stamp"
mkdir -p "$(dirname "$STAMP")"; date -u +%Y-%m-%dT%H:%M:%SZ > "$STAMP"
echo "== read what touches us, fold it into the post, answer what is owed. stamp: $STAMP"
