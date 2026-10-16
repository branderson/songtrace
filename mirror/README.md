# GitHub mirror

songtrace is developed on a private forge. `github-mirror.sh` publishes its
`main` and the tags on `main` to
[github.com/branderson/songtrace](https://github.com/branderson/songtrace),
hourly, from a host that can push to GitHub. Issues and pull requests are off
there.

The mirror's `main` is the forge's plus one commit, which adds a read-only notice
under the README's title. Every forge commit is in the mirror unchanged, and
each sync fast-forwards. If the README loses its `# ` title, the sync stops
instead of publishing without the notice.

## Setup

1. Set up the timer as described at the top of
   `songtrace-github-mirror.service`. Give it a deploy key with write access to
   the GitHub repository.
2. Run `github-mirror.sh --dry-run` to build what a sync would publish without
   pushing it.
3. Once the first sync has pushed, run `github-mirror.sh --settings` (needs
   `gh`). It turns off issues, pull requests, projects, wiki and discussions.

The timer runs a copy of the script, not the checkout's. Reinstall it after
reviewing a change to `mirror/`.
