# 継続試験の実行手順と限界

Issue 59 の受入れ済み5ケースを、リポジトリ内の隔離テストとして移植した。

## 実行手順

pytestなどの開発用依存関係を導入したPython環境を有効にし、リポジトリ直下で次を実行する。`python`はその環境の実行ファイルを指す。

```bash
tmp_root=$(mktemp -d /tmp/hermes-issue59.XXXXXX)
mkdir -p "$tmp_root/home" "$tmp_root/hermes_home"
env -i PATH="$PATH" HOME="$tmp_root/home" HERMES_HOME="$tmp_root/hermes_home" HERMES_KANBAN_DB="$tmp_root/hermes_home/kanban.db" HERMES_TEST_ISOLATION="$tmp_root/hermes_home" LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/continuity_accepted/test_isolated_continuity_scenario.py tests/tui_gateway/test_kanban_notify_poller.py::TestCollectKanbanNotifications::test_two_tui_sessions_get_independent_claims_for_same_event -q --tb=short --junitxml=.local-evidence/issue59/issue59.xml
```

## 確認範囲

- モデル応答はローカルの OpenAI 互換スタブで確認する。
- TUI 通知 poller、GoalManager、SessionDB、Kanban DB、製品 gateway watcher は実コードを import して通す。
- `HERMES_HOME`、`HOME`、Kanban DB は一時ディレクトリへ固定する。
- 外部 API、実プロファイル、本番 gateway 再起動は使わない。

## 限界

DB に保存された応答行と TUI の `message.complete` 通知には、常に使える共通 ID がない。
そのため、このテストは同じセッション ID と応答本文、保存済み会話、通知カーソルで対応を確認する。
観測だけのために製品側へ新しい ID は追加しない。
