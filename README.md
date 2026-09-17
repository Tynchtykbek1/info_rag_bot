# Messina Info

Tools for ingesting a Telegram JSON export into a normalized local dataset.

The repository intentionally excludes raw Telegram exports, secrets, databases,
and downloaded model or embedding caches. Tests use only synthetic fixtures.

## Usage

```python
from messina_info import load_telegram_export

export = load_telegram_export("path/to/result.json")
for message in export.messages:
    print(message.date, message.text)
```

Telegram rich-text fragments are flattened into plain text. Service records are
ignored, while malformed message records produce a `TelegramExportError`.
