Aubot Assistant
=========

[![Docker Image CI](https://github.com/Peidon/aubot/actions/workflows/docker-image.yml/badge.svg?branch=master)](https://github.com/Peidon/aubot/actions/workflows/docker-image.yml)

<p align="center">
<img src="docs/Aubot.gif" width="200" height="200" />
</p>

How to use it ?
--

Run the container

```commandline
docker build -t aubot_server .
docker run --rm -p 8000:8000 aubot_server
```

Use the browser extension

| See [the guidebook](https://github.com/Peidon/kit/blob/master/internal/form-bot/guidebook.md). |
|------------------------------------------------------------------------------------------------|

Key features
--
Enter information once, and it can automatically populate the same fields in future interactions.  

- **Simple** -- No login or account creation required. No LLM token costs. Just install the extension and click buttons.
- **Fast**, -- The system uses a lightweight NLP model, with response times under 500 ms. LLMs are only used for offline tasks, making the experience both smart and fast.  
- **Secure** -- Your data is stored locally on your device. I do not collect any personal information.