FROM python:3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
      parallel \
      && rm -rf /var/lib/apt/lists/*
RUN useradd -u 1001 -m pipuser
USER pipuser

ENV LANG C

ENV PATH "$PATH:/github/home/.local/bin" 
WORKDIR /tmp/

ENTRYPOINT "./entrypoint.sh"

