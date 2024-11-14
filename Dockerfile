FROM python:3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
      parallel \
      && rm -rf /var/lib/apt/lists/*
RUN useradd -ms /bin/bash pipuser
USER pipuser

ENV LANG en_US.UTF-8
ENV PATH "$PATH:/home/pipuser/.local/bin" 
RUN pip install --user pipenv
WORKDIR /tmp/

ENTRYPOINT "./entrypoint.sh"

