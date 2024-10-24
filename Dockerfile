FROM python:3.12
RUN useradd -ms /bin/bash pipuser
USER pipuser

ENV LANG en_US.UTF-8
ENV PATH "$PATH:/home/pipuser/.local/bin" 
RUN pip install --user pipenv
WORKDIR /tmp/

ENTRYPOINT "./entrypoint.sh"

