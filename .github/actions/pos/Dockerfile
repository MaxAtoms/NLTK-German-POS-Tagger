FROM python:3.12
RUN useradd -u 1001 -m pipuser
USER pipuser

ENV LANG en_US.UTF-8
ENV PATH "$PATH:/github/home/.local/bin" 
WORKDIR /tmp/

ENTRYPOINT "./entrypoint.sh"

