FROM ubuntu:latest
LABEL authors="samsonqian"

ENTRYPOINT ["top", "-b"]