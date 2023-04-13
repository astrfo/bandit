FROM python:3.9.13

ARG username=vscode
ARG useruid=1000
ARG usergid=${useruid}

RUN apt-get -y update && apt-get upgrade -qqy && apt-get -y install \
    sudo \
    bash \
    git \
    vim

RUN pip install --upgrade setuptools

RUN groupadd --gid ${usergid} ${username} && \
    useradd -s /bin/bash --uid ${useruid} --gid ${usergid} -m ${username} && \
    echo ${username}' ALL=NOPASSWD: ALL' >> /etc/sudoers

USER ${username}
WORKDIR /home/${username}

RUN pip install black