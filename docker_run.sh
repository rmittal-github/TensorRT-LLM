#!/bin/bash
set -x
make -C docker run LOCAL_USER=1 IMAGE_NAME=t5tts CONTAINER_NAME=t5tts
