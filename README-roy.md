# (optional) build container
docker_build.sh

# start container
docker_run.sh

# inside container:

# optional: build trtllm from source
build_trtllm.sh

# install trtllm and make symlink
install_trtlllm.sh

# convert checkpoint
convert_checkpoint.sh

# build engine
build_engine.sh

# run inference
run_inference.sh
